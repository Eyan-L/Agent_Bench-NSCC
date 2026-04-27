package sampling

import (
	"log"
	"math"
	"math/rand"
	"sort"
)

// PromptSizeFn extracts a scalar "prompt size" per entry. For Azure traces
// this is typically len(data["prompt"]) (byte length if string, element
// count if []any).
type PromptSizeFn[T any] func(T) float64

// WassersteinDownsample is a bit-for-bit reimplementation of LLMLoadgen's
// `YieldTargetSample -> RecursiveRollDownSampling`, minus the gonum/plot
// visualization. For identical inputs + parameters + per-entry prompt sizes
// it selects exactly the same subset as /LLMLoadgen.
//
// Behavior faithfully matched:
//
//  1. The initial "previousSample" is `rand.Perm(N)` seeded by the master
//     seed. LLMLoadgen gets this via `SampleData(data, N, randomSeed)` —
//     the `size > len(data)` shortcut does NOT fire when size == N, so the
//     full Perm runs.
//
//  2. Step size is `int(N * 0.01)`. Each iteration spawns `trials`
//     candidates via `SampleData(previousSample, size, int64(i))`. The trial
//     seed is hard-coded to the trial index — the master seed does NOT
//     propagate into trial RNGs.
//
//  3. The candidate with the smallest Wasserstein distance to
//     previousSample (on prompt-size distributions) wins. Ties are broken
//     by earlier-trial-wins, matching LLMLoadgen's strict `<`.
//
//  4. Results are stored in a map keyed by `round(size/N, 2)`. The final
//     return is `results[fraction]` — **a raw lookup, not a nearest-key
//     search.** If `fraction` isn't already a 0.01-aligned value that
//     coincides with one of the loop's stopping sizes, the lookup MISSES
//     and the function returns nil. LLMLoadgen has the same behavior; we
//     print a warning so the caller can tell what happened.
//
//  5. For N < 100 the LLMLoadgen loop has stepSize=0 and spins forever.
//     We explicitly guard against this and return the input unchanged with
//     a warning rather than hang.
func WassersteinDownsample[T any](in []T, fraction float64, trials int, seed int64, sizeOf PromptSizeFn[T]) []T {
	N := len(in)
	if N == 0 || fraction >= 1.0 {
		return in
	}
	if fraction <= 0 {
		return nil
	}
	if trials <= 0 {
		trials = 10
	}

	minSize := int(float64(N) * fraction)
	stepSize := int(float64(N) * 0.01)
	if stepSize == 0 {
		log.Printf("wasserstein downsample: dataset too small (N=%d, stepSize=0) — returning input unchanged to avoid infinite loop", N)
		return in
	}
	if minSize <= 0 {
		return nil
	}

	// Cache per-entry prompt sizes once.
	sizes := make([]float64, N)
	for i, e := range in {
		sizes[i] = sizeOf(e)
	}
	indexSize := func(idx []int) []float64 {
		out := make([]float64, len(idx))
		for i, j := range idx {
			out[i] = sizes[j]
		}
		return out
	}

	// Step 1 — base order: Perm(N, master seed). Matches LLMLoadgen's
	// SampleData(data, N, randomSeed) which runs a full permutation because
	// the shortcut is size > len(data), not size >= len(data).
	baseRng := rand.New(rand.NewSource(seed))
	currentSample := baseRng.Perm(N)

	results := map[float64][]int{
		1.0: currentSample,
	}
	previousSample := currentSample

	// Step 2 — roll down one step at a time, picking the representative
	// candidate at each step.
	for size := N - stepSize; size >= minSize; size -= stepSize {
		// LLMLoadgen initializes bestSample as nil and only overwrites on
		// strict `<`. Since bestDist starts at +Inf, trial 0 always wins,
		// so bestSample is guaranteed non-nil after the inner loop.
		var bestSample []int
		bestDist := math.Inf(1)

		for trial := 0; trial < trials; trial++ {
			r := rand.New(rand.NewSource(int64(trial)))
			perm := r.Perm(len(previousSample))[:size]
			cand := make([]int, size)
			for k, p := range perm {
				cand[k] = previousSample[p]
			}
			d := computeWassersteinDistance(indexSize(previousSample), indexSize(cand))
			if d < bestDist {
				bestDist = d
				bestSample = cand
			}
		}

		// Key: round(size/N, 2). Exact port of LLMLoadgen's
		// roundToTwoDecimals(float64(size)/float64(totalEntries)).
		key := math.Round(float64(size)/float64(N)*100) / 100
		results[key] = bestSample
		previousSample = bestSample
	}

	// Step 3 — final lookup by RAW fraction. LLMLoadgen does
	// `finalSample = samples[minFraction]`; the rounded-key vs. raw-
	// fraction mismatch means this returns nil whenever `fraction` isn't
	// exactly the rounded value of some visited size/N. Warn so callers
	// don't chase a silent empty result.
	finalIdx, ok := results[fraction]
	if !ok {
		log.Printf("wasserstein downsample: no result stored at fraction=%g (N=%d, visited_keys=%v) — returning empty to match LLMLoadgen", fraction, N, sortedKeys(results))
		return nil
	}

	log.Printf("wasserstein downsample: %d -> %d (fraction=%.3g, trials=%d, seed=%d)", N, len(finalIdx), fraction, trials, seed)

	// Preserve original input order on output. Caller typically sorts by
	// timestamp afterward; LLMLoadgen sorts in YieldTargetSample itself.
	// Either way, the SET we return equals LLMLoadgen's — that's the
	// invariant callers rely on.
	sort.Ints(finalIdx)
	out := make([]T, 0, len(finalIdx))
	for _, i := range finalIdx {
		out = append(out, in[i])
	}
	return out
}

func sortedKeys(m map[float64][]int) []float64 {
	ks := make([]float64, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	sort.Float64s(ks)
	return ks
}

// computeWassersteinDistance returns the 1-D Wasserstein distance between
// the empirical distributions of x and y. Algorithm mirrors
// LLMLoadgen/frontend/sampler.go#computeWassersteinDistance verbatim.
func computeWassersteinDistance(x, y []float64) float64 {
	if len(x) == 0 || len(y) == 0 {
		return 0.0
	}
	xs := append([]float64(nil), x...)
	ys := append([]float64(nil), y...)
	sort.Float64s(xs)
	sort.Float64s(ys)

	all := append([]float64(nil), xs...)
	all = append(all, ys...)
	sort.Float64s(all)

	var (
		distance       float64
		xIndex, yIndex int
	)
	for i := 1; i < len(all); i++ {
		delta := all[i] - all[i-1]
		for xIndex < len(xs) && xs[xIndex] <= all[i-1] {
			xIndex++
		}
		for yIndex < len(ys) && ys[yIndex] <= all[i-1] {
			yIndex++
		}
		xCDF := float64(xIndex) / float64(len(xs))
		yCDF := float64(yIndex) / float64(len(ys))
		distance += math.Abs(xCDF-yCDF) * delta
	}
	return distance
}
