// Package sampling offers down-sampling helpers used by the Azure and
// Alibaba replayers. The Alibaba replayer uses simple session-level random
// sampling; the Azure replayer additionally supports a Wasserstein-distance
// representative-sample approach ported from LLMLoadgen.
//
// The Wasserstein code here is intentionally free of plotting dependencies;
// LLMLoadgen's original implementation drew a distance-vs-fraction curve via
// gonum/plot, which we don't need for replay.
package sampling

import "math/rand"

// RandomIndices returns size distinct indices in [0, n) selected uniformly
// at random using the supplied seed. If size >= n, it returns every index
// in natural order (i.e. the identity permutation).
func RandomIndices(n, size int, seed int64) []int {
	if size <= 0 || n <= 0 {
		return nil
	}
	if size >= n {
		out := make([]int, n)
		for i := range out {
			out[i] = i
		}
		return out
	}
	r := rand.New(rand.NewSource(seed))
	perm := r.Perm(n)[:size]
	return perm
}

// RandomSubset picks size entries at random from in, preserving input
// order of the chosen entries. Equivalent to RandomIndices + index-lookup.
func RandomSubset[T any](in []T, size int, seed int64) []T {
	idx := RandomIndices(len(in), size, seed)
	out := make([]T, 0, len(idx))
	// RandomIndices returns a permutation slice; to keep input order we sort
	// indices before indexing back into the input.
	sorted := make([]int, len(idx))
	copy(sorted, idx)
	// Simple insertion sort: typical size ~ O(hundreds-thousands); avoids
	// pulling in sort.Slice reflection overhead for tiny sorts.
	for i := 1; i < len(sorted); i++ {
		for j := i; j > 0 && sorted[j-1] > sorted[j]; j-- {
			sorted[j-1], sorted[j] = sorted[j], sorted[j-1]
		}
	}
	for _, i := range sorted {
		out = append(out, in[i])
	}
	return out
}
