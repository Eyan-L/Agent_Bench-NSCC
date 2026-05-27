"""
Claim/evidence registry for disciplined claim tracking.

Every quantitative claim must map to a metric, artifact path, and kind (measured/inferred).
"""

from typing import List, Dict, Optional


class ClaimRegistry:
    """
    Registry of claims with evidence pointers.

    Each claim has:
        - claim_text: The prose claim
        - metric: The metric name supporting it
        - artifact_path: Path to CSV/figure/table with the data
        - kind: "measured" or "inferred"
        - params: Optional dict of inference parameters
    """

    def __init__(self):
        self.claims: List[Dict] = []
        self.unsupported: List[Dict] = []

    def add_claim(
        self,
        claim_text: str,
        metric: str,
        artifact_path: str,
        kind: str,
        params: Optional[Dict] = None,
    ):
        """Register a claim with evidence."""
        self.claims.append(
            {
                "claim_text": claim_text,
                "metric": metric,
                "artifact_path": artifact_path,
                "kind": kind,
                "params": params or {},
            }
        )

    def flag_unsupported(self, claim_text: str, reason: str):
        """Flag a claim as unsupported."""
        self.unsupported.append({"claim_text": claim_text, "reason": reason})

    def get_unsupported(self) -> List[Dict]:
        """Return all unsupported claims."""
        return self.unsupported

    def to_dict(self) -> Dict:
        """Export to JSON-serializable dict."""
        return {"claims": self.claims, "unsupported": self.unsupported}

    def count(self) -> int:
        """Return number of registered claims."""
        return len(self.claims)
