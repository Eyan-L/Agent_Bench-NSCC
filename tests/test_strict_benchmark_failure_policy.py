import pytest

from agent_bench.core.spec import ConfigError
from agent_bench.core.strict_benchmark import validate_benchmark_counts


def test_strict_benchmark_policy_accepts_completed_verifier_outcomes():
    validate_benchmark_counts(10, 0)


def test_strict_benchmark_policy_rejects_any_infrastructure_error():
    with pytest.raises(ConfigError, match="9 completed, 1 error"):
        validate_benchmark_counts(9, 1)


def test_strict_benchmark_policy_rejects_all_error_cell():
    with pytest.raises(ConfigError, match="0 completed, 1 error"):
        validate_benchmark_counts(0, 1)
