import math

import pytest

from examples.quickstart import run_quickstart


@pytest.mark.parametrize("recipe", ["ze-eden", "zip-sr"])
def test_quickstart_runs_with_packed_optimizer_state(recipe: str) -> None:
    summary = run_quickstart(recipe, steps=1)

    assert math.isfinite(float(summary["final_loss"]))
    assert summary["parameters_updated"] is True
    assert int(summary["packed_state_tensors"]) > 0


@pytest.mark.parametrize("steps", [0, -1])
def test_quickstart_rejects_non_positive_steps(steps: int) -> None:
    with pytest.raises(ValueError, match="steps must be positive"):
        run_quickstart(steps=steps)
