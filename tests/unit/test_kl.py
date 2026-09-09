"""KL estimator tests.

The KL probe is the one number in this pipeline that cannot be eyeballed, so its
alignment and its zero point are pinned down here.
"""

from contextlib import contextmanager

import pytest

torch = pytest.importorskip("torch")

from joel_nvk.core.kl import kl_between_states  # noqa: E402


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0


class FakeStatefulModel:
    """Returns per-state logits of shape [1, T, V], ignoring the input tokens."""

    def __init__(self, logits_by_state: dict[str, "torch.Tensor"]):
        self._logits = logits_by_state
        self._state = "base"

    def parameters(self):
        yield torch.zeros(1)

    def __call__(self, input_ids=None, attention_mask=None):
        class Output:
            logits = self._logits[self._state]

        return Output()

    def set_adapter(self, name: str) -> None:
        self._state = name

    @contextmanager
    def disable_adapter(self):
        previous, self._state = self._state, "base"
        try:
            yield self
        finally:
            self._state = previous


def _probe(n_prompt: int, n_continuation: int):
    return [
        {
            "prompt_ids": list(range(1, n_prompt + 1)),
            "continuation_ids": list(range(1, n_continuation + 1)),
        }
    ]


def test_identical_states_have_zero_kl():
    logits = torch.randn(1, 6, 5)
    model = FakeStatefulModel({"base": logits, "a": logits.clone()})

    result = kl_between_states(model, FakeTokenizer(), _probe(4, 2), [("a", "base")])

    assert result["a||base"]["kl"] == pytest.approx(0.0, abs=1e-6)
    assert result["a||base"]["n_tokens"] == 2


def test_kl_matches_the_hand_computed_value():
    # One continuation token. policy = [0.25, 0.75], reference = [0.5, 0.5].
    n_prompt, n_cont = 3, 1
    width = n_prompt + n_cont
    policy = torch.zeros(1, width, 2)
    policy[0, n_prompt - 1] = torch.tensor([0.0, torch.tensor(3.0).log().item()])
    reference = torch.zeros(1, width, 2)

    model = FakeStatefulModel({"base": reference, "a": policy})
    result = kl_between_states(model, FakeTokenizer(), _probe(n_prompt, n_cont), [("a", "base")])

    expected = 0.25 * torch.tensor(0.5).log() + 0.75 * torch.tensor(1.5).log()
    assert result["a||base"]["kl"] == pytest.approx(float(expected), abs=1e-6)


def test_prompt_positions_are_never_scored():
    """Wildly different logits over the prompt must not move the KL."""
    n_prompt, n_cont = 5, 2
    width = n_prompt + n_cont
    shared = torch.zeros(1, width, 3)

    poisoned = shared.clone()
    poisoned[0, : n_prompt - 1] = torch.tensor([50.0, -50.0, 0.0])

    model = FakeStatefulModel({"base": shared, "a": poisoned})
    result = kl_between_states(model, FakeTokenizer(), _probe(n_prompt, n_cont), [("a", "base")])

    assert result["a||base"]["kl"] == pytest.approx(0.0, abs=1e-6)


def test_direction_is_policy_against_reference():
    """KL is asymmetric, so swapping the pair must change the number."""
    n_prompt, n_cont = 3, 1
    width = n_prompt + n_cont
    skewed = torch.zeros(1, width, 3)
    skewed[0, n_prompt - 1] = torch.tensor([3.0, 0.0, 0.0])
    flat = torch.zeros(1, width, 3)

    model = FakeStatefulModel({"base": flat, "a": skewed})
    forward = kl_between_states(model, FakeTokenizer(), _probe(n_prompt, n_cont), [("a", "base")])
    backward = kl_between_states(model, FakeTokenizer(), _probe(n_prompt, n_cont), [("base", "a")])

    assert forward["a||base"]["kl"] > 0
    assert backward["base||a"]["kl"] > 0
    assert forward["a||base"]["kl"] != pytest.approx(backward["base||a"]["kl"], abs=1e-3)
