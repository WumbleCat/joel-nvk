"""State switching between base and named adapters.

The base state must be reachable on a model that never had an adapter attached.
transformers models carry their own ``set_adapter``/``disable_adapters``, so
detecting PEFT by duck-typing on ``set_adapter`` silently sends a plain base
model down the adapter path — that is the bug these tests pin down.
"""

from contextlib import contextmanager

import pytest

from joel_nvk.models.loading import adapter_state


class PlainModel:
    """A transformers-style model: has set_adapter, has no PEFT wrapper."""

    def set_adapter(self, name):  # pragma: no cover - must never be called
        raise AssertionError("set_adapter called on a model with no adapters")

    def disable_adapters(self):  # pragma: no cover - must never be called
        raise AssertionError("disable_adapters called on a model with no adapters")


class PeftLikeModel:
    def __init__(self):
        self.state = "math"
        self.active_adapters = ["math"]
        self.disabled = False

    def set_adapter(self, name):
        self.state = name
        self.active_adapters = [name]

    @contextmanager
    def disable_adapter(self):
        self.disabled = True
        try:
            yield self
        finally:
            self.disabled = False


def test_base_state_on_a_plain_model_is_a_no_op():
    model = PlainModel()
    with adapter_state(model, "base") as yielded:
        assert yielded is model


def test_named_state_on_a_plain_model_is_refused():
    with pytest.raises(ValueError, match="no adapters"):
        with adapter_state(PlainModel(), "math"):
            pass


def test_base_state_disables_adapters():
    model = PeftLikeModel()
    with adapter_state(model, "base"):
        assert model.disabled
    assert not model.disabled


def test_named_state_activates_and_restores():
    model = PeftLikeModel()
    with adapter_state(model, "mmlu"):
        assert model.state == "mmlu"
    assert model.state == "math"
