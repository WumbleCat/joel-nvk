import pytest

from joel_nvk.data.collate import IGNORE_INDEX, encode_example
from joel_nvk.data.prompts import (
    Example,
    answer_letter,
    math_prompt,
    mmlu_prompt,
    render_prompt,
)


class FakeTokenizer:
    """Whitespace tokeniser: enough to test masking without loading a model."""

    def __init__(self, chat_template: str | None = None):
        self.chat_template = chat_template
        self.eos_token_id = 99

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [len(word) for word in text.split()]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return f"<|user|>{messages[0]['content']}<|assistant|>"


class TestPrompts:
    def test_math_prompt_carries_the_problem(self):
        assert "1 + 1" in math_prompt("1 + 1")

    def test_mmlu_prompt_letters_the_options(self):
        rendered = mmlu_prompt("Q?", ["first", "second", "third", "fourth"])
        assert "A. first" in rendered
        assert "D. fourth" in rendered

    def test_too_many_choices_raises(self):
        with pytest.raises(ValueError):
            mmlu_prompt("Q?", [str(i) for i in range(20)])

    def test_answer_letter(self):
        assert answer_letter(0) == "A"
        assert answer_letter(3) == "D"
        with pytest.raises(ValueError):
            answer_letter(99)

    def test_render_uses_chat_template_when_present(self):
        assert render_prompt(FakeTokenizer("tpl"), "hi").startswith("<|user|>")

    def test_render_falls_back_without_template(self):
        assert render_prompt(FakeTokenizer(None), "hi") == "hi\n\nAnswer:"


class TestEncoding:
    def test_prompt_tokens_are_masked_and_target_tokens_are_not(self):
        tokenizer = FakeTokenizer(None)
        example = Example(prompt="aa bb", target="cccc")
        encoded = encode_example(tokenizer, example, max_length=64)

        n_prompt = len(tokenizer("aa bb\n\nAnswer:")["input_ids"])
        assert encoded["labels"][:n_prompt] == [IGNORE_INDEX] * n_prompt
        assert IGNORE_INDEX not in encoded["labels"][n_prompt:]
        assert len(encoded["labels"]) == len(encoded["input_ids"])

    def test_target_ends_with_eos(self):
        tokenizer = FakeTokenizer(None)
        encoded = encode_example(tokenizer, Example("a", "b"), max_length=64)
        assert encoded["input_ids"][-1] == tokenizer.eos_token_id

    def test_prompt_longer_than_budget_is_dropped(self):
        tokenizer = FakeTokenizer(None)
        assert encode_example(tokenizer, Example("a b c d", "x"), max_length=2) is None
