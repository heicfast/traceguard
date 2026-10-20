"""Generic Transformers chat adapter with explicit reasoning semantics.

Unlike the Qwen3 adapter, this class never assumes that an untagged output is
private reasoning.  Only model-emitted think delimiters are admitted to the
reasoning channel; otherwise the text remains the visible assistant response.
This fail-open-to-observation rule prevents tool calls from disappearing when
a third-party chat template changes.
"""

from __future__ import annotations

from typing import Literal

from .interface import GenerationSettings, split_reasoning
from .qwen import QwenHFProvider


class GenericHFProvider(QwenHFProvider):
    provider_name = "generic_hf"

    def __init__(
        self,
        *args,
        reasoning_protocol: Literal["think_tags", "none"] = "none",
        **kwargs,
    ):
        self.reasoning_protocol = reasoning_protocol
        super().__init__(*args, **kwargs)

    def _render_prompt(
        self, messages: list[dict[str, str]], settings: GenerationSettings
    ) -> str:
        # Generic chat templates do not share Qwen's enable_thinking keyword.
        # The exact rendered template is retained in the privileged evidence.
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _split_generated_output(
        self,
        new_ids: list[int],
        decoded_output: str,
        settings: GenerationSettings,
    ) -> tuple[str | None, str, str]:
        del new_ids
        if self.reasoning_protocol == "think_tags":
            # reasoning_mode=False still parses an explicitly emitted block;
            # it merely declines Qwen's unsafe "all untagged text is thought"
            # fallback.
            turn = split_reasoning(decoded_output, reasoning_mode=False)
            return turn.reasoning_text, turn.assistant_text, turn.reasoning_parse_status
        return None, decoded_output.strip("\n"), "absent"


__all__ = ["GenericHFProvider"]
