"""Load a Hugging Face causal LM and run one chat completion (text → text)."""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _hf_hub_token() -> str | None:
    """Token from ``HF_TOKEN`` or ``HUGGING_FACE_HUB_TOKEN`` (e.g. after ``load_repo_dotenv``)."""
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return t if t else None


class HfCausalSvgGenerator:
    """Minimal chat completion for SVG+JSON extraction prompts."""

    def __init__(
        self,
        model_id: str,
        *,
        max_new_tokens: int = 8192,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        tok_kw: dict = {"trust_remote_code": True}
        hub_tok = _hf_hub_token()
        if hub_tok:
            tok_kw["token"] = hub_tok
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kw)
        kwargs: dict = {"trust_remote_code": True, "torch_dtype": dtype}
        if hub_tok:
            kwargs["token"] = hub_tok
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        if not torch.cuda.is_available():
            self.model = self.model.to(torch.device("cpu"))

    @property
    def device(self) -> torch.device:
        try:
            return torch.device(next(self.model.parameters()).device)
        except StopIteration:
            return torch.device("cpu")

    @torch.inference_mode()
    def chat(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        dev = self.device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
        gen = out[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True)
