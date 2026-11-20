"""LLM backends. Anything with a `generate(messages, max_new_tokens) -> str` method works."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}

DEFAULT_LLM = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@runtime_checkable
class LLM(Protocol):
    def generate(self, messages: list[Message], max_new_tokens: int = 300) -> str: ...


class HFTransformersLLM:
    """A local Hugging Face chat model, optionally loaded in 4-bit.

    `load_in_4bit=True` needs an NVIDIA GPU and the `[quant]` extra. Generation is
    greedy, so the same input gives the same output on the same setup.
    """

    def __init__(
        self,
        model: str = DEFAULT_LLM,
        *,
        load_in_4bit: bool = False,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model, quantization_config=quant, device_map="auto"
            )
        else:
            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = AutoModelForCausalLM.from_pretrained(model).to(device)

    def generate(self, messages: list[Message], max_new_tokens: int = 300) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class OpenAICompatibleLLM:
    """Any server with an OpenAI-style `/chat/completions` endpoint (vLLM, llama.cpp, ...).

    `base_url` includes the API prefix, for example `http://localhost:8000/v1`.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(f"base_url must start with http:// or https://, got {base_url!r}")
        self.base_url = base_url.rstrip("/")
        self.name = model
        self.api_key = api_key
        self.timeout = timeout

    def generate(self, messages: list[Message], max_new_tokens: int = 300) -> str:
        body = {
            "model": self.name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": 0,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"LLM server returned HTTP {err.code}: {detail}") from err
        except urllib.error.URLError as err:
            raise RuntimeError(
                f"could not reach LLM server at {self.base_url}: {err.reason}"
            ) from err
        try:
            return payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError) as err:
            raise RuntimeError(
                f"unexpected response from LLM server: {str(payload)[:500]}"
            ) from err
