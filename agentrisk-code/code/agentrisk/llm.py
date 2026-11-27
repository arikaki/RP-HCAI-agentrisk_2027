"""Thin client for any OpenAI-compatible chat endpoint.

Works with a local vLLM server, llama.cpp's llama-server, Ollama
(http://localhost:11434/v1), or a hosted API such as OpenRouter. Also has a
"mock" backend that returns random valid answers, for testing the pipeline
without a GPU.
"""

from __future__ import annotations

import json
import random
import re
import time
import zlib

from .labels import HARM_DOMAINS, RISK_DRIVERS, RISK_FACTORS

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_json(text: str) -> dict | None:
    """Return the first JSON object in `text` (ignores <think> blocks and code fences)."""
    if not text:
        return None
    text = _THINK_RE.sub("", text)
    text = text.replace("```json", "```")
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None


class ChatClient:
    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1",
                 api_key: str = "EMPTY", temperature: float = 0.0, max_tokens: int = 1024,
                 disable_thinking: bool = False, json_mode: bool = False, seed: int | None = None,
                 reasoning_effort: str | None = None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        # Ollama ignores chat_template_kwargs; "none" turns thinking off there.
        self.reasoning_effort = reasoning_effort
        self.extra_body: dict = {}
        if disable_thinking:
            # Understood by vLLM / SGLang chat templates (e.g. Qwen3-style models).
            self.extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        self.json_mode = json_mode
        self.mock = model == "mock"
        if not self.mock:
            from openai import OpenAI

            self.client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, messages: list[dict], temperature: float | None = None,
             retries: int = 3) -> str:
        if self.mock:
            return _mock_answer(messages)
        kwargs = dict(model=self.model, messages=messages,
                      temperature=self.temperature if temperature is None else temperature,
                      max_tokens=self.max_tokens)
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as exc:  # network errors, timeouts, server overload
                last_err = exc
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")


def _mock_answer(messages: list[dict]) -> str:
    """Random but valid output, so the whole pipeline can be tested offline."""
    rng = random.Random(zlib.crc32(messages[-1]["content"].encode("utf-8")))
    if "Rate each factor" in messages[-1]["content"]:
        return json.dumps({k: {"rating": rng.randint(1, 5), "reason": "mock"} for k in RISK_FACTORS})
    sev = rng.randint(1, 5)
    dom = ["No Material Harm"] if sev == 1 else rng.sample(HARM_DOMAINS[:-1], rng.randint(1, 2))
    drv = ["No Material Risk"] if sev == 1 else rng.sample(RISK_DRIVERS[:-1], rng.randint(1, 2))
    return "<think>mock reasoning</think>" + json.dumps(
        {"rationale": "mock", "risk_severity": sev, "risk_domains": dom, "risk_drivers": drv})
