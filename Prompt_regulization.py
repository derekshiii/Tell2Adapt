"""Context-Aware Prompts Regularization (CAPR) for Tell2Adapt.

The paper applies CAPR once per dataset-level prompt list:

1. an LLM reads the complete list and infers the global anatomical site and
   modality;
2. every sub-prompt is corrected and rewritten as
   ``[Target] in [Anatomical Site] [Modality]``;
3. the canonical prompts are returned as a ``[SEP]``-separated list and parsed
   back to labels before being injected into the BiomedParse NPZ files.

This module implements steps 1-3. Prompt injection is handled by
``inject_capr_prompts.py``; the model forward pass intentionally does not call
an LLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import httpx
from openai import OpenAI

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_BASE_URL = None
DEFAULT_META_PROMPT = Path(__file__).with_name("Meta-Prompt.txt")
DEFAULT_TIMEOUT = 120.0
DEFAULT_VALIDATION_RETRIES = 3
DEFAULT_MAX_TOKENS = 4096

# The paper's canonical format starts with the target, not with an instruction
# verb. Rejecting these prefixes catches the paraphrase templates that were
# previously injected into the NPZ files.
_BANNED_PREFIXES = (
    "segmentation of ",
    "visualization of ",
    "characterization of ",
    "delineation of ",
    "identification of ",
    "localization of ",
    "post-contrast ",
    "mr imaging of ",
    "mri imaging of ",
    "ct imaging of ",
)

_PLACEHOLDER_RE = re.compile(
    r"\bunknown\b|\bn/?a\b|\bplaceholder\b|\[[^\]]*\]|\{[^}]*\}|\.\.\.",
    re.IGNORECASE,
)
_IN_RE = re.compile(r"\s+in\s+", re.IGNORECASE)
_SEP_RE = re.compile(r"\[\s*sep\s*\]", re.IGNORECASE)

# LLMs preserve dataset spellings such as "gall bladder" and "postcava".
# These deterministic fixes align the target term with BiomedParse's working
# vocabulary without changing the inferred site/modality.
_TARGET_TERM_FIXES = {
    "gall bladder": "gallbladder",
    "urinary bladder": "bladder",
    "postcava": "inferior vena cava",
    "arota": "aorta",
    "right heart ventricle": "right ventricle",
    "left heart ventricle": "left ventricle",
}


def _normalize_target_terms(prompt: str) -> str:
    match = _IN_RE.search(prompt)
    if match is None:
        return prompt
    target = prompt[: match.start()].strip()
    fixed = _TARGET_TERM_FIXES.get(target.lower())
    if fixed is None:
        return prompt
    return f"{fixed} in {prompt[match.end():].lstrip()}"


class CAPRValidationError(RuntimeError):
    """Raised when the LLM response cannot be validated after all retries."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        last_response: Optional[str] = None,
        errors: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_response = last_response
        self.errors = list(errors or [])


@dataclass
class CAPRResult:
    """Validated CAPR output plus the provenance needed to reproduce it."""

    prompts: Dict[int, str]
    raw_prompts: Dict[int, str]
    model: str
    attempts: int
    response_text: str
    context_hint: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    meta_prompt_sha256: str = ""
    api_style: str = "openai"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "api_style": self.api_style,
            "attempts": self.attempts,
            "meta_prompt_sha256": self.meta_prompt_sha256,
            "context_hint": self.context_hint,
            "raw_prompts": {str(k): v for k, v in sorted(self.raw_prompts.items())},
            "prompts": {str(k): v for k, v in sorted(self.prompts.items())},
            "warnings": list(self.warnings),
            "response_text": self.response_text,
        }


def _read_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _read_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _load_meta_prompt(path: Union[str, Path]) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()
    # Meta-Prompt.txt was copied out of a Python string literal.
    text = re.sub(r'^"""\s*', "", text)
    text = re.sub(r'\s*"""$', "", text)
    return text.strip()


def _load_extra_body() -> Optional[Dict[str, Any]]:
    raw = os.getenv("CAPR_EXTRA_BODY")
    if not raw:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("CAPR_EXTRA_BODY must be a JSON object")
    return value


def _clean_raw_prompt(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"prompt must be a string, got {type(text).__name__}")
    value = re.sub(r"\s+", " ", text).strip()
    if not value:
        raise ValueError("prompt must not be empty")
    if _SEP_RE.search(value):
        raise ValueError(f"raw prompt must not contain a separator: {value!r}")
    return value


def _normalise_raw_prompts(
    raw_prompts: Union[Mapping[Union[int, str], str], Sequence[str]]
) -> Dict[int, str]:
    if isinstance(raw_prompts, Mapping):
        result: Dict[int, str] = {}
        for key, value in raw_prompts.items():
            if str(key).strip().lower() == "instance_label":
                continue
            try:
                label_id = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"prompt key must be an integer label ID: {key!r}") from exc
            if label_id in result:
                raise ValueError(f"duplicate label ID: {label_id}")
            result[label_id] = _clean_raw_prompt(value)
        if not result:
            raise ValueError("at least one prompt is required")
        return result

    if isinstance(raw_prompts, Sequence) and not isinstance(raw_prompts, str):
        if not raw_prompts:
            raise ValueError("at least one prompt is required")
        return {i + 1: _clean_raw_prompt(value) for i, value in enumerate(raw_prompts)}

    raise TypeError("raw_prompts must be a mapping or a sequence of strings")


def _clean_model_prompt(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", value)
    value = re.sub(r"\s*```$", "", value)
    value = value.strip().strip('"').strip("'").strip()
    value = re.sub(r"^\s*(?:\d+[\.\)]\s*|[-*]\s*)", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value.rstrip(" .;，。;")


def _split_response(text: str, expected_count: int) -> List[str]:
    """Parse a CAPR response, tolerating fences, numbering, and [sep] case."""

    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = _SEP_RE.sub("[SEP]", cleaned)

    parts = [_clean_model_prompt(part) for part in cleaned.split("[SEP]")]
    parts = [part for part in parts if part]
    if len(parts) == expected_count:
        return parts

    # Some providers ignore the separator instruction and emit one prompt per
    # line. Only accept this fallback when the count matches exactly.
    lines = [_clean_model_prompt(line) for line in cleaned.splitlines()]
    lines = [
        line
        for line in lines
        if line
        and _IN_RE.search(line)
        and not line.lower().startswith(("here", "the inferred", "final", "output"))
    ]
    if len(lines) == expected_count:
        return lines

    return parts


def _validate_prompts(parts: Sequence[str], expected_count: int) -> List[str]:
    errors: List[str] = []
    if len(parts) != expected_count:
        errors.append(f"expected {expected_count} prompts, got {len(parts)}")
        return errors

    seen: Dict[str, int] = {}
    for index, prompt in enumerate(parts, start=1):
        if not prompt:
            errors.append(f"prompt {index} is empty")
            continue
        if _SEP_RE.search(prompt):
            errors.append(f"prompt {index} still contains a separator")
        if _PLACEHOLDER_RE.search(prompt):
            errors.append(f"prompt {index} contains a placeholder: {prompt!r}")
        if not _IN_RE.search(prompt):
            errors.append(
                f"prompt {index} is not in canonical '<target> in <site> <modality>' form: {prompt!r}"
            )
            continue
        suffix = _IN_RE.split(prompt, maxsplit=1)[-1].strip()
        if len(suffix.split()) < 2:
            errors.append(
                f"prompt {index} is missing the anatomical site or modality: {prompt!r}"
            )
        lowered = prompt.lower()
        for prefix in _BANNED_PREFIXES:
            if lowered.startswith(prefix):
                errors.append(
                    f"prompt {index} starts with a banned instruction prefix {prefix!r}: {prompt!r}"
                )
                break
        key = prompt.lower()
        if key in seen:
            errors.append(
                f"prompt {index} duplicates prompt {seen[key]}: {prompt!r}"
            )
        else:
            seen[key] = index

    return errors


def _user_message(raw: Mapping[int, str], context_hint: Optional[str]) -> str:
    prompt_list = "[SEP]".join(raw[label_id] for label_id in sorted(raw))
    if context_hint and context_hint.strip():
        return (
            "Optional global context hint (verify it against the list): "
            f"{context_hint.strip()}\n\n{prompt_list}"
        )
    return prompt_list


def _repair_message(
    raw: Mapping[int, str],
    context_hint: Optional[str],
    error: str,
    previous_response: str,
) -> str:
    expected = len(raw)
    return (
        f"Your previous response was rejected: {error}.\n"
        f"Return exactly {expected} prompts, in the same order as the input list. "
        "Each prompt must use the canonical form '<target> in <anatomical site> <modality>'. "
        "Separate prompts only with [SEP]. Do not add headings, numbering, explanations, "
        "or markdown fences.\n\n"
        "Previous response (for reference only):\n"
        f"{previous_response.strip()}\n\n"
        "Input list:\n"
        f"{_user_message(raw, context_hint)}"
    )


class CAPRClient:
    """LLM client implementing the paper's CAPR prompt regularization."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        meta_prompt_path: Optional[Union[str, Path]] = None,
        timeout: Optional[float] = None,
        validation_retries: Optional[int] = None,
        client: Any = None,
        extra_body: Optional[Dict[str, Any]] = None,
        api_style: Optional[str] = None,
    ) -> None:
        style = (api_style or os.getenv("CAPR_API_STYLE") or "").strip().lower()
        if not style:
            style = (
                "anthropic"
                if os.getenv("ANTHROPIC_AUTH_TOKEN") and not os.getenv("CAPR_API_KEY")
                else "openai"
            )
        if style not in {"openai", "anthropic"}:
            raise ValueError("api_style must be 'openai' or 'anthropic'")
        self.api_style = style

        if style == "anthropic":
            env_model = os.getenv("ANTHROPIC_MODEL")
            env_base_url = os.getenv("ANTHROPIC_BASE_URL")
            env_key = os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY")
        else:
            env_model = os.getenv("STEPFUN_MODEL") or os.getenv("OPENAI_MODEL")
            env_base_url = os.getenv("STEPFUN_BASE_URL") or os.getenv("OPENAI_BASE_URL")
            env_key = os.getenv("STEPFUN_API_KEY") or os.getenv("OPENAI_API_KEY")

        self.model = model or os.getenv("CAPR_MODEL") or env_model or DEFAULT_MODEL
        self.base_url = (
            base_url or os.getenv("CAPR_BASE_URL") or env_base_url or DEFAULT_BASE_URL
        )
        self.timeout = (
            float(timeout)
            if timeout is not None
            else _read_env_float("CAPR_TIMEOUT", DEFAULT_TIMEOUT)
        )
        self.validation_retries = (
            int(validation_retries)
            if validation_retries is not None
            else _read_env_int("CAPR_VALIDATION_RETRIES", DEFAULT_VALIDATION_RETRIES)
        )
        if self.validation_retries < 0:
            raise ValueError("validation_retries must be >= 0")

        meta_prompt_path = (
            meta_prompt_path
            or os.getenv("CAPR_META_PROMPT")
            or DEFAULT_META_PROMPT
        )
        self.meta_prompt_path = Path(meta_prompt_path)
        self.meta_prompt = _load_meta_prompt(self.meta_prompt_path)
        self.meta_prompt_sha256 = hashlib.sha256(
            self.meta_prompt.encode("utf-8")
        ).hexdigest()
        self.extra_body = extra_body if extra_body is not None else _load_extra_body()
        self.max_tokens = _read_env_int("CAPR_MAX_TOKENS", DEFAULT_MAX_TOKENS)
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.anthropic_version = os.getenv("CAPR_ANTHROPIC_VERSION", "2023-06-01")
        self.api_key = api_key or os.getenv("CAPR_API_KEY") or env_key

        if client is not None:
            self.client = client
            if not self.api_key:
                self.api_key = "injected-client"
            return

        if style == "anthropic" and not self.base_url:
            raise RuntimeError(
                "ANTHROPIC_BASE_URL or CAPR_BASE_URL must be set for the "
                "Anthropic API style."
            )

        if not self.api_key:
            if style == "anthropic":
                raise RuntimeError(
                    "Anthropic auth token is not set. Export ANTHROPIC_AUTH_TOKEN "
                    "(and optionally ANTHROPIC_BASE_URL / ANTHROPIC_MODEL), or set "
                    "CAPR_API_KEY / CAPR_BASE_URL / CAPR_MODEL."
                )
            raise RuntimeError(
                "CAPR_API_KEY is not set. Export CAPR_API_KEY (and optionally "
                "CAPR_BASE_URL / CAPR_MODEL) before running CAPR."
            )

        if style == "openai":
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        else:
            self.client = httpx.Client(timeout=self.timeout)

    def _call(self, messages: Sequence[Dict[str, str]]) -> str:
        if self.api_style == "anthropic":
            return self._call_anthropic(messages)
        return self._call_openai(messages)

    def _call_openai(self, messages: Sequence[Dict[str, str]]) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        response = self.client.chat.completions.create(**kwargs)
        if not getattr(response, "choices", None):
            raise RuntimeError("CAPR LLM returned no choices")
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise RuntimeError("CAPR LLM response was truncated (finish_reason=length)")
        content = getattr(choice.message, "content", None)
        if not content or not content.strip():
            raise RuntimeError("CAPR LLM returned an empty response")
        return content

    def _anthropic_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/v1/messages"):
            return base
        if base.endswith("/v1"):
            return base + "/messages"
        return base + "/v1/messages"

    def _call_anthropic(self, messages: Sequence[Dict[str, str]]) -> str:
        system = "\n\n".join(
            str(message["content"])
            for message in messages
            if message["role"] == "system"
        )
        conversation = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
            if message["role"] != "system"
        ]
        body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "system": system,
            "messages": conversation,
        }
        if self.extra_body:
            body.update(self.extra_body)

        headers = {
            "x-api-key": str(self.api_key),
            "Authorization": f"Bearer {self.api_key}",
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
        }
        response = self.client.post(
            self._anthropic_url(), headers=headers, json=body
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"CAPR Anthropic API error {response.status_code}: "
                f"{response.text[:1000]}"
            )
        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError(
                f"CAPR Anthropic API returned non-JSON: {response.text[:1000]}"
            ) from exc
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"CAPR Anthropic API error: {data['error']}")

        text = ""
        content = data.get("content") if isinstance(data, dict) else None
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                str(block["text"])
                for block in content
                if isinstance(block, dict)
                and block.get("type") in (None, "text")
                and isinstance(block.get("text"), str)
            )
        elif isinstance(data, dict) and data.get("choices"):
            try:
                text = data["choices"][0]["message"]["content"] or ""
            except Exception:
                text = ""
        if isinstance(data, dict) and data.get("stop_reason") == "max_tokens":
            raise RuntimeError(
                "CAPR Anthropic response was truncated (stop_reason=max_tokens)"
            )
        if not str(text).strip():
            raise RuntimeError("CAPR Anthropic returned an empty response")
        return str(text)

    def regularize(
        self,
        raw_prompts: Union[Mapping[Union[int, str], str], Sequence[str]],
        context_hint: Optional[str] = None,
    ) -> CAPRResult:
        """Regularize a dataset-level prompt list into canonical prompts.

        The call is made once for the whole list so that the LLM sees the same
        global context for every label, as described in the paper.
        """

        raw = _normalise_raw_prompts(raw_prompts)
        expected_count = len(raw)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.meta_prompt},
            {"role": "user", "content": _user_message(raw, context_hint)},
        ]

        last_response: Optional[str] = None
        last_error = ""
        attempts = self.validation_retries + 1
        for attempt in range(1, attempts + 1):
            last_response = self._call(messages)
            parts = [_normalize_target_terms(part) for part in _split_response(last_response, expected_count)]
            errors = _validate_prompts(parts, expected_count)
            if not errors:
                mapping = {
                    label_id: parts[index]
                    for index, label_id in enumerate(sorted(raw))
                }
                return CAPRResult(
                    prompts=mapping,
                    raw_prompts=raw,
                    model=self.model,
                    attempts=attempt,
                    response_text=last_response,
                    context_hint=context_hint,
                    meta_prompt_sha256=self.meta_prompt_sha256,
                    api_style=self.api_style,
                )

            last_error = "; ".join(errors)
            if attempt < attempts:
                messages = [
                    {"role": "system", "content": self.meta_prompt},
                    {
                        "role": "user",
                        "content": _repair_message(
                            raw, context_hint, last_error, last_response
                        ),
                    },
                ]

        raise CAPRValidationError(
            f"CAPR response failed validation after {attempts} attempt(s): {last_error}",
            attempts=attempts,
            last_response=last_response,
            errors=last_error.split("; ") if last_error else None,
        )


def correct_text(text: str) -> str:
    """Backward-compatible wrapper for ``[SEP]``-joined prompt strings.

    New code should call :meth:`CAPRClient.regularize` so that label IDs and
    provenance are preserved. This wrapper exists for the legacy call shape.
    """

    parts = [
        part.strip()
        for part in re.split(r"\[\s*sep\s*\]", str(text), flags=re.IGNORECASE)
        if part.strip()
    ]
    if not parts:
        raise ValueError("text must contain at least one prompt")
    result = CAPRClient().regularize(parts)
    return "[SEP]".join(result.prompts[index] for index in sorted(result.prompts))


def _cli_load_prompts(args: argparse.Namespace) -> Dict[int, str]:
    if args.prompts:
        parts = [
            part.strip()
            for part in re.split(r"\[\s*sep\s*\]", args.prompts, flags=re.IGNORECASE)
            if part.strip()
        ]
        return {index + 1: part for index, part in enumerate(parts)}

    if args.prompts_file:
        path = Path(args.prompts_file)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return _normalise_raw_prompts(payload)
            if isinstance(payload, list):
                return _normalise_raw_prompts(payload)
            raise TypeError("JSON prompt file must contain an object or a list")
        parts = [
            part.strip()
            for part in re.split(r"\[\s*sep\s*\]", path.read_text(encoding="utf-8"), flags=re.IGNORECASE)
            if part.strip()
        ]
        return {index + 1: part for index, part in enumerate(parts)}

    raise ValueError("provide --prompts or --prompts_file")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Regularize prompts with CAPR")
    parser.add_argument("--prompts", help="[SEP]-separated raw prompts")
    parser.add_argument("--prompts_file", help="JSON mapping or [SEP]/newline file")
    parser.add_argument("--context", default=None, help="optional global context hint")
    parser.add_argument("--model", default=None, help="override CAPR_MODEL")
    parser.add_argument("--base_url", default=None, help="override CAPR_BASE_URL")
    parser.add_argument(
        "--api_style",
        choices=("openai", "anthropic"),
        default=None,
        help="API protocol; use anthropic for /v1/messages endpoints",
    )
    parser.add_argument("--meta_prompt", default=None, help="override Meta-Prompt.txt")
    parser.add_argument("--out", default=None, help="write provenance JSON to this path")
    args = parser.parse_args(argv)

    raw = _cli_load_prompts(args)
    client = CAPRClient(
        base_url=args.base_url,
        model=args.model,
        api_style=args.api_style,
        meta_prompt_path=args.meta_prompt,
    )
    result = client.regularize(raw, context_hint=args.context)

    for label_id in sorted(result.prompts):
        print(f"{label_id}: {result.prompts[label_id]}")
    if args.out:
        Path(args.out).write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"provenance -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
