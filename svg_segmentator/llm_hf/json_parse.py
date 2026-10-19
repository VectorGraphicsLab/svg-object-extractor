"""JSON extraction from LLM output (same strategy as ``extract_objects.py``)."""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from typing import Optional

try:
    from json_repair import repair_json  # type: ignore
except ImportError:
    repair_json = None

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n?|```$", re.MULTILINE)
_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def _try_repair(candidate: str) -> Optional[dict]:
    if repair_json is None or not candidate:
        return None
    try:
        repaired = repair_json(candidate, return_objects=True)
    except Exception:
        return None
    if isinstance(repaired, dict):
        return repaired
    if isinstance(repaired, list):
        return {"objects": repaired}
    return None


def extract_json(raw: str) -> dict:
    text = raw.strip()
    text = _THINK_RE.sub("", text).strip()
    text = _FENCE_RE.sub("", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = _OBJ_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    for candidate in filter(None, (m.group(0) if m else None, text)):
        repaired = _try_repair(candidate)
        if repaired is not None:
            truncated = not text.rstrip().endswith("}")
            hint = " (likely truncated — bump --max-new-tokens)" if truncated else ""
            print(f"[warn] json-repair salvaged malformed model output{hint}", file=sys.stderr)
            return repaired

    raise ValueError("Could not locate JSON object in model output.")


def validate_svg_fragment(svg_str: str) -> Optional[str]:
    s = (svg_str or "").strip()
    if not s:
        return "empty"
    if "<svg" not in s.lower():
        return "no <svg> root"
    try:
        ET.fromstring(s)
    except ET.ParseError as e:
        return f"parse error: {e}"
    return None


_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str) -> str:
    return _SAFE_CHARS_RE.sub("_", name).strip("._") or "object"
