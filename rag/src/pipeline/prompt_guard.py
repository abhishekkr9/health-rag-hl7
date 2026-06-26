"""Detect and block prompt-injection / jailbreak attempts before LLM calls."""

from __future__ import annotations

import re

INJECTION_REFUSAL = (
    "I can only help with questions about FHIR patient records. "
    "I can't change my instructions or role. Please ask a clinical data question."
)

# Patterns target instruction-override attempts, not clinical phrasing such as
# "ignore previous medications".
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bignore\s+(?:all\s+)?(?:previous|prior|above|earlier|your)\s+"
        r"(?:instructions?|rules?|prompts?|directives?|guidelines?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:disregard|forget|override|bypass)\s+(?:all\s+)?"
        r"(?:previous|prior|your|the|my)?\s*"
        r"(?:instructions?|rules?|prompts?|directives?|guidelines?|restrictions?|guardrails?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:do\s+not|don't)\s+follow\s+(?:your|the|any)\s+"
        r"(?:instructions?|rules?|prompts?|guidelines?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:reveal|show|print|display|repeat|output)\s+(?:your|the|system)\s+"
        r"(?:system\s+)?(?:prompt|instructions?|rules?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:what\s+(?:is|are)\s+your|tell\s+me\s+your)\s+"
        r"(?:system\s+)?(?:prompt|instructions?|rules?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:you\s+are\s+now|from\s+now\s+on\s+you\s+(?:are|must|will)|"
        r"act\s+as|pretend\s+(?:to\s+be|you\s+are)|roleplay\s+as)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:new|updated|secret|hidden)\s+instructions?\s*:",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:jailbreak|dan\s+mode|developer\s+mode|sudo\s+mode|god\s+mode)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bignore\b[^.?!\n]{0,80}\b(?:and|,)\s*"
        r"(?:tell\s+me|say|respond|answer|give\s+me)\b",
        re.IGNORECASE,
    ),
)


def is_prompt_injection(text: str) -> bool:
    """Return True when the user message looks like a prompt-injection attempt."""
    if not text or not text.strip():
        return False
    normalized = " ".join(text.split())
    return any(pattern.search(normalized) for pattern in _INJECTION_PATTERNS)
