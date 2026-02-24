#!/usr/bin/env python3
"""
Hybrid mission tagging engine: rule-based scoring first, LLM fallback.

Per spec: deterministic rules (high precision) first; model inference second for edge cases.
Outputs mission_primary, mission_secondary, mission_scores, mission_confidence,
mission_evidence_tokens, mission_rule_version for auditability.

Usage:
    from mission_tagging_engine import tag_mission_hybrid, tag_mission_hybrid_batch
    result = tag_mission_hybrid(product_row)
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SIGNALS_PATH = PROJECT_ROOT / "config" / "mission_signals.yaml"
RULE_VERSION = "1.0"


def _s(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        import math
        if math.isnan(v):
            return ""
    s = str(v).strip()
    return "" if s in ("", "nan", "None", "{}") else s


def _load_signals(config_path: Path | None = None) -> dict:
    """Load mission signals from YAML."""
    import yaml
    path = config_path or DEFAULT_SIGNALS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Mission signals config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_text_for_tagging(row: dict) -> tuple[str, str]:
    """
    Build normalized text for rule matching from product row.
    Returns (text_lower, category_lower) for token matching.
    """
    parts = []
    name = _s(row.get("name") or row.get("title") or row.get("product_name"))
    leaf_cat = _s(row.get("leaf_category"))
    breadcrumb = _s(row.get("category_breadcrumb") or row.get("category_breadcrumbs"))
    description = _s(row.get("description") or row.get("document") or "")
    material = _s(row.get("material_text") or row.get("materials_section") or row.get("material"))
    parts.extend([name, leaf_cat, breadcrumb, description, material])
    parts.extend([
        _s(row.get("product_details")),
        _s(row.get("bag_structure")),
        _s(row.get("interior_features")),
        _s(row.get("exterior_features")),
        _s(row.get("closure_type")),
        _s(row.get("handle_type")),
        _s(row.get("fabric_name")),
    ])
    text = " ".join(p for p in parts if p).lower()
    category = (leaf_cat or breadcrumb).lower()
    return text, category


def _tokenize(s: str) -> set[str]:
    """Extract word tokens for matching."""
    s = re.sub(r"[^a-z0-9\s]", " ", (s or "").lower())
    return {t for t in s.split() if len(t) > 1}


def _collect_mission_tokens(signals_config: dict) -> dict[str, set[str]]:
    """Build per-mission token sets from config."""
    out: dict[str, set[str]] = {}
    mt = signals_config.get("mission_tokens") or {}
    for mission_id, cfg in mt.items():
        tokens = set()
        for key in ("tokens", "shapes", "materials", "strap", "closure", "features", "handles", "category_keywords"):
            vals = cfg.get(key) or []
            for v in vals:
                if isinstance(v, str):
                    tokens.update(re.findall(r"\b[a-z0-9]+\b", v.lower()))
                else:
                    tokens.add(str(v).lower())
        out[mission_id] = tokens
    return out


def _compute_rule_scores(
    text_tokens: set[str],
    category: str,
    mission_token_sets: dict[str, set[str]],
    signals_config: dict,
) -> tuple[dict[str, float], set[str], str | None, tuple[str, str] | None]:
    """
    Apply token matching and priority rules.
    Returns (scores, evidence_tokens, override_mission, dual_mission).
    override_mission: if set, that mission wins (single).
    dual_mission: if set, (primary, secondary) from dual_mission rule.
    """
    scores: dict[str, float] = {m: 0.0 for m in mission_token_sets}
    evidence: set[str] = set()
    override_mission: str | None = None
    dual_mission: tuple[str, str] | None = None

    # Token matching: each matched token adds 0.33 strength (so 3 -> ~1.0)
    token_weight = 0.33
    for mission_id, tokens in mission_token_sets.items():
        matched = text_tokens & tokens
        if matched:
            scores[mission_id] += len(matched) * token_weight
            evidence.update(matched)

    # Apply priority rules
    rules = signals_config.get("priority_rules") or []
    for rule in rules:
        cond = rule.get("condition") or {}
        action = rule.get("action", "")
        mission = rule.get("mission") or rule.get("primary")

        # category_contains
        cat_keywords = cond.get("category_contains") or []
        if cat_keywords:
            for kw in cat_keywords:
                if kw.lower() in category:
                    if action == "category_override":
                        override_mission = mission
                    elif action == "boost":
                        scores[mission] = scores.get(mission, 0) + (rule.get("strength") or 0.9)
                        evidence.add(kw)
                    break

        # token_present
        tok = cond.get("token_present")
        if tok and tok.lower() in text_tokens:
            if action == "category_override":
                override_mission = mission
            elif action == "boost":
                strength = rule.get("strength", 0.9)
                scores[mission] = scores.get(mission, 0) + strength
                evidence.add(tok)

        # tokens_all -> dual_mission
        toks_all = cond.get("tokens_all") or []
        if toks_all and all(t.lower() in text_tokens for t in toks_all):
            if action == "dual_mission":
                prim = rule.get("primary")
                sec = rule.get("secondary")
                if prim:
                    dual_mission = (prim, sec or "")
                    scores[prim] = scores.get(prim, 0) + 0.9
                    if sec:
                        scores[sec] = scores.get(sec, 0) + 0.8
                    evidence.update(t.lower() for t in toks_all)

    return scores, evidence, override_mission, dual_mission


def _confidence_from_sum(signal_sum: float, k: float) -> float:
    """confidence = 1 - exp(-k * signal_sum). k tuned so 3 strong signals -> ~0.8."""
    import math
    return round(1 - math.exp(-k * signal_sum), 4)


def _pick_primary_secondary(
    scores: dict[str, float],
    dual_threshold: float,
) -> tuple[str | None, str | None]:
    """Pick primary (top score) and secondary (within threshold of primary)."""
    if not scores or all(v <= 0 for v in scores.values()):
        return None, None
    sorted_items = sorted(scores.items(), key=lambda x: -x[1])
    primary_id, primary_score = sorted_items[0]
    if primary_score <= 0:
        return None, None
    secondary_id = None
    if len(sorted_items) > 1:
        sec_id, sec_score = sorted_items[1]
        if primary_score - sec_score <= dual_threshold and sec_score > 0:
            secondary_id = sec_id
    return primary_id, secondary_id


def tag_mission_hybrid(
    row: dict,
    config_path: Path | None = None,
    use_llm_fallback: bool = True,
    confidence_threshold: float | None = None,
) -> dict:
    """
    Tag a single product with hybrid rules + optional LLM fallback.

    Returns dict with:
        mission_primary, mission_secondary, mission_tags (pipe-separated),
        mission_scores (JSON), mission_confidence, mission_evidence_tokens (pipe-separated),
        mission_rule_version, mission_source ("rules" | "llm").
    """
    signals = _load_signals(config_path)
    k = signals.get("confidence_k", 1.6)
    threshold = confidence_threshold if confidence_threshold is not None else signals.get("rule_confidence_threshold", 0.5)
    dual_threshold = signals.get("dual_mission_score_threshold", 0.1)

    text, category = _build_text_for_tagging(row)
    text_tokens = _tokenize(text)
    mission_tokens = _collect_mission_tokens(signals)

    scores, evidence, override, dual = _compute_rule_scores(text_tokens, category, mission_tokens, signals)

    if dual:
        primary, secondary = dual[0], dual[1] or None
    elif override:
        primary = override
        secondary = _pick_primary_secondary(scores, dual_threshold)[1]
    else:
        primary, secondary = _pick_primary_secondary(scores, dual_threshold)

    signal_sum = sum(scores.values()) if scores else 0
    confidence = _confidence_from_sum(signal_sum, k)

    if primary and confidence >= threshold:
        tags = [primary]
        if secondary:
            tags.append(secondary)
        return {
            "mission_primary": primary,
            "mission_secondary": secondary or "",
            "mission_tags": "|".join(tags),
            "mission_scores": json.dumps({m: round(v, 4) for m, v in sorted(scores.items(), key=lambda x: -x[1]) if v > 0}),
            "mission_confidence": confidence,
            "mission_evidence_tokens": "|".join(sorted(evidence)) if evidence else "",
            "mission_rule_version": signals.get("version", RULE_VERSION),
            "mission_source": "rules",
        }

    # LLM fallback
    if use_llm_fallback:
        try:
            import sys
            scripts_dir = Path(__file__).resolve().parent
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            from llm_mission_tagger import tag_missions_batch
            tag_lists = tag_missions_batch([row], batch_size=1)
            llm_tags = tag_lists[0] if tag_lists else []
            primary_llm = llm_tags[0] if llm_tags else ""
            secondary_llm = "|".join(llm_tags[1:]) if len(llm_tags) > 1 else ""
            return {
                "mission_primary": primary_llm,
                "mission_secondary": secondary_llm,
                "mission_tags": "|".join(llm_tags),
                "mission_scores": "",
                "mission_confidence": 0.0,
                "mission_evidence_tokens": "",
                "mission_rule_version": RULE_VERSION,
                "mission_source": "llm",
            }
        except Exception as e:
            logger.warning("LLM fallback failed: %s. Using rule result if any.", e)

    # No LLM or LLM failed: use rule result even if low confidence
    tags = []
    if primary:
        tags.append(primary)
    if secondary:
        tags.append(secondary)
    return {
        "mission_primary": primary or "",
        "mission_secondary": secondary or "",
        "mission_tags": "|".join(tags),
        "mission_scores": json.dumps({m: round(v, 4) for m, v in sorted(scores.items(), key=lambda x: -x[1]) if v > 0}),
        "mission_confidence": confidence,
        "mission_evidence_tokens": "|".join(sorted(evidence)) if evidence else "",
        "mission_rule_version": signals.get("version", RULE_VERSION),
        "mission_source": "rules" if primary else "none",
    }


def tag_mission_hybrid_batch(
    rows: list[dict],
    config_path: Path | None = None,
    use_llm_fallback: bool = True,
    confidence_threshold: float | None = None,
) -> list[dict]:
    """Tag a batch of products. Processes sequentially (rule engine is fast; LLM fallback per-row)."""
    return [tag_mission_hybrid(r, config_path, use_llm_fallback, confidence_threshold) for r in rows]
