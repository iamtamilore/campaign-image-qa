"""
prompt_logger.py — master log of every kie.ai call, any script, any terminal.

Appends to ~/.claude/kie/prompt_json.log (JSON Lines — one JSON object per
line, immutable, append-only). Two phases per generation, joined by task_id:
  phase="request" — written the moment a task is submitted to kie.ai
  phase="result"  — written once the task resolves (success/fail)
Two-phase instead of one mutable record because multiple terminals/scripts
write concurrently — appending a new line is safe, rewriting an existing
line is not.

Schema is intentionally over-inclusive rather than minimal: this log doubles
as informal pilot data for the CA2 Practicum's own 2x2 factorial design
(structured-vs-plain prompt x ref-vs-no-ref — see reference_conditioning and
prompt_format below, which map directly onto Table 2 of
x24335592_practicum_CA2_report.pdf), as the automatic feed for
BRAIN_MODULES.txt M13 (credit spend), M15 (error log), M16 (measured
findings) and 17.16 (reuse bank), and as raw thesis-appendix material.
See PROMPT_LOG_SCHEMA.md in this directory for the full field reference.
"""

import json
import os
import time
from pathlib import Path

MASTER_LOG = Path(__file__).parent / "prompt_json.log"

# The research log lives at ~/.claude/kie/prompt_json.log, on the machine where the
# work was done. This module is vendored into a public repo, and a cloner has no
# ~/.claude/kie. Rather than guess from whether a file happens to exist (which is
# true on the research machine and false on a fresh clone), decide by where this
# module physically lives: vendored inside a repo's src/ means the repo is the
# owner of the log, so it goes to the repo's own logs/ directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if (_REPO_ROOT / "src" / "api" / "prompt_logger.py").is_file():
    MASTER_LOG = _REPO_ROOT / "logs" / "prompt_json.log"


def _resolve_log_path() -> Path:
    """Which log does this call belong to?

    Precedence:
      1. $KIE_PROMPT_LOG            explicit override, always wins
      2. a .kielog marker file      found by walking UP from the working directory
      3. the system-wide master log at ~/.claude/kie/prompt_json.log

    The marker exists so that a whole project tree can claim its own log without every
    script in it having to remember to opt in. Any generation run from anywhere under a
    marked directory - by any script, any terminal, any model, whether or not it knows
    the project exists - lands in that project's log instead of the master. A .kielog
    file contains one line: the path to the log, absolute or relative to the marker.

    This is what keeps the Practicum's 2x2 experiment data from mixing with everyday
    creative work, which is a different dataset answering a different question.
    """
    explicit = os.environ.get("KIE_PROMPT_LOG")
    if explicit:
        return Path(explicit).expanduser()
    try:
        here = Path.cwd().resolve()
    except OSError:
        return MASTER_LOG
    for d in [here, *here.parents]:
        marker = d / ".kielog"
        if marker.is_file():
            try:
                target = marker.read_text().strip().splitlines()[0].strip()
            except (OSError, IndexError):
                break
            if not target:
                break
            t = Path(target).expanduser()
            return t if t.is_absolute() else (d / t).resolve()
    return MASTER_LOG


LOG_PATH = _resolve_log_path()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _guess_prompt_format(prompt: str) -> str:
    "best-effort heuristic only -- not authoritative, re-check before citing in the thesis"
    p = prompt.strip()
    if p.startswith("{") and '"' in p:
        return "structured_json"
    slots = ("SUBJECT:", "STYLE:", "OUTFIT", "LAYOUT:", "TECHNICAL:")
    if sum(s in prompt for s in slots) >= 3:
        return "structured_5slot"
    return "plain"


def _guess_ca2_condition(prompt_format: str, has_ref: bool) -> str | None:
    "maps onto Table 2 of the CA2 PDF -- A=plain/no-ref, B=structured/no-ref, C=plain/ref, D=structured/ref"
    structured = prompt_format in ("structured_5slot", "structured_json")
    if structured and has_ref:
        return "D"
    if structured and not has_ref:
        return "B"
    if not structured and has_ref:
        return "C"
    return "A"


def log_request(task_id: str, model: str, job_type: str, prompt: str,
                 input_urls: list[str] | None = None,
                 input_params: dict | None = None,
                 project: str | None = None,
                 ward: str | None = None,
                 parent_call_id: str | None = None,
                 condition: str | None = None,
                 tags: list[str] | None = None) -> None:
    """call right after a kie.ai task is submitted (task_id known).
    condition: explicit "A"/"B"/"C"/"D" override (CA2 Table 2) when the caller
    already knows which cell of the 2x2 design this call belongs to -- pass it
    explicitly whenever known, it's more reliable than the format heuristic."""
    input_urls = input_urls or []
    prompt_format = _guess_prompt_format(prompt)
    has_ref = len(input_urls) > 0
    entry = {
        "log_version": "1.0",
        "phase": "request",
        "timestamp": _utc_now(),
        "task_id": task_id,
        "pid": os.getpid(),
        "project": project or Path.cwd().name,
        "ward": ward,
        "model": model,
        "job_type": job_type,
        "prompt": prompt,
        "prompt_format": prompt_format,
        "reference_conditioning": has_ref,
        "ca2_condition": condition or _guess_ca2_condition(prompt_format, has_ref),
        "ca2_condition_is_guess": condition is None,
        "input_urls": input_urls,
        "input_params": input_params or {},
        "parent_call_id": parent_call_id,
        "tags": tags or [],
    }
    _append(entry)


def log_result(task_id: str, state: str,
               credits_consumed=None, fail_reason: str | None = None,
               output_url: str | None = None,
               local_output_path: str | None = None,
               duration_seconds: float | None = None) -> None:
    "call once the task resolves (success, fail, or timeout)"
    output_hash = None
    if local_output_path and os.path.exists(local_output_path):
        import hashlib
        output_hash = "sha256:" + hashlib.sha256(Path(local_output_path).read_bytes()).hexdigest()
    entry = {
        "log_version": "1.0",
        "phase": "result",
        "timestamp": _utc_now(),
        "task_id": task_id,
        "state": state,
        "credits_consumed": credits_consumed,
        "fail_reason": fail_reason,
        "output_url": output_url,
        "local_output_path": local_output_path,
        "output_hash": output_hash,
        "duration_seconds": duration_seconds,
    }
    _append(entry)


def log_review(task_id: str, review: str, notes: str | None = None) -> None:
    """optional third phase -- add human judgment after looking at the output.
    review is free text but 'keep' / 'reject' / 'fix_face' / 'regenerate' are the
    established values so far. Call any time after the result phase, even much
    later (a separate review pass is fine, it's still just an appended line)."""
    entry = {
        "log_version": "1.0",
        "phase": "review",
        "timestamp": _utc_now(),
        "task_id": task_id,
        "review": review,
        "notes": notes,
    }
    _append(entry)


def log_qa(task_id: str, emb_sim: float | None = None,
           color_distance: float | None = None,
           ocr_confidence: float | None = None,
           gate_pass: bool | None = None, gate: str | None = None) -> None:
    "optional fourth phase -- BRAIN_MODULES Gate 1/2/2.5 metrics, once computed"
    entry = {
        "log_version": "1.0",
        "phase": "qa",
        "timestamp": _utc_now(),
        "task_id": task_id,
        "emb_sim": emb_sim,
        "color_distance": color_distance,
        "ocr_confidence": ocr_confidence,
        "gate_pass": gate_pass,
        "gate": gate,
    }
    _append(entry)


def _append(entry: dict) -> None:
    # Re-resolved per call, not cached at import: a long-running session can change
    # directory, and the log a call belongs to is decided by where the call happens.
    path = _resolve_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
