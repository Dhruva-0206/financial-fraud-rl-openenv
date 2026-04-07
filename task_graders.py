from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    difficulty: str
    description: str
    score_map: Dict[str, float]


TASK_DEFINITIONS: Dict[str, TaskDefinition] = {
    "easy": TaskDefinition(
        task_id="easy",
        difficulty="easy",
        description=(
            "Conservative fraud triage with generous partial-progress credit. "
            "Designed for onboarding and baseline stability."
        ),
        score_map={
            "true_positive": 1.0,
            "early_warning": 0.90,
            "true_negative": 1.0,
            "hold_progress": 0.70,
            "false_positive": 0.25,
            "false_negative": 0.00,
        },
    ),
    "medium": TaskDefinition(
        task_id="medium",
        difficulty="medium",
        description=(
            "Balanced fraud detection task with moderate penalty for false alarms "
            "and misses while retaining partial progress reward."
        ),
        score_map={
            "true_positive": 1.0,
            "early_warning": 0.80,
            "true_negative": 1.0,
            "hold_progress": 0.60,
            "false_positive": 0.10,
            "false_negative": 0.00,
        },
    ),
    "hard": TaskDefinition(
        task_id="hard",
        difficulty="hard",
        description=(
            "Strict production-style task emphasizing precision and recall under "
            "high uncertainty with reduced partial-progress credit."
        ),
        score_map={
            "true_positive": 1.0,
            "early_warning": 0.75,
            "true_negative": 1.0,
            "hold_progress": 0.50,
            "false_positive": 0.00,
            "false_negative": 0.00,
        },
    ),
}


def normalize_task_id(task_id: str | None) -> str:
    if not task_id:
        return "medium"
    normalized = task_id.strip().lower()
    return normalized if normalized in TASK_DEFINITIONS else "medium"


def classify_outcome(raw_reward: float, done: bool) -> str:
    if raw_reward >= 9.99:
        return "true_positive"
    if 1.99 <= raw_reward <= 2.01:
        return "early_warning"
    if raw_reward <= -9.99:
        return "false_negative"
    if -5.01 <= raw_reward <= -4.99:
        return "false_positive"
    if done and raw_reward >= 5.09:
        return "true_negative"
    return "hold_progress"


def grade_step(task_id: str, raw_reward: float, done: bool) -> Tuple[float, str]:
    normalized_task_id = normalize_task_id(task_id)
    task = TASK_DEFINITIONS[normalized_task_id]
    outcome = classify_outcome(raw_reward=raw_reward, done=done)
    score = float(task.score_map.get(outcome, 0.0))
    score = max(0.0, min(1.0, score))
    return score, outcome


def _coerce_reward_done(*args, **kwargs) -> Tuple[float, bool]:
    """
    Best-effort parser for validator grader calls with varying signatures.
    Returns (raw_reward, done).
    """
    raw_reward = kwargs.get("raw_reward", kwargs.get("reward", 0.0))
    done = kwargs.get("done", False)

    if args:
        # Accept common positional pattern: (raw_reward, done)
        if isinstance(args[0], (int, float)):
            raw_reward = float(args[0])
        if len(args) > 1 and isinstance(args[1], bool):
            done = bool(args[1])

    return float(raw_reward), bool(done)


def grade_easy(*args, **kwargs) -> float:
    raw_reward, done = _coerce_reward_done(*args, **kwargs)
    score, _ = grade_step(task_id="easy", raw_reward=raw_reward, done=done)
    return score


def grade_medium(*args, **kwargs) -> float:
    raw_reward, done = _coerce_reward_done(*args, **kwargs)
    score, _ = grade_step(task_id="medium", raw_reward=raw_reward, done=done)
    return score


def grade_hard(*args, **kwargs) -> float:
    raw_reward, done = _coerce_reward_done(*args, **kwargs)
    score, _ = grade_step(task_id="hard", raw_reward=raw_reward, done=done)
    return score
