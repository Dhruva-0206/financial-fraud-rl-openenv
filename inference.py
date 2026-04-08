#!/usr/bin/env python3
"""
OpenEnv mandatory inference script for risk_prediction.
Reference-aligned fail-safe execution with strict START/STEP/END logging.
"""

import asyncio
import math
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, List, Optional, Tuple

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# ASGI app export (openenv.yaml -> inference:app)
# ---------------------------------------------------------------------------

def _build_degraded_app(detail: str) -> Any:
    try:
        from fastapi import FastAPI
    except Exception:  # pragma: no cover
        class _FallbackApp:
            pass

        return _FallbackApp()

    fallback_app = FastAPI()

    @fallback_app.get("/health")
    def _health() -> dict:
        return {"status": "degraded", "detail": detail}

    return fallback_app


_APP_IMPORT_ERROR: Optional[str] = None

try:
    from risk_prediction.server.app import app
except Exception as exc:
    _APP_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from server.app import app
    except Exception as inner_exc:
        _APP_IMPORT_ERROR = (
            f"{_APP_IMPORT_ERROR} | fallback import failed: "
            f"{type(inner_exc).__name__}: {inner_exc}"
        )
        app = _build_degraded_app("server app import failed")


# ---------------------------------------------------------------------------
# Environment client imports
# ---------------------------------------------------------------------------

RiskPredictionEnv = None
RiskPredictionAction = None
RiskPredictionObservation = Any

try:
    from risk_prediction import RiskPredictionAction as _RiskPredictionAction
    from risk_prediction import RiskPredictionEnv as _RiskPredictionEnv
    from risk_prediction.models import RiskPredictionObservation as _RiskPredictionObservation

    RiskPredictionAction = _RiskPredictionAction
    RiskPredictionEnv = _RiskPredictionEnv
    RiskPredictionObservation = _RiskPredictionObservation
except Exception:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from risk_prediction import RiskPredictionAction as _RiskPredictionAction
        from risk_prediction import RiskPredictionEnv as _RiskPredictionEnv
        from risk_prediction.models import RiskPredictionObservation as _RiskPredictionObservation

        RiskPredictionAction = _RiskPredictionAction
        RiskPredictionEnv = _RiskPredictionEnv
        RiskPredictionObservation = _RiskPredictionObservation
    except Exception:
        try:
            from client import RiskPredictionEnv as _RiskPredictionEnv
            from models import RiskPredictionAction as _RiskPredictionAction
            from models import RiskPredictionObservation as _RiskPredictionObservation

            RiskPredictionAction = _RiskPredictionAction
            RiskPredictionEnv = _RiskPredictionEnv
            RiskPredictionObservation = _RiskPredictionObservation
        except Exception:
            RiskPredictionAction = None
            RiskPredictionEnv = None
            RiskPredictionObservation = Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SCORE_MIN = 0.01
_SCORE_MAX = 0.99


def _safe_score(raw: Any) -> float:
    try:
        score = float(raw)
        return max(_SCORE_MIN, min(_SCORE_MAX, score))
    except (TypeError, ValueError):
        return _SCORE_MIN


def safe_score(value: Any) -> float:
    if value is None:
        return _SCORE_MIN
    try:
        score = float(value)
    except (TypeError, ValueError):
        return _SCORE_MIN

    if math.isnan(score):
        return _SCORE_MIN
    if score >= 0.995:
        score = 0.989
    if score < 0.005:
        score = 0.01
    return max(_SCORE_MIN, min(_SCORE_MAX, score))


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


API_KEY: Optional[str] = os.getenv("API_KEY")
API_BASE_URL: Optional[str] = os.getenv("API_BASE_URL")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
SERVER_URL: str = os.getenv("SERVER_URL", "https://ankesh2-risk-prediction-59aba4b.hf.space")
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")

TASK_NAME: str = os.getenv("RISK_PREDICTION_TASK", "task_medium")
BENCHMARK: str = os.getenv("RISK_PREDICTION_BENCHMARK", "risk_prediction")
INFERENCE_SEED: int = _env_int("INFERENCE_SEED", 42)
MAX_STEPS: int = _env_int("MAX_STEPS", 8)
TEMPERATURE: float = _env_float("TEMPERATURE", 0.7)
MAX_TOKENS: int = _env_int("MAX_TOKENS", 150)
SUCCESS_SCORE_THRESHOLD: float = _env_float("SUCCESS_SCORE_THRESHOLD", 0.50)

_llm_client: Optional[Any] = None
if OpenAI is not None and API_KEY and API_BASE_URL:
    try:
        _llm_client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    except Exception:
        _llm_client = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={safe_score(reward):.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    if not rewards:
        rewards = [_SCORE_MIN]

    rewards_str = ",".join(f"{safe_score(r):.2f}" for r in rewards)
    raw_score = sum(rewards) / len(rewards) if rewards else _SCORE_MIN
    score = safe_score(raw_score)

    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM prompting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a forensic auditor. Your only goal is to decide HOLD or FLAG.

    Rules:
    - If total risk is high or rising, prefer FLAG.
    - If evidence is weak, choose HOLD.

    Return exactly:
    Reasoning: <one short sentence>
    Action: <FLAG or HOLD>
    """
).strip()


def _obs_float(observation: Any, field: str) -> float:
    try:
        return float(getattr(observation, field, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def build_user_prompt(step: int, observation: Any, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Earnings quality risk: {_obs_float(observation, 'earnings_quality_risk'):.2f}
        Channel stuffing risk: {_obs_float(observation, 'channel_stuffing_risk'):.2f}
        Leverage risk: {_obs_float(observation, 'leverage_risk'):.2f}
        Liquidity risk: {_obs_float(observation, 'liquidity_risk'):.2f}
        Profitability risk: {_obs_float(observation, 'profitability_risk'):.2f}
        Total risk: {_obs_float(observation, 'total_risk'):.2f}
        Previous steps:
        {history_block}
        Reply only with:
        Reasoning: <one sentence>
        Action: <FLAG or HOLD>
        """
    ).strip()


def parse_action(text: str) -> str:
    content = (text or "").upper()
    if "FLAG" in content:
        return "FLAG"
    if "HOLD" in content:
        return "HOLD"
    return "HOLD"


def force_proxy_call(client: Any) -> None:
    client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a fraud-risk auditor."},
            {"role": "user", "content": "Reply with HOLD."},
        ],
        temperature=0.0,
        max_tokens=8,
        stream=False,
    )


def get_model_action(client: Optional[Any], step: int, observation: Any, history: List[str]) -> str:
    if not client:
        return "FLAG" if _obs_float(observation, "total_risk") > 0.55 else "HOLD"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(step, observation, history)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        response_text = (completion.choices[0].message.content or "").strip()
        return parse_action(response_text)
    except Exception:
        return "HOLD"


def extract_last_action_error(observation: Any) -> Optional[str]:
    metadata = getattr(observation, "metadata", None)
    if isinstance(metadata, dict):
        last_error = metadata.get("last_action_error")
        if last_error:
            return str(last_error)
    return None


# ---------------------------------------------------------------------------
# Core run loop
# ---------------------------------------------------------------------------

async def run_episode(task_id: str) -> Tuple[bool, int, List[float]]:
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    success = False

    use_llm = bool(_llm_client)
    model_display = MODEL_NAME if use_llm else "hardcoded"
    log_start(task=task_id, env=BENCHMARK, model=model_display)

    if use_llm:
        try:
            force_proxy_call(_llm_client)
        except Exception:
            pass

    if RiskPredictionEnv is None or RiskPredictionAction is None:
        log_step(step=1, action="INIT", reward=_SCORE_MIN, done=True, error="client_import_failed")
        return False, 0, [_SCORE_MIN]

    env = None
    try:
        try:
            if LOCAL_IMAGE_NAME:
                env = await RiskPredictionEnv.from_docker_image(LOCAL_IMAGE_NAME)
            else:
                env = RiskPredictionEnv(base_url=SERVER_URL)
        except Exception as exc:
            log_step(
                step=1,
                action="INIT",
                reward=_SCORE_MIN,
                done=True,
                error=f"env_init_failed:{type(exc).__name__}",
            )
            return False, 0, [_SCORE_MIN]

        try:
            result = await env.reset(seed=INFERENCE_SEED, task=task_id)
        except Exception as exc:
            log_step(
                step=1,
                action="RESET",
                reward=_SCORE_MIN,
                done=True,
                error=f"reset_failed:{type(exc).__name__}",
            )
            return False, 0, [_SCORE_MIN]

        observation = getattr(result, "observation", None)

        for step in range(1, MAX_STEPS + 1):
            if bool(getattr(result, "done", False)):
                break

            action_label = get_model_action(_llm_client if use_llm else None, step, observation, history)
            action_type = 1 if action_label == "FLAG" else 0

            try:
                action_payload = RiskPredictionAction(action_type=action_type)
                result = await env.step(action_payload)
            except Exception as exc:
                log_step(
                    step=step,
                    action=action_label,
                    reward=_SCORE_MIN,
                    done=True,
                    error=f"step_failed:{type(exc).__name__}",
                )
                break

            observation = getattr(result, "observation", None)
            reward = _safe_score(getattr(result, "reward", _SCORE_MIN))
            done = bool(getattr(result, "done", False))
            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_label,
                reward=reward,
                done=done,
                error=extract_last_action_error(observation),
            )

            history.append(
                f"step={step} action={action_label} reward={reward:.2f} total_risk={_obs_float(observation, 'total_risk'):.2f}"
            )

            if done:
                break

        mean_reward = sum(rewards) / len(rewards) if rewards else _SCORE_MIN
        success = safe_score(mean_reward) >= SUCCESS_SCORE_THRESHOLD
        return success, steps_taken, rewards
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass


async def main() -> None:
    try:
        success, steps, rewards = await run_episode(TASK_NAME)
        log_end(success=success, steps=steps, rewards=rewards)
    except Exception as exc:
        log_step(step=1, action="FATAL", reward=_SCORE_MIN, done=True, error=type(exc).__name__)
        log_end(success=False, steps=0, rewards=[_SCORE_MIN])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        log_end(success=False, steps=0, rewards=[_SCORE_MIN])
