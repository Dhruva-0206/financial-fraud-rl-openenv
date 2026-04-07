import asyncio
import os
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    local_env = Path(__file__).resolve().with_name(".env")
    if local_env.exists():
        load_dotenv(local_env)
    else:
        load_dotenv()

# ---------------------------------------------------------------------------
# Import resolution
# ---------------------------------------------------------------------------
try:
    from risk_prediction import RiskPredictionAction, RiskPredictionEnv
    from risk_prediction.models import RiskPredictionObservation
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from risk_prediction import RiskPredictionAction, RiskPredictionEnv
        from risk_prediction.models import RiskPredictionObservation
    except ModuleNotFoundError:
        from client import RiskPredictionEnv
        from models import RiskPredictionAction, RiskPredictionObservation


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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


# Use getenv globally so the script doesn't crash if imported during a dry-run
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

SERVER_URL: str = os.getenv("SERVER_URL", "https://ankesh2-risk-prediction-59aba4b.hf.space")
TASK_NAME: str = os.getenv("RISK_PREDICTION_TASK", "medium")
BENCHMARK: str = os.getenv("RISK_PREDICTION_BENCHMARK", "risk_prediction")
INFERENCE_SEED: int = _env_int("INFERENCE_SEED", 42)

MAX_STEPS: int = _env_int("MAX_STEPS", 8)
TEMPERATURE: float = _env_float("TEMPERATURE", 0.7)
MAX_TOKENS: int = _env_int("MAX_TOKENS", 150)
SUCCESS_SCORE_THRESHOLD: float = _env_float("SUCCESS_SCORE_THRESHOLD", 0.50)

MAX_TOTAL_REWARD: float = float(MAX_STEPS) if MAX_STEPS > 0 else 1.0

FALLBACK_ACTION: int = 0  # HOLD

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """
        You are a forensic auditor. Your ONLY goal is to FLAG fraud.

        CRITICAL RULE:
        - If earnings_quality_risk is high (>0.60) while profitability_risk is low, this is fraud. FLAG IT.
        - If total_risk > 0.50, FLAG IT.
        - If you are unsure, but the risk is increasing, FLAG IT.
        - In the first observation (Step 1), if ANY risk score is > 0.60, it is an immediate FLAG. Do not wait for a second step to confirm.
        - Fraudsters often spike Earnings Quality early to inflate IPO or funding values. High EQ on Step 1 is a 90% indicator of fraud.

        A false flag is a -5 penalty, but missing a fraud is a -10 disaster. Be aggressive.

        Format your response:
        Reasoning: [1 sentence]
        Action: [FLAG or HOLD]
    """
).strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def parse_action(response_text: str) -> int:
    upper_text = (response_text or "").upper()
    if "FLAG" in upper_text:
        return 1
    if "HOLD" in upper_text:
        return 0

    return FALLBACK_ACTION


def normalize_score(rewards: List[float]) -> float:
    if not rewards or MAX_TOTAL_REWARD <= 0.0:
        return 0.0
    total_reward = sum(rewards)
    score = total_reward / MAX_TOTAL_REWARD
    return max(0.0, min(1.0, score))


def build_user_prompt(step: int, observation: RiskPredictionObservation, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Earnings quality risk: {observation.earnings_quality_risk:.2f}
        Channel stuffing risk: {observation.channel_stuffing_risk:.2f}
        Leverage risk: {observation.leverage_risk:.2f}
        Liquidity risk: {observation.liquidity_risk:.2f}
        Profitability risk: {observation.profitability_risk:.2f}
        Total risk: {observation.total_risk:.2f}
        Previous steps:
        {history_block}
        Decide your next action.
        Reply with:
        Reasoning: <one sentence>
        Action: <FLAG or HOLD>
        """
    ).strip()


def get_model_action(
    client: OpenAI,
    step: int,
    observation: RiskPredictionObservation,
    history: List[str],
) -> str:
    user_prompt = build_user_prompt(step, observation, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        response_text = (completion.choices[0].message.content or "").strip()
        action_type = parse_action(response_text)
        return "FLAG" if action_type == 1 else "HOLD"
    except Exception:
        return "HOLD"


def extract_last_action_error(observation: RiskPredictionObservation) -> Optional[str]:
    metadata = getattr(observation, "metadata", None)
    if isinstance(metadata, dict):
        last_error = metadata.get("last_action_error")
        if last_error:
            return str(last_error)
    return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    # Keep OpenAI client initialization resilient even if API_KEY is absent.
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "missing-api-key")

    env = None
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        if LOCAL_IMAGE_NAME:
            env = await RiskPredictionEnv.from_docker_image(LOCAL_IMAGE_NAME)
        else:
            env = RiskPredictionEnv(base_url=SERVER_URL)

        result = await env.reset(seed=INFERENCE_SEED, task=TASK_NAME)
        observation = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_label = get_model_action(client, step, observation, history)
            action_type = 1 if action_label == "FLAG" else 0
            result = await env.step(RiskPredictionAction(action_type=action_type))
            observation = result.observation

            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step
            step_error = extract_last_action_error(observation)

            log_step(step=step, action=action_label, reward=reward, done=result.done, error=step_error)

            history.append(
                f"Step {step}: action={action_label}, reward={reward:.2f}, total_risk={observation.total_risk:.2f}"
            )

            if result.done:
                break

    except Exception:
        pass
    finally:
        score = normalize_score(rewards)
        success = score >= SUCCESS_SCORE_THRESHOLD

        if env:
            try:
                await env.close()
            except Exception:
                pass

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())