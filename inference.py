import asyncio
import os
import re
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


API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
API_KEY: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
IMAGE_NAME: Optional[str] = os.getenv("IMAGE_NAME")

SERVER_URL: str = os.getenv("SERVER_URL", "https://ankesh2-risk-prediction.hf.space")
TASK_NAME: str = os.getenv("RISK_PREDICTION_TASK", "medium")
BENCHMARK: str = os.getenv("RISK_PREDICTION_BENCHMARK", "risk_prediction")
INFERENCE_SEED: int = _env_int("INFERENCE_SEED", 42)

MAX_STEPS: int = _env_int("MAX_STEPS", 20)
TEMPERATURE: float = _env_float("TEMPERATURE", 0.1)
MAX_TOKENS: int = _env_int("MAX_TOKENS", 100) # Increased for CoT reasoning
SUCCESS_SCORE_THRESHOLD: float = _env_float("SUCCESS_SCORE_THRESHOLD", 0.50)

MIN_TOTAL_REWARD: float = _env_float("MIN_TOTAL_REWARD", -10.0)
MAX_TOTAL_REWARD: float = _env_float("MAX_TOTAL_REWARD", 10.0)

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
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def parse_action(response_text: str) -> int:
    if not response_text:
        return FALLBACK_ACTION

    # Search the entire response text for action keywords.
    upper_text = response_text.upper()
    if "FLAG" in upper_text:
        return 1
    if "HOLD" in upper_text:
        return 0

    return FALLBACK_ACTION


def normalize_score(total_reward: float, rewards: List[float], steps_taken: int) -> float:
    if not rewards: return 0.0
    
    # Strict Failure Penalty: Zero reward on final step = Failure
    if rewards[-1] <= 0.0: return 0.0
    
    # Max possible total reward for normalization
    if MAX_TOTAL_REWARD <= MIN_TOTAL_REWARD: return 0.0
    
    score = (total_reward - MIN_TOTAL_REWARD) / (MAX_TOTAL_REWARD - MIN_TOTAL_REWARD)
    return max(0.0, min(1.0, score))


def rule_based_action(observation: RiskPredictionObservation) -> Optional[int]:
    # Only handle absolute extremes in Python
    if observation.risk_level == "CRITICAL": return 1
    if observation.total_risk < 0.20 and not observation.flags: return 0
    return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None
    env = None
    history, rewards = [], []
    total_reward, steps_taken = 0.0, 0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        if IMAGE_NAME:
            env = await RiskPredictionEnv.from_docker_image(IMAGE_NAME)
        else:
            env = RiskPredictionEnv(base_url=SERVER_URL)

        result = await env.reset(seed=INFERENCE_SEED, task=TASK_NAME)
        observation = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done: break

            action_type = rule_based_action(observation)
            
            if action_type is None:
                response_text = "HOLD"
                if client:
                    try:
                        user_prompt = f"Step: {step}\nScores: {observation}\nHistory: {history[-3:]}\nDecide:"
                        completion = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
                            temperature=TEMPERATURE, max_tokens=MAX_TOKENS
                        )
                        response_text = completion.choices[0].message.content or "HOLD"
                    except: response_text = "HOLD"
                action_type = parse_action(response_text)

            action_label = "FLAG" if action_type == 1 else "HOLD"
            result = await env.step(RiskPredictionAction(action_type=action_type))
            observation = result.observation
            
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            total_reward += reward
            steps_taken = step
            
            log_step(step=step, action=action_label, reward=reward, done=result.done, error=None)
            history.append(f"S{step}: {action_label}")
            if result.done: break

        score = normalize_score(total_reward, rewards, steps_taken)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[CRITICAL DEBUG] {e}", file=sys.stderr)
        score = normalize_score(total_reward, rewards, steps_taken)
    finally:
        if env: await env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())