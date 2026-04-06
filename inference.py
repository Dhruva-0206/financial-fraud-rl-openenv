"""
Inference Script — Financial Fraud Detection RL Environment
===========================================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME  Optional local Docker image name when using from_docker_image().

- Defaults are set only for API_BASE_URL and MODEL_NAME (active setup defaults):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

- The inference script must be named `inference.py` and placed in the root
  directory of the project.
- Participants must use the OpenAI Client for all LLM calls using the above variables.
"""

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
# Import resolution — works whether called as:
#   python inference.py              (from inside risk_prediction/)
#   python -m risk_prediction.inference  (from parent directory)
# ---------------------------------------------------------------------------
try:
    from risk_prediction import RiskPredictionAction, RiskPredictionEnv
    from risk_prediction.models import RiskPredictionObservation
except ModuleNotFoundError:
    # Running directly from inside the package directory
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from risk_prediction import RiskPredictionAction, RiskPredictionEnv
        from risk_prediction.models import RiskPredictionObservation
    except ModuleNotFoundError:
        # Running in container layout where files are at repo root
        from client import RiskPredictionEnv
        from models import RiskPredictionAction, RiskPredictionObservation

# ---------------------------------------------------------------------------
# Configuration (read from environment — do NOT hard-code secrets)
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
API_KEY: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
# URL of the running OpenEnv environment server
SERVER_URL: str = os.getenv("SERVER_URL", "https://ankesh2-risk-prediction.hf.space")
TASK_NAME: str = os.getenv("RISK_PREDICTION_TASK", "medium")
BENCHMARK: str = os.getenv("RISK_PREDICTION_BENCHMARK", "risk_prediction")
INFERENCE_SEED: int = int(os.getenv("INFERENCE_SEED", "42"))

MAX_STEPS: int = 20          # Maximum window steps per episode before truncation
TEMPERATURE: float = 0.1     # Low temperature for deterministic forensic decisions
MAX_TOKENS: int = 50         # We only need a single word: FLAG or HOLD
FALLBACK_ACTION: int = 0     # HOLD — conservative fallback on LLM failure
FLAG_TOTAL_RISK_THRESHOLD: float = float(os.getenv("FLAG_TOTAL_RISK_THRESHOLD", "0.55"))
FLAG_DIM_THRESHOLD: float = float(os.getenv("FLAG_DIM_THRESHOLD", "0.65"))
FLAG_MIN_TOTAL_FOR_MULTI_DIM: float = float(os.getenv("FLAG_MIN_TOTAL_FOR_MULTI_DIM", "0.50"))
SUCCESS_SCORE_THRESHOLD: float = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.50"))
MIN_TOTAL_REWARD: float = float(os.getenv("MIN_TOTAL_REWARD", "-10.0"))
MAX_TOTAL_REWARD: float = float(os.getenv("MAX_TOTAL_REWARD", "10.0"))

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a forensic financial auditor using an AI-assisted risk assessment system.
    You review a company's 4-year financial window at each step and decide whether
    to continue monitoring or report the company for financial fraud.

    You receive six risk scores, each in the range [0.0, 1.0]:
      - earnings_quality_risk : Accrual bloat — reported profits far exceed actual cash flow
      - channel_stuffing_risk : Receivables spike — receivables growing faster than revenue
      - leverage_risk         : Debt burden — long-term debt relative to total assets
      - liquidity_risk        : Cash exhaustion — cash reserves critically low
      - profitability_risk    : Margin collapse — sustained, severe negative ROA
      - total_risk            : Weighted composite of all five dimensions

    Risk levels:  LOW < 0.30 | MEDIUM 0.30–0.55 | HIGH 0.55–0.75 | CRITICAL >= 0.75

    Your action choices:
      HOLD  — The financials look acceptable; continue to the next reporting period.
              Small positive reward for each safe progression on a clean company.
      FLAG  — You are reporting this company for suspected fraud.
              Correct flag on active fraud window: +10.0 reward.
              Early flag on a fraud company (before the fraud window): +2.0 reward.
              False flag on a clean company: -5.0 penalty (defamation lawsuit).
              Missed active fraud (HOLD during fraud window): -10.0 (auditors fired).

    Decision guidance:
            - Reply FLAG when total_risk is HIGH (>= 0.55) or risk_level is CRITICAL.
            - If total_risk is LOW, reply HOLD.
            - If total_risk is MEDIUM, reply FLAG only when at least two specific
                risk scores are high (>= 0.65).
            - If only one specific risk score is high and total_risk is below 0.50,
                reply HOLD.
            - Balance caution with fraud detection: avoid false accusations, but do
                not ignore sustained high-risk patterns.

    Reply with EXACTLY one word: FLAG  or  HOLD
    Do not include any explanation, punctuation, or additional text.
    """
).strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _risk_bar(score: float, width: int = 20) -> str:
    """Visual ASCII bar for a risk score in [0, 1]."""
    filled = round(score * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {score:.3f}"


def _risk_label(score: float) -> str:
    if score < 0.30:
        return "LOW"
    elif score < 0.55:
        return "MEDIUM"
    elif score < 0.75:
        return "HIGH"
    return "CRITICAL"


def build_history_lines(history: List[str]) -> str:
    """Return the last 4 history entries (or 'None' if empty)."""
    if not history:
        return "  (none)"
    return "\n".join(f"  {line}" for line in history[-4:])


def build_user_prompt(
    step: int,
    observation: RiskPredictionObservation,
    history: List[str],
) -> str:
    """Construct the per-step user message from the current observation."""
    flags_str = ", ".join(observation.flags) if observation.flags else "(none)"

    # Show warning markers next to high-risk dimensions
    def warn(score: float) -> str:
        if score >= 0.75:
            return "  ⛔ CRITICAL"
        elif score >= 0.55:
            return "  ⚠ HIGH"
        elif score >= 0.30:
            return "  ~ MEDIUM"
        return ""

    prompt = textwrap.dedent(
        f"""
        Step: {step}
        Company ID (gvkey): {observation.gvkey}
        Fiscal year (most recent in window): {observation.fiscal_year}

        ── FORENSIC RISK ANALYSIS (4-year window) ──────────────────────
          Earnings Quality Risk  : {_risk_bar(observation.earnings_quality_risk)}{warn(observation.earnings_quality_risk)}
          Channel Stuffing Risk  : {_risk_bar(observation.channel_stuffing_risk)}{warn(observation.channel_stuffing_risk)}
          Leverage Risk          : {_risk_bar(observation.leverage_risk)}{warn(observation.leverage_risk)}
          Liquidity Risk         : {_risk_bar(observation.liquidity_risk)}{warn(observation.liquidity_risk)}
          Profitability Risk     : {_risk_bar(observation.profitability_risk)}{warn(observation.profitability_risk)}
          ──────────────────────────────────────────────────────────────
          Total Risk Score       : {_risk_bar(observation.total_risk)}  [{observation.risk_level}]{warn(observation.total_risk)}

        Active forensic flags: {flags_str}

        Previous actions:
        {build_history_lines(history)}

        Reply with exactly one word: FLAG  or  HOLD
        """
    ).strip()
    return prompt


def parse_action(response_text: str) -> int:
    """
    Parse the LLM response into an integer action.

    Returns:
        1  if the model said FLAG (any case, any position)
        0  (HOLD) otherwise — conservative fallback
    """
    if not response_text:
        return FALLBACK_ACTION
    # Accept FLAG anywhere in the response (some models add trailing whitespace/newline)
    if re.search(r"\bFLAG\b", response_text, re.IGNORECASE):
        return 1
    if re.search(r"\bHOLD\b", response_text, re.IGNORECASE):
        return 0
    return FALLBACK_ACTION


def has_strong_flag_signal(observation: RiskPredictionObservation) -> bool:
    high_dims = sum(
        score >= FLAG_DIM_THRESHOLD
        for score in (
            observation.earnings_quality_risk,
            observation.channel_stuffing_risk,
            observation.leverage_risk,
            observation.liquidity_risk,
            observation.profitability_risk,
        )
    )
    if observation.risk_level == "CRITICAL":
        return True
    if observation.total_risk >= FLAG_TOTAL_RISK_THRESHOLD:
        return True
    if high_dims >= 2 and observation.total_risk >= FLAG_MIN_TOTAL_FOR_MULTI_DIM:
        return True
    return False


def stabilize_action(proposed_action: int, observation: RiskPredictionObservation) -> int:
    # Suppress aggressive early FLAG calls unless the risk window has strong signals.
    if proposed_action == 1 and not has_strong_flag_signal(observation):
        return 0
    return proposed_action


def _flags_to_str(flags: List[str]) -> str:
    return ",".join(flags) if flags else "none"


def _single_line(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = _single_line(error) if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def normalize_score(total_reward: float, rewards: List[float]) -> float:
    if rewards and all(0.0 <= r <= 1.0 for r in rewards):
        return max(0.0, min(1.0, sum(rewards) / len(rewards)))
    if MAX_TOTAL_REWARD <= MIN_TOTAL_REWARD:
        return 0.0
    score = (total_reward - MIN_TOTAL_REWARD) / (MAX_TOTAL_REWARD - MIN_TOTAL_REWARD)
    return max(0.0, min(1.0, score))


def extract_last_action_error(observation: RiskPredictionObservation) -> Optional[str]:
    metadata = getattr(observation, "metadata", None)
    if isinstance(metadata, dict):
        last_error = metadata.get("last_action_error")
        if last_error:
            return str(last_error)
    return None


# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------

def main() -> None:
    client: Optional[OpenAI] = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None
    async_env = None
    env = None

    history: List[str] = []
    rewards: List[float] = []
    total_reward = 0.0
    executed_steps = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        if LOCAL_IMAGE_NAME:
            async_env = asyncio.run(RiskPredictionEnv.from_docker_image(LOCAL_IMAGE_NAME))
        else:
            async_env = RiskPredictionEnv(base_url=SERVER_URL)

        env = async_env.sync()
        result = env.reset(seed=INFERENCE_SEED, task=TASK_NAME)
        observation: RiskPredictionObservation = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            user_prompt = build_user_prompt(step, observation, history)
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}],
                },
            ]

            response_text = "HOLD"
            step_error: Optional[str] = None
            if client is None:
                step_error = "missing_hf_token"
            else:
                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        stream=False,
                    )
                    response_text = (completion.choices[0].message.content or "").strip() or "HOLD"
                except Exception as exc:  # noqa: BLE001
                    step_error = str(exc)

            proposed_action = parse_action(response_text)
            action_type = stabilize_action(proposed_action, observation)
            action_label = "FLAG" if action_type == 1 else "HOLD"

            result = env.step(RiskPredictionAction(action_type=action_type))
            observation = result.observation

            reward = float(result.reward or 0.0)
            done = bool(result.done)
            last_action_error = extract_last_action_error(observation)
            if not step_error and last_action_error:
                step_error = last_action_error

            rewards.append(reward)
            total_reward += reward
            executed_steps = step

            log_step(step=step, action=action_label, reward=reward, done=done, error=step_error)

            history.append(
                f"Step {step:>2}: {action_label} -> reward {reward:+.2f}"
                + (f" [flags: {', '.join(observation.flags)}]" if observation.flags else "")
            )

            if done:
                break

        score = normalize_score(total_reward, rewards)
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception:
        success = False
        score = normalize_score(total_reward, rewards)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        log_end(success=success, steps=executed_steps, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
