import os
from pathlib import Path
from typing import List
from uuid import uuid4

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Absolute path to the data directory — works regardless of working directory
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DEFAULT_CSV = str(_DATA_DIR / "fraud_dataset.csv")
_REQUIRED_COLUMNS = [
    "fyear",
    "gvkey",
    "misstate",
    "at",
    "che",
    "dltt",
    "lt",
    "ni",
    "rect",
    "sale",
]


def _candidate_csv_paths(csv_path: str = None) -> List[Path]:
    """Build an ordered list of dataset paths to try."""
    candidates = [
        csv_path,
        os.environ.get("FRAUD_CSV_PATH"),
        _DEFAULT_CSV,
        "/app/env/data/fraud_dataset.csv",
        "/app/data/fraud_dataset.csv",
        str(Path.cwd() / "data" / "fraud_dataset.csv"),
    ]

    seen = set()
    resolved: List[Path] = []
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate).expanduser()
        key = str(candidate_path)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(candidate_path)
    return resolved


def _load_dataset(csv_path: str = None) -> tuple[pd.DataFrame, str]:
    """Load CSV data from candidate paths and fail if none are valid."""
    load_errors = []

    for candidate in _candidate_csv_paths(csv_path):
        if not candidate.is_file():
            load_errors.append(f"{candidate} (file not found)")
            continue

        try:
            df = pd.read_csv(candidate, usecols=_REQUIRED_COLUMNS, on_bad_lines="skip")
            if not df.empty:
                return df, str(candidate)
            load_errors.append(f"{candidate} (empty dataframe)")
        except Exception as exc:  # noqa: BLE001
            load_errors.append(f"{candidate} ({exc})")

    raise FileNotFoundError(
        "Unable to load fraud dataset from configured paths: " + "; ".join(load_errors)
    )

# Import the forensic risk engine — supports both package and standalone usage
try:
    from ..risk_engine_v5 import evaluate_company
except ImportError:
    from risk_engine_v5 import evaluate_company  # type: ignore[no-redef]

# Import Pydantic models — supports both package and standalone usage
try:
    from ..models import RiskPredictionAction, RiskPredictionObservation
except ImportError:
    from models import RiskPredictionAction, RiskPredictionObservation  # type: ignore[no-redef]

try:
    from ..task_graders import TASK_DEFINITIONS, grade_step, normalize_task_id
except ImportError:
    from task_graders import TASK_DEFINITIONS, grade_step, normalize_task_id  # type: ignore[no-redef]

class FinancialFraudEnv(gym.Env):
    """
    Custom OpenEnv / Gymnasium Environment for Financial Red Flag Detection.
    The agent steps through a company's timeline and decides when to blow the whistle.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, csv_path=None, window_size=4):
        super(FinancialFraudEnv, self).__init__()

        self.window_size = window_size

        # 1. Load and prepare the dataset
        self.df, resolved_csv_path = _load_dataset(csv_path=csv_path)
        print(f"Loading dataset from: {resolved_csv_path}")
        self.df = self.df.sort_values(by=["gvkey", "fyear"]).fillna(0)

        for column in ("fyear", "misstate", "at", "che", "dltt", "lt", "ni", "rect", "sale"):
            self.df[column] = pd.to_numeric(self.df[column], errors="coerce")
        self.df = self.df.dropna(subset=["gvkey", "fyear"])
        self.df["gvkey"] = self.df["gvkey"].astype(str)
        self.df[["misstate", "at", "che", "dltt", "lt", "ni", "rect", "sale"]] = self.df[
            ["misstate", "at", "che", "dltt", "lt", "ni", "rect", "sale"]
        ].fillna(0.0)
        self.df["fyear"] = self.df["fyear"].fillna(0).astype(int)
        self.df["misstate"] = self.df["misstate"].astype(int)
        
        # Filter out companies with too little data
        counts = self.df['gvkey'].value_counts()
        valid_gvkeys = counts[counts >= self.window_size].index
        self.df = self.df[self.df['gvkey'].isin(valid_gvkeys)]
        if self.df.empty:
            raise ValueError("Dataset has no companies with enough history for configured window_size")

        self.companies = self.df['gvkey'].unique().tolist()

        # Precompute deterministic company pools for seeded resets.
        self.fraud_companies = sorted(
            self.df[self.df['misstate'] == 1]['gvkey'].unique().tolist()
        )
        company_labels = self.df.groupby('gvkey')['misstate'].max()
        self.clean_companies = sorted(company_labels[company_labels == 0].index.tolist())
        
        # 2. Define Action Space: 0 = Hold (Looks Clean), 1 = Flag (Fraud!)
        self.action_space = spaces.Discrete(2)
        
        # 3. Define Observation Space
        # Features: [earnings_quality, channel_stuffing, leverage, liquidity, profitability, total_risk]
        # All values from risk_engine_v5 are naturally clamped between 0.0 and 1.0
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        
        # Episode state variables
        self.current_company_df = None
        self.current_gvkey = None
        self.current_step = 0
        self.max_steps = 0
        self.is_fraud_company = False
        self.window_has_fraud = False
        # Stores the full evaluate_company() result so wrappers can read it
        self._last_eval_result: dict = {}

    def _get_company_timeseries(self, window_df):
        """Formats the dataframe slice for the risk engine."""
        return {
            "revenue": window_df["sale"].tolist(),
            "cash": window_df["che"].tolist(),
            "debt": window_df["dltt"].tolist(),
            "assets": window_df["at"].tolist(),
            "liabilities": window_df["lt"].tolist(),
            "receivables": window_df["rect"].tolist(),
            "profit": window_df["ni"].tolist(),
        }

    def _get_observation(self):
        """Extracts the 4-year window, runs the risk engine, and returns the state."""
        start_idx = self.current_step
        end_idx = self.current_step + self.window_size
        window_df = self.current_company_df.iloc[start_idx:end_idx]
        
        # Check if fraud occurs in this specific window
        self.window_has_fraud = window_df['misstate'].max() == 1
        
        timeseries = self._get_company_timeseries(window_df)

        eval_result = evaluate_company(timeseries)
        self._last_eval_result = eval_result  # cache for wrappers
        risks = eval_result["risk_breakdown"]
        obs = np.array([
            risks["earnings_quality"],
            risks["channel_stuffing"],
            risks["leverage"],
            risks["liquidity"],
            risks["profitability"],
            eval_result["total_risk"],
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        """Starts a new episode with a new random company."""
        super().reset(seed=seed)

        # Balanced, seed-aware company sampling for reproducible baseline episodes.
        target_fraud = bool(self.np_random.integers(0, 2))
        if target_fraud and self.fraud_companies:
            pool = self.fraud_companies
        elif self.clean_companies:
            pool = self.clean_companies
        else:
            pool = sorted(self.companies)

        self.current_gvkey = pool[int(self.np_random.integers(0, len(pool)))]

        self.current_company_df = self.df[self.df['gvkey'] == self.current_gvkey].copy()
        self.is_fraud_company = self.current_company_df['misstate'].max() == 1
        
        # We slide a 4-year window across the company's lifespan
        self.max_steps = len(self.current_company_df) - self.window_size
        self.current_step = 0
        
        observation = self._get_observation()
        info = {
            "gvkey": self.current_gvkey, 
            "is_fraud_company": self.is_fraud_company,
            "starting_year": self.current_company_df.iloc[self.current_step]['fyear']
        }
        
        return observation, info

    def step(self, action):
        """Executes the agent's action and advances the environment."""
        done = False
        truncated = False
        reward = 0.0
        
        # ACTION 1: BLOW THE WHISTLE (Flag Fraud)
        if action == 1:
            if self.window_has_fraud:
                reward = 10.0   # True Positive! Caught them red-handed.
            elif self.is_fraud_company:
                reward = 2.0    # Early warning (caught it before the SEC did)
            else:
                reward = -5.0   # False Positive! Sued for defamation.
            done = True         # Episode ends once you blow the whistle
            
        # ACTION 0: HOLD (Looks Clean)
        elif action == 0:
            if self.window_has_fraud:
                reward = -10.0  # False Negative! You missed active fraud.
                done = True     # Episode ends, the auditors got fired.
            else:
                reward = 0.1    # Small positive reward for safely progressing
                self.current_step += 1
                
                # Check if we reached the end of the company's timeline
                if self.current_step >= self.max_steps:
                    done = True
                    if not self.is_fraud_company:
                        reward += 5.0  # True Negative! Successfully audited a clean company.
        
        observation = self._get_observation() if not done else np.zeros(6, dtype=np.float32)
        
        info = {
            "gvkey": self.current_gvkey,
            "year": self.current_company_df.iloc[min(self.current_step, len(self.current_company_df)-1)]['fyear'],
            "is_fraud_company": self.is_fraud_company,
            "was_fraud": self.window_has_fraud
        }
        
        return observation, reward, done, truncated, info

    def render(self):
        """Console output for human debugging."""
        obs = self._get_observation()
        print(f"Year: {self.current_company_df.iloc[self.current_step + self.window_size - 1]['fyear']} | "
              f"Risk Score: {obs[5]:.2f} | Action needed...")

# =====================================================================
# OpenEnv Wrapper
# =====================================================================

class RiskPredictionEnvironment(
    Environment[RiskPredictionAction, RiskPredictionObservation, State]
):
    """
    OpenEnv wrapper around FinancialFraudEnv.

    Translates between the Gymnasium numpy/tuple interface and the
    Pydantic model interface expected by the OpenEnv HTTP/WebSocket server.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, csv_path: str = None, window_size: int = 4, task_id: str = None):
        super().__init__()
        self._gym_env = FinancialFraudEnv(csv_path=csv_path, window_size=window_size)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        configured_task = task_id or os.environ.get("RISK_PREDICTION_TASK", "task_medium")
        self._task_id = normalize_task_id(configured_task)
        self._task_definition = TASK_DEFINITIONS[self._task_id]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_obs(
        self,
        obs_arr: np.ndarray,
        info: dict,
        reward: float = 0.0,
        done: bool = False,
        raw_reward: float = 0.0,
        grader_score: float = 0.0,
        grader_outcome: str = "hold_progress",
    ) -> RiskPredictionObservation:
        """Convert a gym observation array + info dict → Pydantic Observation."""
        eval_result = self._gym_env._last_eval_result
        risks = eval_result["risk_breakdown"]
        fiscal_year = info["year"] if "year" in info else info["starting_year"]

        return RiskPredictionObservation(
            # Risk dimensions
            earnings_quality_risk=float(risks["earnings_quality"]),
            channel_stuffing_risk=float(risks["channel_stuffing"]),
            leverage_risk=float(risks["leverage"]),
            liquidity_risk=float(risks["liquidity"]),
            profitability_risk=float(risks["profitability"]),
            total_risk=float(eval_result["total_risk"]),
            # Risk engine summary
            risk_level=eval_result["risk_level"],
            flags=eval_result["flags"],
            # Episode context
            gvkey=str(info["gvkey"]),
            fiscal_year=int(fiscal_year),
            is_fraud_company=bool(info["is_fraud_company"]),
            step_number=self._state.step_count,
            task_id=self._task_id,
            task_difficulty=self._task_definition.difficulty,
            grader_score=grader_score,
            raw_reward=raw_reward,
            # Gym signals embedded in observation (OpenEnv convention)
            reward=reward,
            done=done,
            metadata={
                "task_id": self._task_id,
                "task_difficulty": self._task_definition.difficulty,
                "task_description": self._task_definition.description,
                "grader_outcome": grader_outcome,
                "raw_reward": raw_reward,
                "grader_score": grader_score,
            },
        )

    # ------------------------------------------------------------------
    # OpenEnv Environment interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed=None,
        episode_id=None,
        task=None,
        **kwargs,
    ) -> RiskPredictionObservation:
        self._reset_rubric()
        if task is not None:
            self._task_id = normalize_task_id(str(task))
            self._task_definition = TASK_DEFINITIONS[self._task_id]
        obs_arr, info = self._gym_env.reset(seed=seed)
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        return self._build_obs(
            obs_arr,
            info,
            reward=0.0,
            done=False,
            raw_reward=0.0,
            grader_score=0.0,
            grader_outcome="hold_progress",
        )

    def step(
        self,
        action: RiskPredictionAction,
        timeout_s=None,
        **kwargs,
    ) -> RiskPredictionObservation:
        obs_arr, raw_reward, done, truncated, info = self._gym_env.step(action.action_type)
        self._state.step_count += 1
        grader_score, grader_outcome = grade_step(
            task_id=self._task_id,
            raw_reward=float(raw_reward),
            done=done or truncated,
        )
        return self._build_obs(
            obs_arr,
            info,
            reward=grader_score,
            done=done or truncated,
            raw_reward=float(raw_reward),
            grader_score=grader_score,
            grader_outcome=grader_outcome,
        )

    @property
    def state(self) -> State:
        return self._state

    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata
        task_summary = "; ".join(
            f"{task.task_id}:{task.difficulty}" for task in TASK_DEFINITIONS.values()
        )
        return EnvironmentMetadata(
            name="Financial Fraud Detection",
            description=(
                "RL environment where an agent audits company financial statements "
                "using forensic risk scores and decides when to flag fraud. "
                f"Available graded tasks: {task_summary}."
            ),
            version="1.0.0",
        )


# =====================================================================
# Quick Test Block
# =====================================================================
if __name__ == "__main__":
    env = FinancialFraudEnv()
    obs, info = env.reset()
    print(f"Started episode with Company {info['gvkey']} (Fraudulent? {info['is_fraud_company']})")
    
    done = False
    total_reward = 0
    while not done:
        # Random agent: 10% chance to flag, 90% chance to hold
        action = np.random.choice([0, 1], p=[0.9, 0.1]) 
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        print(f"Action: {'FLAG' if action==1 else 'HOLD'} | Reward: {reward:.1f} | Done: {done}")
        
    print(f"Episode finished. Total Reward: {total_reward:.1f}")