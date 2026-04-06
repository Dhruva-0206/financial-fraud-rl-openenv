---
title: Risk Prediction Environment Server
emoji: 📊
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Financial Fraud Risk Prediction (OpenEnv)

This environment simulates a real-world financial fraud auditing workflow.
An agent inspects rolling 4-year company fundamentals and decides whether to:

- `HOLD` (continue monitoring), or
- `FLAG` (report suspected fraud).

It is built for OpenEnv and supports HTTP + WebSocket usage, Docker deployment,
and Hugging Face Spaces publishing.

## OpenEnv Spec Coverage

This environment implements the full expected OpenEnv structure:

- Typed models in `models.py`
- Environment server in `server/risk_prediction_environment.py`
- API wrapper in `server/app.py`
- Manifest in `openenv.yaml`
- Typed client in `client.py`

Core endpoints exposed by the server:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`
- `WS /ws`

## Tasks and Agent Graders

Three graded tasks are provided (easy → medium → hard):

| Task | Difficulty | Purpose | Reward Range |
|------|------------|---------|--------------|
| `easy` | easy | onboarding task with generous partial-progress credit | `[0.0, 1.0]` |
| `medium` | medium | balanced benchmark task for default submissions | `[0.0, 1.0]` |
| `hard` | hard | strict production-style grading | `[0.0, 1.0]` |

Task graders are implemented in `task_graders.py` and applied server-side in
`RiskPredictionEnvironment.step()`.

## Action and Observation Spaces

### Action
`RiskPredictionAction`

- `action_type: int`
  - `0` = HOLD
  - `1` = FLAG

### Observation
`RiskPredictionObservation`

Includes core fraud-risk features and grader context:

- Risk scores: `earnings_quality_risk`, `channel_stuffing_risk`, `leverage_risk`, `liquidity_risk`, `profitability_risk`, `total_risk`
- Risk summary: `risk_level`, `flags`
- Episode context: `gvkey`, `fiscal_year`, `is_fraud_company`, `step_number`
- Grader fields: `task_id`, `task_difficulty`, `grader_score`, `raw_reward`
- OpenEnv fields: `reward`, `done`, `metadata`

## Reward Design

The environment computes an underlying domain reward (`raw_reward`) and then
maps it through task-specific graders to normalized `reward` in `[0.0, 1.0]`.

This provides:

- Meaningful domain outcomes (true positives, false positives, missed fraud, etc.)
- Partial-progress signals each step
- Comparable normalized scores across task difficulty levels

## Quick Start (Local)

```bash
cd risk_prediction

# Install dependencies (recommended)
uv sync

# Validate environment structure
openenv validate

# Run server locally
uv run server
```

Health check:

```bash
curl http://localhost:8000/health
```

## Reproducible Baseline Inference

The baseline script is `inference.py` and supports deterministic replay:

```bash
INFERENCE_SEED=42 \
RISK_PREDICTION_TASK=medium \
python inference.py
```

Required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Optional:

- `SERVER_URL` (defaults to deployed Space URL)
- `LOCAL_IMAGE_NAME` (uses `from_docker_image()` path)
- `RISK_PREDICTION_TASK` (`easy`, `medium`, `hard`)

## Docker

Build and run:

```bash
docker build -t risk_prediction-env:latest -f server/Dockerfile .
docker run -p 8000:8000 risk_prediction-env:latest
```

## Hugging Face Spaces Deployment

From this directory (where `openenv.yaml` exists):

```bash
openenv push --repo-id <username>/risk-prediction
```

Then validate deployment:

```bash
bash scripts/validate-submission.sh https://<username>-risk-prediction.hf.space
```

## Project Layout

```
risk_prediction/
├── client.py
├── inference.py
├── models.py
├── openenv.yaml
├── task_graders.py
├── risk_engine_v5.py
├── server/
│   ├── app.py
│   └── risk_prediction_environment.py
└── data/
    └── fraud_dataset.csv
```
