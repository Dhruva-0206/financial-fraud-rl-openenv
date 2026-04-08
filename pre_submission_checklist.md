# Pre-Submission Checklist

* **HF Space deploys**
  * Automated ping to the Space URL — must return 200 and respond to `reset()`
* **OpenEnv spec compliance**
  * Validate `openenv.yaml`, typed models, and `step()`/`reset()`/`state()` endpoints
* **Dockerfile builds**
  * Automated docker build on the submitted repo
* **Baseline reproduces**
  * Run the submitted inference script — must complete without error and produce scores
* **3+ tasks with graders**
  * Enumerate tasks, run each grader, and verify scores/reward are in the 0.0–1.0 range

---

## Mandatory Additional Instructions

Before submitting, ensure the following variables are defined in your environment configuration:
* `API_BASE_URL`: The API endpoint for the LLM.
* `MODEL_NAME`: The model identifier to use for inference.
* `API_KEY`: Your validator-injected API key.

**Requirements:**
* The inference script must be named `inference.py` and placed in the root directory of the project.
* Participants must use the OpenAI Client for all LLM calls using the variables listed above.
* Participants must emit structured stdout logs strictly following the `[START]`, `[STEP]`, and `[END]` format defined in the sample `inference.py` provided. Any deviation in field names, ordering, or formatting will result in incorrect evaluation scoring. Refer to the Sample Inference Script for the complete format specification and examples.

---

## Infra Restrictions

* Runtime of the inference script should be less than 20 minutes.
* Make sure your environment and inference script can run on a machine with `vcpu=2` and `memory=8gb`.

---

## Validator

* Run the pre-submission validation script before submitting.