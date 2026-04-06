## Setup Instructions

### 1. Clone the repository
git clone https://github.com/Dhruva-0206/financial-fraud-rl-openenv.git
cd financial-fraud-rl-openenv

### 2. Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Set environment variables
set API_BASE_URL=https://router.huggingface.co/v1
set HF_TOKEN=your_huggingface_token
set MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct

### 5. Start the OpenEnv server
python -m risk_prediction.server.app

### 6. Run inference agent (in another terminal)
python -m risk_prediction.inference

### 7. Run tests
pytest
