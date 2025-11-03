# Assignment 3: User Interface for ML Models
## Absenteeism Prediction System

**Team Members:**
- Krishna Kumar Bais (241110038)
- Rohan (241110057)

**Course:** CS698Y - Human-AI Interaction

## Overview

This project extends Assignment 2 by providing a React-based user interface for the absenteeism prediction model. The interface enables HR professionals to make informed decisions while ensuring transparency, fairness, and responsible AI use.


## How to Run

### Prerequisites
- Python 3.9+
- Node.js 16+ and npm

### 1) Backend setup
```bash
cd "Assignment 4"
python3 -m venv venv
source venv/bin/activate
# On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Start the Flask API
```bash
python app.py
```

### 3) Frontend
Open a new terminal:
```bash
cd "frontend"
npm install
npm run dev
```

### 4) Evaluation Script
To evaluate explainability quality metrics:
```bash
# Activate virtual environment first
source venv/bin/activate
python scripts/eval_explainability.py
```

This generates:
- `explain_eval.json`: LIME fidelity, SHAP stability, and fairness gaps
- `fairness_table.csv`: MAE by sensitive group

## Model Performance

- **RMSE**: 11.43 hours
- **MAE**: 6.44 hours
- **R² Score**: -0.20

### Fairness Metrics
- **Age Group MAE Gap**: 20.78 hours (Poor)
- **Education MAE Gap**: 13.36 hours (Moderate)
- **Service Time MAE Gap**: 3.76 hours (Good)

## Explainability

The system provides comprehensive explainability features to understand model predictions using SHAP, LIME, and counterfactual analysis. Explanations address key user questions: **WHY** (feature contributions), **WHICH** (global importance), and **WHAT IF** (counterfactual suggestions).

### Frontend Usage

1. **Make a Prediction**: Fill out the employee information form and click "Predict Absenteeism"
2. **View Explanations**: After prediction, the Explanation panel appears automatically with SHAP feature contributions
3. **Explore Alternatives**: 
   - Click "Show LIME" for local model explanations
   - Click "Show Counterfactual" for suggested changes
   - Use "What-If" sliders to explore different scenarios
4. **Global Dashboard**: Navigate to "Explainability Dashboard" tab to see overall feature importance

### Privacy & Security

- **No PII Returned**: Explanations only include feature names and values, no employee identifiers
- **Anonymized Outputs**: All explanations are anonymized and sanitized
- **Synthetic Background Data**: Global explanations use synthetic samples, not actual employee records

## Explainability API

The API provides explainability endpoints to understand model predictions using SHAP, LIME, and counterfactual analysis.

### GET /explain/global

Returns global feature importance across the dataset using SHAP values.

```bash
curl http://localhost:5000/explain/global
```

**Response:**
```json
{
  "feature_importance": [
    {"feature": "Service time", "mean_abs_shap": 2.45},
    {"feature": "Age", "mean_abs_shap": 1.89}
  ],
  "explainer_type": "LinearExplainer",
  "sample_size": 100,
  "cached": false
}
```

### POST /explain/local

Returns local SHAP values for a specific prediction, showing how each feature contributes.

```bash
curl -X POST http://localhost:5000/explain/local \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "Age": 35,
      "Education": 2,
      "Service time": 5,
      "Work load Average/day ": 270.0,
      "Transportation expense": 200,
      "Distance from Residence to Work": 10,
      "Social drinker": 0,
      "Social smoker": 0,
      "Pet": 1,
      "Son": 1,
      "Hit target": 1,
      "Month of absence": 6,
      "Day of the week": 2,
      "Seasons": 2,
      "Reason for absence": 0,
      "Disciplinary failure": 0
    }
  }'
```

**Response:**
```json
{
  "prediction": 8.5,
  "contributions": [
    {"feature": "Service time", "shap": 1.2, "value": 0.5},
    {"feature": "Age", "shap": -0.8, "value": 0.3}
  ],
  "text_summary": "Predicted 8.50 hours. Top factors: Service time increases prediction by 1.20 hours, Age decreases prediction by 0.80 hours."
}
```

### POST /explain/lime

Returns LIME explanation highlighting the most influential features for a specific prediction.

```bash
curl -X POST http://localhost:5000/explain/lime \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "Age": 35,
      "Education": 2,
      "Service time": 5,
      "Work load Average/day ": 270.0,
      "Transportation expense": 200,
      "Distance from Residence to Work": 10,
      "Social drinker": 0,
      "Social smoker": 0,
      "Pet": 1,
      "Son": 1,
      "Hit target": 1,
      "Month of absence": 6,
      "Day of the week": 2,
      "Seasons": 2,
      "Reason for absence": 0,
      "Disciplinary failure": 0
    }
  }'
```

**Response:**
```json
{
  "prediction": 8.5,
  "top_features": [
    {"feature": "Service time", "weight": 1.2},
    {"feature": "Age", "weight": -0.8}
  ],
  "explanation_score": 0.85
}
```

### POST /explain/cf

Generates counterfactual suggestions showing minimal changes needed to reduce predicted absenteeism.

```bash
curl -X POST http://localhost:5000/explain/cf \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "Age": 35,
      "Education": 2,
      "Service time": 5,
      "Work load Average/day ": 270.0,
      "Transportation expense": 200,
      "Distance from Residence to Work": 10,
      "Social drinker": 0,
      "Social smoker": 0,
      "Pet": 1,
      "Son": 1,
      "Hit target": 1,
      "Month of absence": 6,
      "Day of the week": 2,
      "Seasons": 2,
      "Reason for absence": 0,
      "Disciplinary failure": 0
    },
    "target": 0.8
  }'
```

**Response:**
```json
{
  "original_prediction": 10.0,
  "target_prediction": 8.0,
  "candidates": [
    {
      "feature": "Work load Average/day ",
      "original_value": 270.0,
      "suggested_value": 250.0,
      "change": -20.0,
      "new_prediction": 8.5,
      "reduction_percent": 15.0,
      "distance": 0.5
    }
  ]
}
```

## Testing Explainability Endpoints

### Using curl

```bash
# Test global explanation
curl http://localhost:5000/explain/global

# Test local explanation (replace with your data)
curl -X POST http://localhost:5000/explain/local \
  -H "Content-Type: application/json" \
  -d '{"input": {"Age": 35, "Education": 2, ...}}'

# Test LIME explanation
curl -X POST http://localhost:5000/explain/lime \
  -H "Content-Type: application/json" \
  -d '{"input": {...}}'

# Test counterfactual
curl -X POST http://localhost:5000/explain/cf \
  -H "Content-Type: application/json" \
  -d '{"input": {...}, "target": 0.8}'
```

### Using Python

```python
import requests

# Global explanation
response = requests.get('http://localhost:5000/explain/global')
print(response.json())

# Local explanation
response = requests.post('http://localhost:5000/explain/local', json={
    'input': {
        'Age': 35,
        'Education': 2,
        # ... other fields
    }
})
print(response.json())
```

## Documentation

For detailed explainability documentation, see:
- **`Report_Assignment_6.md`**: Comprehensive report on explainability techniques, evaluation results, and examples

## Deployment (Free options)

### Option A) Render (Docker) - Free Web Service

1. Push this repository to GitHub.
2. Ensure `Dockerfile` exists at repo root (already done). It builds the frontend and runs the Flask app with gunicorn.
3. In Render:
   - New → Web Service → "Build and deploy from a Git repository"
   - Select your repo
   - Choose "Docker" as the runtime
   - Environment: `PORT=5000`
   - Health Check Path: `/api/health`
   - Click Create Web Service
4. After build completes, open the URL. The frontend is served by Flask from `frontend/dist` and the backend lives under `/api` and `/explain`.

Notes:
- The Dockerfile is multi-stage: it builds the React frontend (Node 18) then runs the Python app (Python 3.9) with `gunicorn app:app`.
- CORS is enabled for `/api/*` and `/explain/*`.

### Option B) Railway (Docker) - Free Tier

1. Push the repo to GitHub.
2. In Railway:
   - New Project → Deploy from GitHub → select this repo
   - It will detect the Dockerfile and build
   - Set variable `PORT=5000`
   - Expose the web service; open the app once deployed

### Local Docker Run (optional)

```bash
docker build -t absenteeism-app .
docker run -p 5000:5000 -e PORT=5000 absenteeism-app
# Open http://localhost:5000
```

If you see a blank page after deployment, verify the build completed and that `frontend/dist` is present inside the container (Dockerfile copies it automatically).
