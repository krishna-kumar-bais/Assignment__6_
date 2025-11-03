<<FRONTEND_PLAN>>

## Frontend Explainability Implementation Plan

### Current State
- Single-page React app with Material-UI
- Uses axios for API calls
- No routing library (single App.jsx component)
- API endpoints: `/api/predict`, `/api/model_info`, `/api/feature_importance`
- Explain endpoints: `/explain/global`, `/explain/local`, `/explain/lime`, `/explain/cf`

### New Files to Create

1. **`frontend/src/components/ExplainPanel.jsx`**
   - Accepts `inputValues` props (form data)
   - Button "Explain" → POST `/explain/local`
   - Display SHAP values as horizontal bar chart (top 6 features)
   - Display `text_summary`
   - Buttons: "Show LIME", "Show Counterfactual" → call respective endpoints
   - What-if sliders for top 2 numeric global features (Work load, Distance)
   - Dynamic updates: On slider change → `/predict` + `/explain/local`

2. **`frontend/src/pages/ExplainDashboard.jsx`**
   - On mount: GET `/explain/global`
   - Display mean_abs_SHAP feature importance as vertical bar chart
   - Optionally display fairness gaps from model_info

3. **`frontend/src/components/Navigation.jsx`** (optional - simple tabs)
   - Or use MUI Tabs in App.jsx
   - Links: "Predict", "Explainability Dashboard"

### Files to Modify

1. **`frontend/package.json`**
   - Add: `react-router-dom@^6.8.0`
   - Add: `recharts@^2.10.0` (for charts)
   - Add: `@types/recharts` (dev dependency, optional)

2. **`frontend/src/App.jsx`**
   - Add routing using react-router-dom
   - Create routes:
     - `/` → Main prediction form (current page)
     - `/dashboard` → ExplainDashboard component
   - Add navigation/tabs to switch between routes
   - Integrate ExplainPanel into main prediction page (below prediction result)

3. **`frontend/vite.config.js`**
   - No changes needed

### NPM Packages to Install

```json
{
  "react-router-dom": "^6.8.0",
  "recharts": "^2.10.0"
}
```

### Implementation Details

**ExplainPanel.jsx:**
- Use Recharts `BarChart` for horizontal SHAP visualization
- Use MUI `Slider` components for what-if scenarios
- State management: `localExplanation`, `limeExplanation`, `counterfactualExplanation`
- Error handling with Alert components

**ExplainDashboard.jsx:**
- Use Recharts `BarChart` (vertical) for global importance
- Fetch global explanation on component mount
- Display cached status if available
- Show top 10-15 features

**Routing:**
- Simple tab-based navigation (MUI Tabs)
- Or use BrowserRouter with Link components
- Keep current prediction form as default route

### Evaluation Script

**`scripts/eval_explainability.py`** (new)
- Evaluate LIME fidelity (R² of local surrogate for 50 samples)
- Evaluate SHAP stability (top-1 feature consistency under noise)
- Compute fairness gaps (MAE by sensitive feature)
- Outputs: `explain_eval.json`, `fairness_table.csv`

### Documentation

**`Report_Assignment_6.md`** (new)
- Overview of explainability techniques (SHAP, LIME, CF)
- User questions answered (WHY, WHY NOT, WHAT IF)
- Global + Local explanation examples
- Evaluation summary (import from JSON/CSV)
- Privacy/fairness notes

**`README.md`** (update)
- Add explainability section
- How to test `/explain` endpoints
- How to run evaluation script
- Privacy note about anonymized explanations

### Testing Plan

1. Component rendering: Verify ExplainPanel and ExplainDashboard render
2. API integration: Test all explain endpoints work from frontend
3. Charts: Verify Recharts displays data correctly
4. What-if sliders: Test dynamic updates trigger API calls
5. Evaluation script: Run and verify output files

### File Structure After Changes

```
frontend/
├── src/
│   ├── components/
│   │   └── ExplainPanel.jsx (NEW)
│   ├── pages/
│   │   └── ExplainDashboard.jsx (NEW)
│   ├── App.jsx (MODIFY - add routing)
│   └── main.jsx (no change)
├── package.json (MODIFY - add dependencies)
└── vite.config.js (no change)

scripts/
└── eval_explainability.py (NEW)

Report_Assignment_6.md (NEW)
README.md (MODIFY - add explainability docs)
```

