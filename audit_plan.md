# Explainability Integration Audit Plan

## Repository Structure Summary

### Files Safe to Modify

1. **`app.py`** (Backend Flask API)
   - **Lines to modify**: Add explainability endpoints (~100-180, new routes)
   - **Risk**: Low - Adding new API endpoints won't break existing functionality
   - **Changes**: Add `/api/explain`, `/api/shap_values`, `/api/lime_explanation` endpoints

2. **`requirements.txt`**
   - **Risk**: Low - Adding new dependencies
   - **Changes**: Add `shap`, `lime`, `numba` (SHAP dependency)

3. **`frontend/src/App.jsx`**
   - **Risk**: Medium - UI changes need careful testing
   - **Changes**: Add explainability visualization components (SHAP plots, LIME tables)

4. **`frontend/package.json`** (if needed for visualization libraries)
   - **Risk**: Low - May need chart libraries like `recharts` or `plotly.js`

### Files NOT to Modify (Read-Only)

- `saving_model.py` - Training script, keep as-is
- `model.pkl` / `trained_absenteeism_model.pkl` - Model artifacts, read-only
- `Dockerfile`, `render.yaml` - Deployment configs (may need minor updates)
- `README.md` - Documentation (may update later)

### Missing Resources Identified

1. **Training Data (`X_train`, `y_train`)**: 
   - Not saved separately - only model artifacts exist
   - **Impact**: SHAP KernelExplainer needs background dataset (can use sample data or saved samples)
   - **Solution**: Load model, generate sample background data from training script OR use smaller background sample

2. **No sklearn Pipeline**: 
   - Preprocessing is manual (StandardScaler + pandas get_dummies in `preprocess_input()`)
   - **Impact**: Need to ensure SHAP/LIME use same preprocessing pipeline
   - **Solution**: Use existing `preprocess_input()` function and ensure consistency

3. **No Test Suite**: 
   - No `tests/` folder or test files found
   - **Impact**: Manual testing required for explainability features
   - **Solution**: Manual testing or create basic integration tests later

## High-Level 3-Step Plan

### Step 1: Backend Explainability Endpoints
**Goal**: Add SHAP and LIME explanation endpoints to Flask API

**Tasks**:
1. Install explainability libraries (`shap`, `lime`)
2. Create `explainability.py` helper module with:
   - SHAP explainer (LinearExplainer for Linear Regression - fast, exact)
   - LIME explainer for tabular data
   - Background data generation/sampling utility
3. Add API endpoints in `app.py`:
   - `POST /api/explain/shap` - Returns SHAP values for a prediction
   - `POST /api/explain/lime` - Returns LIME explanation for a prediction
   - `GET /api/explain/background_sample` - Returns sample background data info

**Files Modified**: `app.py`, `requirements.txt`, new `explainability.py`

### Step 2: Frontend Explainability UI
**Goal**: Add visualization components to display explanations

**Tasks**:
1. Add explainability state management in `App.jsx`
2. Create explanation display components:
   - SHAP feature importance bar/waterfall chart
   - LIME feature contribution table
   - Interactive toggles for different explanation methods
3. Integrate explanation API calls after prediction

**Files Modified**: `frontend/src/App.jsx`, potentially `frontend/package.json` if adding chart libs

### Step 3: Testing & Optimization
**Goal**: Ensure explainability works correctly and optimize performance

**Tasks**:
1. Test with sample predictions
2. Implement caching for SHAP/LIME computations (expensive operations)
3. Add error handling for explainability endpoints
4. Document usage in code comments

**Files Modified**: `app.py` (caching), `explainability.py` (optimization)

## Estimated Risky Operations

### High Risk - Need Caching
1. **SHAP KernelExplainer**: 
   - Would be computationally expensive
   - **Mitigation**: Use `LinearExplainer` instead (Linear Regression → exact, fast calculations)
   - Alternative: Cache SHAP values for common input patterns

2. **LIME Explanation**: 
   - Requires sampling and model inference (moderate compute)
   - **Mitigation**: Limit number of samples (default 5000, reduce to 1000-2000)
   - Cache results for identical inputs (simple hash-based cache)

### Medium Risk
1. **Background Dataset for SHAP**:
   - Model doesn't save training data
   - **Solution**: Generate representative sample from feature distributions OR save small sample during training
   - Use 100-200 samples (sufficient for LinearExplainer)

2. **Preprocessing Consistency**:
   - Must match exact preprocessing in `preprocess_input()`
   - **Solution**: Reuse `preprocess_input()` function, ensure SHAP/LIME get preprocessed data

## Implementation Notes

- **Linear Regression Advantage**: Since model is LinearRegression, SHAP `LinearExplainer` will be exact and very fast (no approximation needed)
- **Feature Count**: Model has many one-hot encoded features (~50+ after encoding) - ensure explainers handle this
- **API Response Format**: Return JSON-serializable explanations (SHAP values as arrays, LIME as dict/array)

## Dependencies to Add

```txt
shap>=0.42.0
lime>=0.2.0.1
```

Optional for frontend visualization:
- `recharts` (React charts) or
- `plotly.js` (interactive plots)

