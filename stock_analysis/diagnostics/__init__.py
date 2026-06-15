"""
Opt-in statistical diagnostics for research workflows.

Use cases:
- Stationarity checks: ADF tests on feature columns before classifier training
  (non-stationary inputs can leak spurious signal into linear models)
- Regression diagnostics: OLS summary on feature→target relationships to
  sanity-check linear assumptions before fitting classifiers

NOT imported by the daily job, Flask API, or paper-trading modules.
Install statsmodels (already a project dependency) to use these helpers.
"""
