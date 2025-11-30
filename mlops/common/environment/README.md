Environment version rationale

- Python: 3.10 — stable with MLflow 2.14.x and numpy 1.26.x wheels.
- MLflow: 2.14.3 — project-wide pin for tracking/logging consistency.
- azureml-mlflow: >=1.6.3,<2.0.0 — range chosen to align with curated environments and avoid plugin/MLflow constructor mismatches while remaining compatible with MLflow 2.14.3.
- numpy: 1.26.4 — compatible with Python 3.10 and MLflow constraints.
- pandas: 2.2.3 — widely compatible with numpy 1.26.x and sklearn 1.1–1.3.
- scikit-learn: 1.3.2 (components) / 1.1.3 (training) — both compatible; prefer 1.3.2 in component envs; training can use 1.1.3 to satisfy older pipelines if needed.

Guidelines
- Keep `mlflow==2.14.3` and `azureml-mlflow>=1.6.3,<2.0.0` aligned across envs to prevent AML artifact logging errors.
- For Python >=3.11 contexts, prefer `numpy==2.x` via conditional markers; otherwise stick to `1.26.4`.
- If curated AML images introduce plugin mismatches, training code includes a fallback to save a local pickle when `mlflow.log_model` fails.
- When updating versions, validate locally: `pip install -r mlops/common/environment/training_requirements.txt` and run a quick training smoke test.
