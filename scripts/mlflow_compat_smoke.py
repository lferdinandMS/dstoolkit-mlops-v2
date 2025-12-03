"""MLflow/AzureML compatibility smoke test.

This module verifies two things:
- Local file-backed MLflow logging and artifact listing work.
- The azureml ArtifactRepository constructor is compatible (no positional-arg error).
"""

import os
import sys
import tempfile
import shutil
import pathlib
from importlib import metadata

import mlflow
from mlflow.tracking import MlflowClient
from mlflow import artifacts as mlfa


def print_versions() -> None:
    """Print relevant package versions for visibility."""

    def v(name: str) -> str:
        try:
            return metadata.version(name)
        except Exception:
            return "<not-installed>"

    print("MLFLOW COMPAT SMOKE")
    print(f"mlflow: {v('mlflow')}")
    print(f"mlflow-skinny: {v('mlflow-skinny')}")
    print(f"azureml-mlflow: {v('azureml-mlflow')}")


def local_artifact_smoke(tracking_uri: str, tmpdir: str) -> int:
    """Run a local MLflow artifact log/list smoke test.

    Returns 0 on success, 2 if the artifact under 'model' is missing, or 1 on error.
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        exp_name = "smoke-compat"
        exp = client.get_experiment_by_name(exp_name)
        exp_id = exp.experiment_id if exp else client.create_experiment(exp_name)

        with mlflow.start_run(experiment_id=exp_id, run_name="compat-smoke") as run:
            run_id = run.info.run_id
            model_dir = pathlib.Path(tmpdir) / "_tmp_model_artifacts"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "README.txt").write_text("smoke artifact", encoding="utf-8")
            mlflow.log_artifacts(str(model_dir), artifact_path="model")

            arts = client.list_artifacts(run_id, path="model")
            print("Artifacts under 'model':", [a.path for a in arts])
            if not arts:
                print("No artifacts found under 'model' path")
                return 2
        return 0
    except Exception as e:
        print("Local artifact smoke failed:", e)
        return 1


def azureml_repo_constructor_smoke() -> int:
    """Exercise azureml ArtifactRepository constructor and return status code.

    Returns:
    - 0 on success
    - 3 on positional-argument TypeError (signature mismatch)
    - 4 on any other constructor-time exception
    """
    test_uri = (
        "azureml://subscriptions/00000000-0000-0000-0000-000000000000/"
        "resourceGroups/rg/providers/Microsoft.MachineLearningServices/"
        "workspaces/ws/experiments/exp/runs/run/artifacts/model"
    )
    try:
        repo = mlfa.get_artifact_repository(test_uri)  # type: ignore[attr-defined]
        print("azureml:// artifact repository instantiated:", repo.__class__.__name__)
        return 0
    except TypeError as te:
        msg = str(te)
        print("azureml:// artifact repository TypeError:", msg)
        if "__init__()" in msg and "positional argument" in msg:
            return 3
        return 3
    except Exception as e:
        print("azureml:// artifact repository instantiation failed:", repr(e))
        return 4


def main() -> int:
    """Entry point: run version print, local smoke, and azureml constructor smoke."""
    print_versions()
    tmpdir = tempfile.mkdtemp(prefix="mlflow_smoke_")
    tracking_uri = f"file://{tmpdir}"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    print(f"Tracking URI: {tracking_uri}")

    try:
        local_code = local_artifact_smoke(tracking_uri, tmpdir)
        if local_code != 0:
            return local_code

        azureml_code = azureml_repo_constructor_smoke()
        if azureml_code != 0:
            return azureml_code
        return 0
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
