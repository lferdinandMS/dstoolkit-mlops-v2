import os
import sys
import tempfile
import shutil
from importlib import metadata


def main() -> int:
    try:
        import mlflow  # noqa: F401
    except Exception as e:
        print(f"Failed to import mlflow: {e}")
        return 1

    # Versions
    def v(name: str):
        try:
            return metadata.version(name)
        except Exception:
            return "<not-installed>"

    print("MLFLOW COMPAT SMOKE")
    print(f"mlflow: {v('mlflow')}")
    print(f"mlflow-skinny: {v('mlflow-skinny')}")
    print(f"azureml-mlflow: {v('azureml-mlflow')}")

    # Use a local file-based tracking directory to avoid needing a server/DB
    tmpdir = tempfile.mkdtemp(prefix="mlflow_smoke_")
    tracking_uri = f"file://{tmpdir}"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    print(f"Tracking URI: {tracking_uri}")

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        from mlflow import artifacts as mlfa
        import pathlib

        # Ensure the runtime uses the explicit tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Create or get a dedicated experiment to avoid 'experiment 0 not found'
        client = MlflowClient()
        exp_name = "smoke-compat"
        exp = client.get_experiment_by_name(exp_name)
        exp_id = exp.experiment_id if exp else client.create_experiment(exp_name)

        with mlflow.start_run(experiment_id=exp_id, run_name="compat-smoke") as run:
            run_id = run.info.run_id
            # Log a tiny artifact under the 'model' path to simulate our use-case
            model_dir = pathlib.Path(tmpdir) / "_tmp_model_artifacts"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "README.txt").write_text("smoke artifact", encoding="utf-8")
            mlflow.log_artifacts(str(model_dir), artifact_path="model")

            # Ensure listing works without plugin signature errors (local filestore)
            arts = client.list_artifacts(run_id, path="model")
            print("Artifacts under 'model':", [a.path for a in arts])
            if not arts:
                print("No artifacts found under 'model' path")
                return 2

        # Explicitly exercise azureml ArtifactRepository constructor only to detect
        # signature mismatches. We instantiate via get_artifact_repository but do not
        # call any methods, avoiding network/URI validation.
        test_uri = (
            "azureml://subscriptions/00000000-0000-0000-0000-000000000000/"
            "resourceGroups/rg/providers/Microsoft.MachineLearningServices/"
            "workspaces/ws/experiments/exp/runs/run/artifacts/model"
        )
        try:
            repo = mlfa.get_artifact_repository(test_uri)  # type: ignore[attr-defined]
            print(
                "azureml:// artifact repository instantiated:",
                repo.__class__.__name__,
            )
        except TypeError as te:
            msg = str(te)
            print("azureml:// artifact repository TypeError:", msg)
            if "__init__()" in msg and "positional argument" in msg:
                # Specific signature mismatch that breaks our register step
                return 3
            # Unknown TypeError: treat as failure to be safe
            return 3
        except Exception as e:
            # Any other exception during instantiation is considered a failure since
            # we are only constructing, not performing operations.
            print("azureml:// artifact repository instantiation failed:", repr(e))
            return 4

        return 0
    except Exception as e:
        print("Smoke test failed:", e)
        return 1
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
