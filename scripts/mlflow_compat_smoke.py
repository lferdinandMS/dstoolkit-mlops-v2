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
        import pathlib

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            # Log a tiny artifact under the 'model' path to simulate our use-case
            model_dir = pathlib.Path(tmpdir) / "_tmp_model_artifacts"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "README.txt").write_text("smoke artifact", encoding="utf-8")
            mlflow.log_artifacts(str(model_dir), artifact_path="model")

            # Ensure listing works without plugin signature errors
            client = MlflowClient()
            arts = client.list_artifacts(run_id, path="model")
            print("Artifacts under 'model':", [a.path for a in arts])
            if not arts:
                print("No artifacts found under 'model' path")
                return 2

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
