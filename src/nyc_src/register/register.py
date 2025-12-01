"""Register a trained model in MLflow with minimal complexity."""
from pathlib import Path
import argparse
import json
import mlflow


def _print_versions() -> None:
    print("=" * 50)
    print("PACKAGE VERSIONS:")
    try:
        from importlib import metadata
        print(f"mlflow: {metadata.version('mlflow')}")
        try:
            print(f"azureml-mlflow: {metadata.version('azureml-mlflow')}")
        except metadata.PackageNotFoundError:
            print("azureml-mlflow: not-installed (curated plugin may be present)")
    except Exception as e:  # noqa: BLE001
        print(f"Could not get package versions: {e}")
    print("=" * 50)


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_run_uri(run_uri: str):
    """Return (run_id, subpath) for URIs like runs:/<run_id>/model."""
    if not run_uri.startswith("runs:/"):
        return None, None
    parts = run_uri.split("/")
    run_id = parts[1] if len(parts) > 1 else None
    subpath = "/".join(parts[2:]) if len(parts) > 2 else None
    return run_id, subpath


def _artifact_exists_for_run(run_uri: str, artifact_subpath: str = "model") -> bool:
    run_id, subpath = _parse_run_uri(run_uri)
    if not run_id:
        return False
    target = artifact_subpath if subpath in (None, "", artifact_subpath) else subpath
    try:
        arts = mlflow.MlflowClient().list_artifacts(run_id, path=target)
        return bool(arts)
    except Exception as e:  # noqa: BLE001
        print(f"Artifact listing failed for run {run_id}: {e}")
        return False


def _set_model_tags(model_name: str, version: str, tags: dict) -> None:
    client = mlflow.MlflowClient()
    for k, v in tags.items():
        client.set_model_version_tag(name=model_name, version=version, key=k, value=v)


def main(model_metadata: str, model_name: str, score_report: str, build_reference: str) -> None:
    """Register an MLflow model version for a given run and tag it.

    Reads run metadata and score report, validates the presence of the
    MLflow "model" artifact under the run, and registers the model to
    the registry with relevant tags (mse, coff, cod, build_id).
    """
    _print_versions()

    # Ensure MLflow tracking is hooked into Azure ML
    try:
        mlflow.set_tracking_uri(mlflow.get_tracking_uri())
    except Exception as _e:  # noqa: BLE001
        print(f"mlflow.set_tracking_uri hook failed: {_e}")

    meta = _read_json(Path(model_metadata))
    run_uri = meta["run_uri"]

    score = _read_json(Path(score_report) / "score.txt")
    tags = {
        "mse": score.get("mse"),
        "coff": score.get("coff"),
        "cod": score.get("cod"),
        "build_id": build_reference,
    }

    # Guard: only register if the MLflow "model" artifact exists for this run
    if not _artifact_exists_for_run(run_uri, artifact_subpath="model"):
        print("No 'model' artifact found for run; skipping MLflow registration.")
        return

    mv = mlflow.register_model(run_uri, model_name)
    _set_model_tags(model_name, mv.version, tags)
    print(mv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("register_model")
    parser.add_argument(
        "--model_metadata",
        type=str,
        help="model metadata on Machine Learning Workspace",
    )
    parser.add_argument("--model_name", type=str, help="model name to be registered")
    parser.add_argument("--score_report", type=str, help="score report for the model")
    parser.add_argument(
        "--build_reference",
        type=str,
        help="Original AzDo build id that initiated experiment",
    )

    args = parser.parse_args()

    print(args.model_metadata)
    print(args.model_name)
    print(args.score_report)
    print(args.build_reference)

    main(args.model_metadata, args.model_name, args.score_report, args.build_reference)
