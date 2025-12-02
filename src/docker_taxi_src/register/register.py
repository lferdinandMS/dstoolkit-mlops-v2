"""Register a trained model in MLflow with fail-fast diagnostics."""
from pathlib import Path
import argparse
import json
import os
import sys
import traceback
import mlflow
from importlib import metadata


def _print_versions() -> None:
    print("=" * 50)
    print("PACKAGE VERSIONS:")
    try:

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


def _get_tracking_uri_safe() -> str | None:
    """Safely retrieve the MLflow tracking URI or return None if unavailable."""
    try:
        return mlflow.get_tracking_uri()
    except Exception:  # noqa: BLE001
        return None


def _print_registration_diagnostics(model_name: str, run_uri: str) -> None:
    """Print structured diagnostics about model registration context."""
    run_id, parsed_subpath = _parse_run_uri(run_uri)
    tracking_uri = _get_tracking_uri_safe()
    print("-" * 50)
    print("REGISTRATION DIAGNOSTICS")
    print(f"tracking_uri: {tracking_uri}")
    print(f"model_name: {model_name}")
    print(f"run_uri: {run_uri}")
    print(f"parsed_run_id: {run_id}")
    print(f"parsed_subpath: {parsed_subpath}")
    print(f"MLFLOW_TRACKING_URI env: {os.environ.get('MLFLOW_TRACKING_URI')}")
    print("-" * 50)


def _ensure_artifact_or_exit(run_uri: str) -> None:
    """Verify required 'model' artifact exists or exit with failure code 2."""
    if _artifact_exists_for_run(run_uri, artifact_subpath="model"):
        return
    run_id, _ = _parse_run_uri(run_uri)
    try:
        client = mlflow.MlflowClient()
        root_list = client.list_artifacts(run_id) if run_id else []
        model_list = client.list_artifacts(run_id, path="model") if run_id else []
        print("Artifacts at run root:")
        for a in root_list:
            print(f"  - {a.path}")
        print("Artifacts under 'model':")
        for a in model_list:
            print(f"  - {a.path}")
    except Exception as diag_err:  # noqa: BLE001
        print(f"Artifact diagnostic listing failed: {diag_err}")
    print("ERROR: No 'model' artifact found for run; failing registration.")
    sys.exit(2)


def _register_and_tag(run_uri: str, model_name: str, tags: dict) -> None:
    """Register the MLflow model and apply tags; exit non-zero on failure."""
    try:
        mv = mlflow.register_model(run_uri, model_name)
        _set_model_tags(model_name, mv.version, tags)
        print(mv)
    except Exception as reg_err:  # noqa: BLE001
        print("Registration failed with exception:")
        print(str(reg_err))
        print("TRACEBACK START")
        print(traceback.format_exc())
        print("TRACEBACK END")
        sys.exit(1)


def main(model_metadata: str, model_name: str, score_report: str, build_reference: str) -> None:
    """Register a model version in MLflow.

    Reads run metadata (containing the MLflow run URI) and score report, emits
    diagnostics, verifies the presence of the MLflow 'model' artifact, and then
    registers and tags the model. Exits with non-zero status when prerequisites
    are missing or registration fails so CI surfaces the issue.

    Parameters:
        model_metadata: Path to JSON with run metadata (includes run_uri).
        model_name: Target MLflow model container name (branch-derived).
        score_report: Directory containing score.txt with metrics.
        build_reference: Build identifier used for tagging.
    """
    _print_versions()
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

    _print_registration_diagnostics(model_name, run_uri)
    _ensure_artifact_or_exit(run_uri)
    _register_and_tag(run_uri, model_name, tags)


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
