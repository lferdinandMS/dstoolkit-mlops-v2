"""Register machine learning models with MLflow, with AML fallback."""
import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

from importlib import metadata
import mlflow


def main(model_metadata, model_name, score_report, build_reference):
    """
    Register the model and assign tags to it.

    Parameters:
      model_metadata (str): model information from previous steps
      model_name (str): model name
      score_report (str): a report from te validation (score) step
      build_reference (str): a build id
    """
    try:
        _print_versions()

        mm_path = _resolve_model_metadata_path(model_metadata)
        md = _read_json(mm_path)
        run_uri = md["run_uri"]
        run_id = md.get("run_id")

        cod, mse, coff = _read_score(Path(score_report) / "score.txt")

        model_version = _register_with_fallback(
            run_uri=run_uri,
            run_id=run_id,
            model_name=model_name,
            mse=mse,
            coff=coff,
            cod=cod,
            build_reference=build_reference,
        )

        # Set tags if MLflow model registry returned a model version
        if hasattr(model_version, "version"):
            client = mlflow.MlflowClient()
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="mse",
                value=mse,
            )
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="coff",
                value=coff,
            )
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="cod",
                value=cod,
            )
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="build_id",
                value=build_reference,
            )

        print(model_version)
    except Exception as ex:
        print(ex)
        raise


def _register_with_fallback(
    run_uri: str,
    run_id: Optional[str],
    model_name: str,
    mse: Optional[float],
    coff: Optional[float],
    cod: Optional[float],
    build_reference: Optional[str],
):
    """Try MLflow registration; on plugin mismatch, fall back to AML model asset."""
    try:
        mv = mlflow.register_model(run_uri, model_name)
        print("Registered via MLflow model registry.")
        return mv
    except TypeError as e:
        print(f"mlflow.register_model failed with TypeError: {e}")
        return _fallback_register_azureml_model(
            model_name=model_name,
            run_id=run_id,
            mse=mse,
            coff=coff,
            cod=cod,
            build_reference=build_reference,
        )


def _fallback_register_azureml_model(
    model_name: str,
    run_id: Optional[str],
    mse: Optional[float],
    coff: Optional[float],
    cod: Optional[float],
    build_reference: Optional[str],
):
    """Fallback: create an Azure ML Model asset from the run's artifact path."""
    try:
        from azure.identity import DefaultAzureCredential
        from azure.ai.ml import MLClient
        from azure.ai.ml.entities import Model
    except Exception as imp_err:
        print("azure.ai.ml not available; cannot fallback:", imp_err)
        return None

    if not run_id:
        print("No run_id present in metadata; cannot construct AML artifact URI.")
        return None

    sub, rg, ws = _get_aml_arm_env()
    if not (sub and rg and ws):
        print("Missing AML ARM environment variables; cannot construct AML artifact URI.")
        return None

    aml_artifact_uri = _build_aml_artifact_uri(sub, rg, ws, run_id)
    print(f"Attempting AML Model registration from: {aml_artifact_uri}")

    try:
        ml_client = MLClient(
            DefaultAzureCredential(),
            subscription_id=sub,
            resource_group_name=rg,
            workspace_name=ws,
        )
        tags = _build_tags(mse, coff, cod, build_reference)
        model = Model(name=model_name, path=aml_artifact_uri, type="mlflow_model", tags=tags)
        created = ml_client.models.create_or_update(model)
        print(f"Registered AML Model asset: {created.name}:{created.version}")

        class _Shim:
            def __init__(self, name: str, version: str):
                self.name = name
                self.version = version

        return _Shim(created.name, created.version)
    except Exception as e:
        print("Fallback AML model registration failed:", e)
        return None


def _print_versions() -> None:
    print("=" * 50)
    print("PACKAGE VERSIONS:")
    try:
        print(f"mlflow: {metadata.version('mlflow')}")
        try:
            print(f"azureml-mlflow: {metadata.version('azureml-mlflow')}")
        except metadata.PackageNotFoundError:
            print("azureml-mlflow: not-installed (curated plugin may be present)")
    except Exception as e:
        print(f"Could not get package versions: {e}")
    print("=" * 50)


def _resolve_model_metadata_path(model_metadata: str) -> Path:
    p = Path(model_metadata)
    if p.is_dir():
        candidate = p / "model_metadata.json"
        if candidate.exists():
            return candidate
        json_files = list(p.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No model metadata JSON found under folder: {p}")
        return json_files[0]
    return p


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _read_score(score_path: Path) -> Tuple[float, float, float]:
    data = _read_json(score_path)
    return data["cod"], data["mse"], data["coff"]


def _get_aml_arm_env() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    sub = os.environ.get("AZUREML_ARM_SUBSCRIPTION")
    rg = os.environ.get("AZUREML_ARM_RESOURCEGROUP") or os.environ.get("AZUREML_ARM_RESOURCE_GROUP")
    ws = os.environ.get("AZUREML_ARM_WORKSPACE_NAME") or os.environ.get("AZUREML_ARM_WORKSPACE")
    return sub, rg, ws


def _build_aml_artifact_uri(sub: str, rg: str, ws: str, run_id: str) -> str:
    return (
        f"azureml://subscriptions/{sub}/resourcegroups/{rg}/workspaces/{ws}/"
        f"datastores/workspaceartifactstore/paths/ExperimentRun/dcid.{run_id}/model"
    )


def _build_tags(
    mse: Optional[float],
    coff: Optional[float],
    cod: Optional[float],
    build_reference: Optional[str],
) -> dict:
    tags = {"build_id": build_reference or ""}
    if mse is not None:
        tags["mse"] = str(mse)
    if coff is not None:
        tags["coff"] = str(coff)
    if cod is not None:
        tags["cod"] = str(cod)
    return tags


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
