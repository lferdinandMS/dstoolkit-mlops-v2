"""Register machine learning models with MLflow, with AML fallback."""
import argparse
import json
import os
from pathlib import Path

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
        # Print package versions for debugging
        print("=" * 50)
        print("PACKAGE VERSIONS:")
        try:
            mlflow_ver = metadata.version("mlflow")
            print(f"mlflow: {mlflow_ver}")
            try:
                azml_mlflow_ver = metadata.version("azureml-mlflow")
                print(f"azureml-mlflow: {azml_mlflow_ver}")
            except metadata.PackageNotFoundError:
                print("azureml-mlflow: not-installed (curated plugin may be present)")
        except Exception as e:
            print(f"Could not get package versions: {e}")
        print("=" * 50)

        # Support both file path and folder input for model_metadata
        mm_path = Path(model_metadata)
        if mm_path.is_dir():
            candidate = mm_path / "model_metadata.json"
            if not candidate.exists():
                json_files = list(mm_path.glob("*.json"))
                if not json_files:
                    raise FileNotFoundError(
                        f"No model metadata JSON found under folder: {mm_path}"
                    )
                candidate = json_files[0]
            mm_path = candidate

        with open(mm_path) as run_file:
            md = json.load(run_file)
        run_uri = md["run_uri"]
        run_id = md.get("run_id")

        with open(Path(score_report) / "score.txt") as score_file:
            score_data = json.load(score_file)
        cod = score_data["cod"]
        mse = score_data["mse"]
        coff = score_data["coff"]

        # Attempt MLflow registration first
        client = mlflow.MlflowClient()
        try:
            # Avoid pre-listing artifacts to reduce plugin surface area
            model_version = mlflow.register_model(run_uri, model_name)
            print("Registered via MLflow model registry.")
        except TypeError as e:
            # Common on curated AML images when plugin/base-class signatures mismatch
            print(f"mlflow.register_model failed with TypeError: {e}")
            model_version = _fallback_register_azureml_model(
                model_name=model_name,
                run_id=run_id,
                mse=mse,
                coff=coff,
                cod=cod,
                build_reference=build_reference,
            )
            if model_version is None:
                raise

        # If MLflow model registry path succeeded, set model version tags
        if hasattr(model_version, "version"):
            client.set_model_version_tag(
                name=model_name, version=model_version.version, key="mse", value=mse
            )
            client.set_model_version_tag(
                name=model_name, version=model_version.version, key="coff", value=coff
            )
            client.set_model_version_tag(
                name=model_name, version=model_version.version, key="cod", value=cod
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


def _fallback_register_azureml_model(
    model_name: str,
    run_id: str | None,
    mse: float | None,
    coff: float | None,
    cod: float | None,
    build_reference: str | None,
):
    """Fallback: create an Azure ML Model asset from the run's artifact path.

    This avoids MLflow artifact repository constructor mismatches by using the
    Azure ML v2 SDK to register the model directly from the workspace artifact store.

    Returns a lightweight object with name/version on success, else None.
    """
    try:
        from azure.identity import DefaultAzureCredential
        from azure.ai.ml import MLClient
        from azure.ai.ml.entities import Model
    except Exception as imp_err:
        print(
            "azure.ai.ml not available in environment; cannot fallback to AML model registration:",
            imp_err,
        )
        return None

    try:
        if not run_id:
            print("No run_id present in metadata; cannot construct AML artifact URI.")
            return None

        sub = os.environ.get("AZUREML_ARM_SUBSCRIPTION")
        rg = os.environ.get("AZUREML_ARM_RESOURCEGROUP") or os.environ.get("AZUREML_ARM_RESOURCE_GROUP")
        ws = os.environ.get("AZUREML_ARM_WORKSPACE_NAME") or os.environ.get("AZUREML_ARM_WORKSPACE")

        if not (sub and rg and ws):
            print(
                "Missing AML ARM environment variables; cannot construct AML artifact URI."
            )
            return None

        # Path to the standard MLflow model artifact within an AML run
        aml_artifact_uri = (
            f"azureml://subscriptions/{sub}/resourcegroups/{rg}/workspaces/{ws}/"
            f"datastores/workspaceartifactstore/paths/ExperimentRun/dcid.{run_id}/model"
        )
        print(f"Attempting AML Model registration from: {aml_artifact_uri}")

        ml_client = MLClient(
            DefaultAzureCredential(), subscription_id=sub, resource_group_name=rg, workspace_name=ws
        )

        tags = {
            "build_id": build_reference or "",
        }
        if mse is not None:
            tags["mse"] = str(mse)
        if coff is not None:
            tags["coff"] = str(coff)
        if cod is not None:
            tags["cod"] = str(cod)

        model = Model(name=model_name, path=aml_artifact_uri, type="mlflow_model", tags=tags)
        created = ml_client.models.create_or_update(model)
        print(f"Registered AML Model asset: {created.name}:{created.version}")

        # Return a lightweight shim with .version to align with MLflow path above
        class _Shim:
            def __init__(self, name, version):
                self.name = name
                self.version = version

        return _Shim(created.name, created.version)
    except Exception as e:
        print("Fallback AML model registration failed:", e)
        return None


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
