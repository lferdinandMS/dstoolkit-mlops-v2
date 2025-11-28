"""This module is designed to register machine learning models with MLflow."""
import mlflow
import argparse
import json
from pathlib import Path

import pkg_resources
print("mlflow:", mlflow.__version__)
try:
    print("azureml-mlflow:", pkg_resources.get_distribution("azureml-mlflow").version)
except Exception as e:
    print("azureml-mlflow not found or from image:", e)

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
            import pkg_resources
            print(f"mlflow: {pkg_resources.get_distribution('mlflow').version}")
            print(f"azureml-mlflow: {pkg_resources.get_distribution('azureml-mlflow').version}")
        except Exception as e:
            print(f"Could not get package versions: {e}")
        print("=" * 50)

        run_file = open(model_metadata)
        md = json.load(run_file)
        run_uri = md["run_uri"]

        score_file = open(Path(score_report) / "score.txt")
        score_data = json.load(score_file)
        cod = score_data["cod"]
        mse = score_data["mse"]
        coff = score_data["coff"]

        # Basic verification that artifacts exist at the expected path
        client = mlflow.MlflowClient()
        if run_uri.startswith("runs:/"):
            parts = run_uri.split("/")
            run_id = parts[1]
            artifact_path = "/".join(parts[2:]) if len(parts) > 2 else ""
            arts = client.list_artifacts(run_id, path=artifact_path)
            print(f"Artifacts under '{artifact_path}':", [a.path for a in arts])
            if not arts:
                raise RuntimeError(
                    f"No artifacts found at path '{artifact_path}' for run {run_id}."
                )

        # Register the model
        model_version = mlflow.register_model(run_uri, model_name)

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
    finally:
        run_file.close()


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
