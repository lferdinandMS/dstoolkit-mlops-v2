"""This module provides the functionality for initializing and running a machine learning model."""

import os
import logging
import pandas as pd
import pathlib
from typing import List
import yaml
from seq_model import NgramModel
from tokenizer import Tokenizer


def init():
    """
    Initialize the service instance on startup.

    You can write the logic here to perform init operations like caching the model in memory.
    """
    global model
    global tokenizer
    global model_cfg

    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model_registration", "model", "model_dict.pkl"
    )
    tokenizer_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"),
        "model_registration",
        "tokenizer",
        "tokenizer.json",
    )
    model_cfg_path = "model_config.yml"

    contents = os.listdir('.')
    for item in contents:
        print(item)

    path = pathlib.Path(__file__).with_name(model_cfg_path)
    cfg = yaml.safe_load(path.open())

    model_cfg = cfg['model']
    # deserialize the model
    model = NgramModel(
        max_prior_token_length=model_cfg["max_prior_token_length"],
        max_top_n=model_cfg["max_top_n"],
    )
    model.load(model_path)

    # deserialize the tokenizer
    tokenizer = Tokenizer()
    tokenizer.load(tokenizer_path)

    logging.info("Init complete")


def run(mini_batch: List[str]) -> pd.DataFrame:
    """
    Execute inferencing logic on a request.

    Processes each file in the mini batch, makes predictions for each line,
    and returns a DataFrame with input sequences and predicted next words.
    """
    results = []

    print("Request received")

    for raw_data in mini_batch:
        print(f"File name: {raw_data}")
        with open(raw_data, "r") as f:
            for line in f:
                data = line.strip().split(" ")
                tokenized_data = tuple(tokenizer.enc(words=data))
                result = model.predict(tokenized_data, top_n=3)
                preds = tokenizer.dec(result)
                print("Input data:", line.strip())
                print("Possible choices for next word:", preds)
                
                # Store actual prediction results
                results.append({
                    "file": raw_data,
                    "input_sequence": line.strip(),
                    "prediction_1": preds[0] if len(preds) > 0 else None,
                    "prediction_2": preds[1] if len(preds) > 1 else None,
                    "prediction_3": preds[2] if len(preds) > 2 else None
                })

        print(f"File name: {raw_data} has been processed")

    return pd.DataFrame(results)
