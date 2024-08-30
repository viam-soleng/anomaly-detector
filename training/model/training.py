############################################################################################################
# Training script for an Isolation Forest model using data from app.viam.com
############################################################################################################

import numpy as np
from onnx import ModelProto
import pandas as pd
from pathlib import Path
import argparse

from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import to_onnx
from sklearn.ensemble import IsolationForest
import asyncio
import os
import bson
from dotenv import load_dotenv
from viam.app.viam_client import ViamClient
from viam.rpc.dial import DialOptions

import logging
from google.cloud import logging_v2
from google.cloud import storage
import joblib

client = logging_v2.client.Client()

# set the format for the log
google_log_format = logging.Formatter(
    fmt="%(name)s | %(module)s | %(funcName)s | %(message)s",
    datefmt="%Y-%m-$dT%H:%M:%S",
)


handler = client.get_default_handler()
handler.setFormatter(google_log_format)

cloud_logger = logging.getLogger("vertex-ai-notebook-logger")
cloud_logger.setLevel("INFO")
cloud_logger.addHandler(handler)

log = logging.getLogger("vertex-ai-notebook-logger")
log.info("This is a log from a Vertex AI Notebook!")

load_dotenv()

############################################################################################################
# CLI Args Parser
############################################################################################################


def parse_args():
    """Returns dataset file, model output directory, and num_epochs if present. These must be parsed as command line
    arguments and then used as the model input and output, respectively. The number of epochs can be used to optionally override the default.
    """
    parser = argparse.ArgumentParser()
    # TODO: Add additional arguments as needed
    # parser.add_argument("--model-name", dest="model_name", type=str)
    parser.add_argument("--model_output_directory", dest="model_dir", type=str)
    parser.add_argument("--dataset_file", dest="data_json", type=str)
    # parser.add_argument("--org-id", dest="org_id", type=str)
    # parser.add_argument("--num_limit", dest="num_limit", type=int)
    args = parser.parse_args()
    return args.model_dir


############################################################################################################
# Get data from app.viam.com
############################################################################################################


async def connect() -> ViamClient:
    """Connect to the VIAM cloud api"""
    dial_options = DialOptions.with_api_key(
        api_key=os.environ.get("API_KEY"), api_key_id=os.environ.get("API_KEY_ID")
    )
    return await ViamClient.create_from_dial_options(dial_options)


async def download(limit: int = 10000):
    """Load the data and print the first 5 rows and a summary of the data."""
    print("Downloading data...")
    viam_client = await connect()
    data_client = viam_client.data_client

    # Query the data set using MongoDB Query Language (MQL)
    tabular_data = await data_client.tabular_data_by_mql(
        organization_id="96b696a0-51b9-403b-ae0d-63753923652f",  # os.getenv("ORGANIZATION_ID"),
        # The MQL aggregation pipeline extract and preprocess the data
        # TODO: Set this to suit your data! https://www.mongodb.com/docs/manual/core/aggregation-pipeline/
        mql_binary=[
            bson.dumps(
                {
                    "$match": {
                        "component_name": "fake-sensor",
                    }
                }
            ),
            bson.dumps(
                {
                    "$project": {
                        "timestamp": {"$dateToString": {"date": "$time_received"}},
                        "value": "$data.readings.a",
                    }
                }
            ),
            bson.dumps({"$limit": limit}),
        ],
    )
    df = pd.DataFrame(tabular_data)
    # TODO: timestamp conversion shouldn't be necessary once the Viam SDK date type in the MQL API is fixed
    # Also remove above in the MQL pipeline
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    viam_client.close()
    print(f"{len(df)} reading(s) downloaded")
    return df


############################################################################################################
# Feature Engineering
############################################################################################################


def featureEng(df: pd.DataFrame):
    print("Feature Engineering...")
    # A variety of resamples which I may or may not use
    # TODO: Push down to the backend -> Agg. pipeline
    df_sampled = df.set_index("timestamp").resample("5min").mean().reset_index()
    # df_sampled = df.set_index("timestamp").resample("h").mean().reset_index()
    # df_sampled = df.set_index("timestamp").resample("D").mean().reset_index()
    # df_sampled = df.set_index("timestamp").resample("W").mean().reset_index()

    # Calculate the rolling mean and lag (weekdays are not used in the model)
    for DataFrame in [df_sampled]:
        DataFrame["Weekday"] = pd.Categorical(
            DataFrame["timestamp"].dt.strftime("%A"),
            categories=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
        )
        DataFrame["Hour"] = DataFrame["timestamp"].dt.hour
        DataFrame["Day"] = DataFrame["timestamp"].dt.weekday
        DataFrame["Month"] = DataFrame["timestamp"].dt.month
        DataFrame["Year"] = DataFrame["timestamp"].dt.year
        DataFrame["Month_day"] = DataFrame["timestamp"].dt.day
        DataFrame["Lag"] = DataFrame["value"].shift(1)
        DataFrame["Rolling_Mean"] = DataFrame["value"].rolling(7, min_periods=1).mean()
        DataFrame = DataFrame.dropna()
    df_sampled.dropna(inplace=True)
    print("Feature Engineering Completed")
    return df_sampled


############################################################################################################
# Model Training
############################################################################################################


def fit_isolation_forest(
    model_data: pd.DataFrame,
) -> ModelProto:
    print("Fitting Isolation Forest...")
    model_data = (
        model_data[
            [
                "value",
                "Hour",
                "Day",
                "Month_day",
                "Month",
                "Rolling_Mean",
                "Lag",
                "timestamp",
            ]
        ]
        .set_index("timestamp")
        .dropna()
    )

    IF = IsolationForest(
        # Parameter tuning:
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest
        random_state=0,
        contamination="auto",
        n_estimators=100,
        max_samples="auto",
    )

    IF.fit(model_data)

    onx = to_onnx(
        IF,
        model_data.to_numpy().astype(np.float32),
        target_opset={"": 15, "ai.onnx.ml": 3},
        initial_types=[("observation", FloatTensorType([None, 7]))],
        final_types=[
            ("label", FloatTensorType([None, 1])),
            ("scores", FloatTensorType([None, 1])),
        ],
    )
    return onx


############################################################################################################
# Script exection
############################################################################################################

if __name__ == "__main__":

    # Parse the CLI arguments
    model_dir = parse_args()
    model_name = "isolation_forest.onnx"
    # Download the training data
    training_data = asyncio.run(download())
    # Feature Engineering
    df = featureEng(training_data)
    # Train the model
    print("Training Data:")
    print(df.head(10))
    onx = fit_isolation_forest(df)
    # Save the model
    # https://cloud.google.com/vertex-ai/docs/training/exporting-model-artifacts#joblib_1
    joblib.dump(onx, model_name)

    # filename = os.path.join(Path.cwd(), model_name)
    # with open(filename, "wb") as f:
    #    f.write(onx.SerializeToString())

    # Upload the model to GCS
    # Upload model artifact to Cloud Storage
    storage_path = os.path.join(model_dir, model_name)
    log.info(f"Uploading model to {storage_path}")
    blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
    blob.upload_from_filename(model_name)

    print(f"Isolation forest fitted and model created: ", model_name)
