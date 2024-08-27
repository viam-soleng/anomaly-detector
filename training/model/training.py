############################################################################################################
# Training script for an Isolation Forest model using data from app.viam.com
############################################################################################################

import numpy as np
from onnx import ModelProto
import pandas as pd
from pathlib import Path

from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import to_onnx
from sklearn.ensemble import IsolationForest
import asyncio
import os
import bson
from dotenv import load_dotenv
from viam.app.viam_client import ViamClient
from viam.rpc.dial import DialOptions

load_dotenv()


# Set the path and name for the data and model files
data_path = Path.cwd()
data_name = "viam_temp.csv"
model_path = Path.cwd().parent / "model"
model_name = "anomaly.onnx"

############################################################################################################
# Get data from app.viam.com
############################################################################################################


async def connect() -> ViamClient:
    """Connect to the VIAM cloud api"""
    dial_options = DialOptions.with_api_key(
        api_key=os.environ.get("API_KEY"), api_key_id=os.environ.get("API_KEY_ID")
    )
    return await ViamClient.create_from_dial_options(dial_options)


async def download():
    """Load the data and print the first 5 rows and a summary of the data."""
    print("Downloading data...")
    viam_client = await connect()
    data_client = viam_client.data_client

    # Query the data set using MongoDB Query Language (MQL)
    tabular_data = await data_client.tabular_data_by_mql(
        organization_id=os.getenv("ORGANIZATION_ID"),
        # The MQL aggregation pipeline extract and preprocess the data
        # TODO: Set this to suit your data! https://www.mongodb.com/docs/manual/core/aggregation-pipeline/
        mql_binary=[
            bson.dumps(
                {
                    "$match": {
                        "component_name": "tmp36",
                        "data.readings.temp": {"$exists": True},
                    }
                }
            ),
            bson.dumps(
                {
                    "$project": {
                        "timestamp": {"$dateToString": {"date": "$time_received"}},
                        "value": "$data.readings.temp",
                    }
                }
            ),
            bson.dumps({"$limit": 10000}),
        ],
    )
    df = pd.DataFrame(tabular_data)
    viam_client.close()
    print("Data downloaded.")
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

training_data = asyncio.run(download())

df = featureEng(training_data)

# Train the model
onx = fit_isolation_forest(df)

# Save the model
with open(model_path / model_name, "wb") as f:
    f.write(onx.SerializeToString())

print(f"Isolation Forest Fitted and model created: ", model_path)
