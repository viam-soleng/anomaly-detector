############################################################################################################
# Training script for an Isolation Forest model using data from app.viam.com
############################################################################################################

import numpy as np
import pandas as pd
from pathlib import Path

from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

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
        api_key=os.getenv("API_KEY"), api_key_id=os.getenv("API_KEY_ID")
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
    df.to_csv(data_path / data_name, index=False)

    viam_client.close()
    print("Data downloaded.")


############################################################################################################
# Load from CSV
############################################################################################################


def loadCSV():
    print("Loading CSV...")
    df = pd.read_csv(data_path / data_name, parse_dates=["timestamp"])
    print("CSV Loaded")
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
) -> pd.DataFrame:
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

    # TODO: This model conversion requires a conflicting protobuf version
    # Potential solution: https://github.com/onnx/onnxmltools/blob/main/docs/examples/plot_convert_sklearn.py

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

    # path = Path.cwd().parent / "model" / model_name
    with open(model_path / model_name, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"Isolation Forest Fitted and model created: ", model_path)


############################################################################################################
# Model inference for validation purposes
############################################################################################################

# https://onnxruntime.ai/docs/api/python/tutorial.html


def inference(df: pd.DataFrame):
    print("Inference...")
    model_data = (
        df[
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
    path = Path.cwd().parent / "model" / model_name
    ort_sess = ort.InferenceSession(
        model_path / model_name,
        providers=ort.get_available_providers(),
    )

    nparray = model_data.to_numpy(dtype=np.float32)

    input_name = ort_sess.get_inputs()[0].name
    output_name = ort_sess.get_outputs()[0].name
    print("Input Name:", input_name)
    print("Output Name:", output_name)

    pred_onx = ort_sess.run(None, {input_name: nparray})

    inference = pd.DataFrame(
        np.column_stack([pred_onx[0], pred_onx[1]]), columns=["outlier", "score"]
    )
    print("Inference Completed")
    return inference


############################################################################################################
# Script exection
############################################################################################################

asyncio.run(download())

df = loadCSV()
df = featureEng(df)

# print(df.head(50))
# print(df.dtypes)
# print(len(df))

fit_isolation_forest(df)

inf = inference(df)
print(inf)
# An anomaly is identified by the model if the outlier column is equal to -1
print("Outliers: ", len(inf.query("outlier != 1")))
