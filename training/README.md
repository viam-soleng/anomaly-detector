# Train Your Own Model

## Prerequisits

If you haven't done so, make sure you have installed all required dependencies:

```sh
pip install -r requirements.txt
```

To be able to connect to the Viam cloud backend, create a `.env` file in the project root `../` with the following variables and add your specific values:

```sh
API_KEY_ID=
API_KEY=
ORGANIZATION_ID=
```

## Train Your Model

This script requires tuning towards your data! I will add hints at a later point in time to make it easier to adapt it.
You should be easily able to figure it out yourself however in the meantime. Otherwise feel free to reach out.

The script works in the following way:

1. Connects to the Viam cloud backend using an MQL aggregation pipeline to download sensor data and stores it in a local csv file
2. It loads the CSV file, applies some type conversions (timestamp to date) and removes null values
3. Calculates additional information such as the rolling mean and also adds the previous value to each observation
4. Trains an `sklearn isolation forest model` and converts it to `onnx` so we can run it with `viam-server`
5. Takes the training dataset and runs inference on it -> prints the results to command line
