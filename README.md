# Tabular Data `Anomaly-Detector` Module

This [module](https://docs.viam.com/registry/#modular-resources) implements the [`rdk:component:base` API](/components/base/#api) in an `anomaly-detector` model.
This repo can be used as a starter kit to implement your personal real time anomaly detection using a machine learning model trained with your data.
This is an advanced topic and Viam basics such as configuring machines, components and services are expected to be known! If you are not familiar with these topics yet, I recommend to start with this [beginner tutorial](https://docs.viam.com/how-tos/configure/) first and then come back.

## Requirements

If you haven't done so, I recommend to familiarize with the topic of creating your own Viam resources. You can find the related documentation here: [Create Your Own Modules](https://docs.viam.com/registry/#create-your-own-modules).

To create your own development setup follow these steps (tested on OSX):

1. Install `viam-server` on your machine [Install Guide](https://docs.viam.com/installation/)
2. Clone this repository to your machine
3. In app.viam.com navigate to the JSON configuration of your machine
4. Add this configuration into the `modules` array:

```json
{
  "type": "registry",
  "name": "viam-labs_onnx-cpu",
  "module_id": "viam-labs:onnx-cpu",
  "version": "0.1.4"
},
{
  "type": "local",
  "name": "local-module",
  "executable_path": "/<YOUR PATH>/anomaly-detector/run.sh"
}
```

5. Add this configuration into the `components` array:

```json
{
  "name": "fake-sensor",
  "namespace": "rdk",
  "type": "sensor",
  "model": "fake",
  "attributes": {}
},
{
  "name": "sensor",
  "namespace": "rdk",
  "type": "sensor",
  "model": "viam-soleng:sensor:anomaly",
  "attributes": {
    {
      "sensor": "fake-sensor",
      "model": "mlmodel",
      "features": [
        {
          "key": "a"
        },
        {
          "key": "Hour"
        },
        {
          "key": "Week_Day"
        },
        {
          "key": "Month_Day"
        },
        {
          "key": "Month"
        },
        {
          "key": "b",
          "rolling_mean": 7
        },
        {
          "key": "c",
          "lag": 1
        }
      ]
    }
  }
}
```

7. Add this configuration to the `services` array:

```json
{
  "name": "mlmodel",
  "namespace": "rdk",
  "type": "mlmodel",
  "model": "viam-labs:mlmodel:onnx-cpu",
  "attributes": {
    "model_path": "/<YOUR PATH>/anomaly-detector/model/sample.onnx",
    "label_path": ""
  }
}
```

6. Save the configuration and start the machine

## Configure your `anomaly-detector` sensor

### Attributes

The following attributes are available for `<INSERT MODEL TRIPLET>` <INSERT API NAME>s:

| Name                   | Type     | Required?    | Description                                                                                                               |
| ---------------------- | -------- | ------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `sensor`               | string   | **Required** | The sensor you want to apply the ml model to                                                                              |
| `model`                | string   | **Required** | The ml model to be used                                                                                                   |
| `features`             | [struct] | **Required** | List of features                                                                                                          |
| `feature.key`          | string   | **Required** | The key to the reading of your input/source sensor or one of `"Year", "Month", "Month_Day", "Week_Day", "Hour", "Minute"` |
| `feature.rolling_mean` | int      | **Optional** | Number of readings to calculate the rolling mean of this feature                                                          |
| `feature.lag`          | int      | **Optional** | Adds the reading `current-n`.                                                                                             |

> [!IMPORTANT]
> The order of features is consistent with the order used for passing them into the ML model.
> Make sure this maps to the order you have trained the model with!

### Next steps

To train your own model with your data we have created a sample training script which should be easy to tune towards your particular data set: [Model Training](training/README.md)

Let us know if you are interested in this topic, have questions, struggle with setting it up, enhancing it etc.. We are happy to help!

## Troubleshooting

N/A
