#!/usr/bin/env python3

import asyncio
from datetime import datetime
from typing import ClassVar, List, Mapping, Optional, Sequence, cast
from typing_extensions import Self
from viam.logging import getLogger
from viam.components.sensor import Sensor
from viam.services.mlmodel import MLModel
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.module.module import Module
from collections import deque
from viam.proto.app.robot import ComponentConfig
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.utils import struct_to_dict
from viam.resource.types import Model, ModelFamily
import numpy as np

LOGGER = getLogger(__name__)


class AnomalySensor(Sensor):
    """Anomaly Sensor component that detects anomalies in sensor readings."""

    MODEL: ClassVar[Model] = Model(ModelFamily("viam-soleng", "sensor"), "anomaly")

    def __init__(self, name: str):
        super().__init__(name)

    source_sensor: Optional[Sensor] = None
    features: List[dict] = []  # List of features {key:str, lag:int, rolling_mean:int}
    inf_features: List[str] = []  # List of features to be used in the inference

    anomaly_model: Optional[MLModel] = None

    # Observations buffer
    queue = Optional[deque]

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Sequence[str]:
        """Validate the configuration of the Anomaly Sensor component"""
        # Validate sensor configuration
        if not config.attributes.fields["sensor"].HasField("string_value"):
            raise Exception("A sensor must be provided as string")
        sensor_name = config.attributes.fields["sensor"].string_value

        # Validate ML model configuration
        if not config.attributes.fields["model"].HasField("string_value"):
            raise Exception("'model' must be provided as string")
        model_name = config.attributes.fields["model"].string_value

        # Validate each feature's configuration
        if not config.attributes.fields["features"].HasField("list_value"):
            raise Exception("'features' must be provided as a list of dictionaries")
        for feature in config.attributes.fields["features"].list_value.values:
            if not feature.HasField("struct_value"):
                raise Exception("'features' must be provided as a list of dictionaries")
            if not feature.struct_value.fields["key"].HasField("string_value"):
                raise Exception("A 'key' must be provided as string")
            if feature.struct_value.fields["lag"].HasField(
                "number_value"
            ) or feature.struct_value.fields["rolling_mean"].HasField("number_value"):
                if feature.struct_value.fields["lag"].HasField(
                    "number_value"
                ) and feature.struct_value.fields["rolling_mean"].HasField(
                    "number_value"
                ):
                    raise Exception("Can only have one of 'lag' or 'rolling_mean'")
                if not feature.struct_value.fields["lag"].number_value.is_integer():
                    raise Exception("'lag' must be provided as integer")
                if not feature.struct_value.fields[
                    "rolling_mean"
                ].number_value.is_integer():
                    raise Exception("'rolling_mean' must be provided as integer")

        return [sensor_name, model_name]

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        sensor = cls(config.name)
        # Call reconfigure to initialize the resource
        sensor.reconfigure(config, dependencies)
        return sensor

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        attributes_dict = struct_to_dict(config.attributes)

        # Get the source sensor
        sensor_name = attributes_dict.get("sensor")
        assert isinstance(sensor_name, str)
        sensor = dependencies[Sensor.get_resource_name(sensor_name)]
        self.source_sensor = cast(Sensor, sensor)

        # Calculate the needed queue length and extract the features
        queue_len = 0
        self.features = []
        for feature in attributes_dict.get("features"):
            assert isinstance(feature, dict)
            assert "key" in feature
            if "lag" in feature and feature["lag"] > queue_len:
                queue_len = int(feature["lag"])
            if "rolling_mean" in feature and feature["rolling_mean"] > queue_len:
                queue_len = int(feature["rolling_mean"])
            self.features.append(feature)
        self.queue = deque(maxlen=queue_len)

        # Get the anomaly model
        model_name = attributes_dict.get("model")
        assert isinstance(model_name, str)
        model = dependencies[MLModel.get_resource_name(model_name)]
        self.anomaly_model = cast(MLModel, model)

        LOGGER.info(
            f"Anomaly Sensor configured with: 'queue length': {queue_len}, 'features': {self.features}, 'model': {model_name}"
        )

    async def get_readings(self, **kwargs):
        """Get the sensor readings and run the anomaly detection"""

        # Get the readings from the source sensor
        reading = await self.source_sensor.get_readings()
        dt_received = datetime.now()
        self.queue.appendleft(reading)

        # if not enough readings available, return the reading and skip inference
        if len(self.queue) < self.queue.maxlen:
            reading["inference"] = {
                "queue": f"{len(self.queue)}/{self.queue.maxlen}",
            }
            return reading

        # Feature engineering
        feature_values = []
        for feature in self.features:
            match feature["key"]:
                case "Year":
                    feature_values.append(dt_received.year)
                case "Month":
                    feature_values.append(dt_received.month)
                case "Month_Day":
                    feature_values.append(dt_received.day)
                case "Week_Day":
                    feature_values.append(dt_received.weekday())
                case "Hour":
                    feature_values.append(dt_received.hour)
                case "Minute":
                    feature_values.append(dt_received.minute)
                case _:  # default case
                    if "rolling_mean" in feature:
                        if feature["rolling_mean"] > 0:
                            feature_values.append(
                                self.rolling_mean(
                                    feature["key"], int(feature["rolling_mean"])
                                ),
                            )
                    elif "lag" in feature:
                        if feature["lag"] > 0:
                            feature_values.append(
                                self.lag(feature["key"], int(feature["lag"])),
                            )
                    else:
                        feature_values.append(reading[feature["key"]])
        nparray = np.array([feature_values], dtype=np.float32)
        LOGGER.debug(f"Observation: {nparray}")

        # Run the anomaly detection
        inference = await self.anomaly_model.infer({"observation": nparray})
        reading["inference"] = {
            "scores": inference["scores"][0][0],
            "anomaly": inference["label"][0][0],
        }
        return reading

    def lag(self, key: str, lag: int) -> float:
        """Calculate the lag of a sensor reading"""
        LOGGER.debug(f"Lag: {self.queue[lag - 1][key]}")
        return self.queue[lag - 1][key]

    def rolling_mean(self, key: str, n: int) -> float:
        """Calculate the rolling mean of a sensor reading"""
        values = [d[key] for d in self.queue if key in d]
        rm = np.mean(values[:n])
        LOGGER.debug(f"Rolling Mean: {rm}")
        return rm


async def main():
    """This function creates and starts a new module, after adding all desired resource models.
    Resource creators must be registered to the resource registry before the module adds the resource model.
    """
    Registry.register_resource_creator(
        Sensor.SUBTYPE,
        AnomalySensor.MODEL,
        ResourceCreatorRegistration(AnomalySensor.new, AnomalySensor.validate_config),
    )

    module = Module.from_args()
    module.add_model_from_registry(Sensor.SUBTYPE, AnomalySensor.MODEL)
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())
