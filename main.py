#!/usr/bin/env python3

import asyncio
from datetime import datetime
from typing import ClassVar, Mapping, Optional, Sequence, cast
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
import pandas as pd

LOGGER = getLogger(__name__)


class AnomalySensor(Sensor):
    """Anomaly Sensor component that detects anomalies in the sensor readings."""

    MODEL: ClassVar[Model] = Model(ModelFamily("viam-soleng", "sensor"), "anomaly")

    def __init__(self, name: str):
        super().__init__(name)

    source_sensor: Optional[Sensor] = None
    anomaly_model: Optional[MLModel] = None

    # Observations buffer
    # https://docs.python.org/3/library/collections.html#deque-recipes
    # https://www.geeksforgeeks.org/deque-in-python/
    queue = deque(maxlen=10)

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Sequence[str]:
        if "sensor" in config.attributes.fields:
            if not config.attributes.fields["sensor"].HasField("string_value"):
                raise Exception("A sensor must be provided")
            sensor_name = config.attributes.fields["sensor"].string_value
        if "model" in config.attributes.fields:
            if not config.attributes.fields["model"].HasField("string_value"):
                raise Exception("A model must be provided")
            model_name = config.attributes.fields["model"].string_value
        if "queue" in config.attributes.fields:
            if not config.attributes.fields["queue"].HasField("number_value"):
                raise Exception("A queue length must be provided")
            queue_length = int(config.attributes.fields["queue"].number_value)
            if (queue_length < 1) and not (isinstance(queue_length, int)):
                raise Exception("The queue length must be greater than 0")
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
        # Get the queue length and intialize the queue
        queue_len = int(attributes_dict.get("queue"))
        assert isinstance(queue_len, int)
        self.queue = deque(maxlen=queue_len)
        # Get the source sensor
        sensor_name = attributes_dict.get("sensor")
        assert isinstance(sensor_name, str)
        sensor = dependencies[Sensor.get_resource_name(sensor_name)]
        self.source_sensor = cast(Sensor, sensor)
        # Get the anomaly model
        model_name = attributes_dict.get("model")
        assert isinstance(model_name, str)
        model = dependencies[MLModel.get_resource_name(model_name)]
        self.anomaly_model = cast(MLModel, model)

    async def get_readings(self, **kwargs):

        # Get the readings from the sensor
        reading = await self.source_sensor.get_readings()
        dt_received = datetime.now()
        self.queue.appendleft(reading["a"])
        # LOGGER.info(f"Queue: {len(self.queue)}")
        # LOGGER.info(f"Reading: {reading["a"]}")

        # Feature engineering

        # Calcualte the rolling mean and lag
        # if not enough readings available, return the reading and skip inference
        if len(self.queue) > 1:
            rolling_mean = np.mean(self.queue)
            lag = self.queue[1]
        else:
            return reading

        observation = pd.DataFrame(
            [
                (
                    reading["a"],  # sensor value
                    dt_received.hour,  # Hour
                    dt_received.weekday(),  # day
                    dt_received.day,  # month_day
                    # now.year,  # year
                    dt_received.month,  # month
                    rolling_mean,  # rolling_mean
                    lag,  # lag - previous reading
                )
            ],
            columns=[
                "value",
                "Hour",
                "Day",
                "Month_day",
                # "Year",
                "Month",
                "Rolling_Mean",
                "Lag",
            ],
        )
        nparray = observation.to_numpy(dtype=np.float32)

        # Run the anomaly detection
        inference = await self.anomaly_model.infer({"observation": nparray})
        reading["inference"] = {
            "scores": inference["scores"][0][0],
            "anomaly": inference["label"][0][0],
        }
        reading["queue"] = len(self.queue)
        return reading


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
