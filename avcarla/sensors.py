from typing import TYPE_CHECKING, Union


if TYPE_CHECKING:
    from .client import CarlaClient
    from avcarla.actor import CarlaActor

import itertools
import math
import weakref

import avstack.sensors
import carla
import numpy as np
from avstack import calibration
from avstack.config import ConfigDict
from avstack.geometry import transformations as tforms
from avstack.modules import BaseModule
from avstack.utils.decorators import apply_hooks
from carla import ColorConverter as cc

from avcarla.config import CARLA
from avcarla.geometry import CarlaReferenceFrame


SensorData = avstack.sensors.SensorData

# =============================================================
# SENSORS
# =============================================================


class CarlaSensor(BaseModule):
    id_iter = itertools.count()
    blueprint_name = ""
    name = ""

    def __init__(
        self,
        name: str,
        attributes: dict,
        mode: str,
        noise: dict,
        reference: Union[CarlaReferenceFrame, ConfigDict],
        do_spawn: bool,
        do_listen: bool,
        client: "CarlaClient",
        parent: "CarlaActor",
        source_ID: Union[int, None] = None,
        *args,
        **kwargs,
    ):
        super().__init__(name="CarlaSensor", *args, **kwargs)

        # -- attributes
        self.client = client
        self.parent = parent
        self.reference = (
            CARLA.build(reference, default_args={"reference": parent.reference})
            if not isinstance(reference, CarlaReferenceFrame)
            else reference
        )
        self.global_ID = next(CarlaSensor.id_iter)
        self.lla_origin = client.map.transform_to_geolocation(carla.Location(0, 0, 0))
        self.mode = mode
        self.noise = noise
        self.t0 = None
        self.frame0 = None
        self.initialized = False

        # -- identifiers
        _source_ID = next(self.next_id)
        if source_ID is not None:
            self.source_ID = source_ID
        else:
            if parent.ID is not None:
                self.source_ID = parent.ID
            else:
                self.source_ID = _source_ID
        self.source_name = name
        self.source_identifier = name + "-" + str(self.source_ID)

        # -- spawn from blueprint
        self.attributes = attributes
        self.bp = parent.world.get_blueprint_library().find(self.blueprint_name)
        for k, v in attributes.items():
            self.bp.set_attribute(k, str(v))
        self.do_listen = do_listen
        self.object = None
        if do_spawn:
            self.spawn()
            # time.sleep(0.2)  # to allow for initialization

    @property
    def _default_subfolder(self):
        return self.source_identifier

    @classmethod
    def reset_next_id(cls):
        cls.next_id = itertools.count()

    def initialize(self, t0, frame0):
        self.t0 = t0
        self.frame0 = frame0
        self.initialized = True

    def destroy(self):
        if self.object is not None:
            self.object.destroy()

    def _on_sensor_event(weak_self):
        """implemented in subclasses"""
        raise NotImplementedError

    def factory(self, do_spawn=True, do_listen=True):
        return self.__class__(
            name=self.source_name,
            mode=self.mode,
            noise=self.noise,
            reference=self.reference,
            do_spawn=do_spawn,
            do_listen=do_listen,
            client=self.client,
            parent=self.parent,
            **self.attributes,
        )

    def spawn(self):
        self.object = self.parent.world.spawn_actor(
            self.bp,
            self.reference.as_carla_transform(local=self.parent.actor is not None),
            attach_to=self.parent.actor if self.parent.actor else None,
            attachment_type=carla.AttachmentType.Rigid,
        )

        # callback
        if self.do_listen:
            weak_self = weakref.ref(self)
            self.object.listen(lambda event: self._on_sensor_event(weak_self, event))

    @apply_hooks
    def _make_data_class(self, timestamp, frame, data, **kwargs):
        if self.initialized:
            data_class = self.base_data(
                timestamp=timestamp - self.t0,
                frame=frame - self.frame0,
                data=data,
                calibration=self.calibration,
                source_ID=self.source_ID,
                source_name=self.source_name,
                **kwargs,
            )
            self.parent.sensor_data_manager.push(data_class)
            return data_class
        else:
            raise RuntimeError("sensor not initialized")


@CARLA.register_module()
class CarlaGnss(CarlaSensor):
    next_id = itertools.count()
    blueprint_name = "sensor.other.gnss"
    name = "gps"
    base_data = avstack.sensors.GpsData

    def __init__(
        self,
        parent: "CarlaActor",
        client: "CarlaClient",
        name: str = "gnss-0",
        mode: str = "lla",
        sensor_tick: float = 0.10,
        noise: dict = {
            "bias": {"east": 0, "north": 0, "up": 0},
            "sigma": {"east": 0, "north": 0, "up": 0},
        },
        reference: ConfigDict = {
            "type": "CarlaReferenceFrame",
            "location": [0.1, 0, 0.1],
        },
        do_spawn: bool = True,
        do_listen: bool = True,
        *args,
        **kwargs,
    ):
        attributes = {
            "sensor_tick": sensor_tick,
        }
        super().__init__(
            name=name,
            attributes=attributes,
            mode=mode,
            noise=noise,
            reference=reference,
            do_spawn=do_spawn,
            do_listen=do_listen,
            parent=parent,
            client=client,
            *args,
            **kwargs,
        )
        self.calibration = calibration.GpsCalibration(self.reference)

    @staticmethod
    def _on_sensor_event(weak_self, gnss):
        self = weak_self()
        lla = [np.pi / 180 * gnss.latitude, np.pi / 180 * gnss.longitude, gnss.altitude]
        ned = tforms.transform_point(lla, "lla", "ned", (np.array([0, 0, 0]), "lla"))
        sR = self.noise["sigma"]
        sb = self.noise["bias"]
        b = np.array([sb["east"], sb["north"], sR["up"]])
        r = np.array([sR["east"], sR["north"], sR["up"]])
        R = np.diag(r**2)
        v_enu = np.squeeze(np.array([ned[1], ned[0], -ned[2]]))
        v_enu = v_enu + b + r * np.random.randn(3)
        self._make_data_class(gnss.timestamp, gnss.frame, v_enu, levar=self.reference.x)


@CARLA.register_module()
class CarlaImu(CarlaSensor):
    next_id = itertools.count()
    blueprint_name = "sensor.other.imu"
    name = "imu"
    base_data = avstack.sensors.ImuData

    def __init__(
        self,
        parent: "CarlaActor",
        client: "CarlaClient",
        name: str = "imu-0",
        mode: str = "fw",
        sensor_tick: float = 0.10,
        noise: dict = {},
        reference: ConfigDict = {
            "type": "CarlaReferenceFrame",
            "location": [1.6, 0, 1.6],
        },
        do_spawn: bool = True,
        do_listen: bool = True,
        *args,
        **kwargs,
    ):
        attributes = {
            "sensor_tick": sensor_tick,
        }
        super().__init__(
            name=name,
            attributes=attributes,
            mode=mode,
            noise=noise,
            reference=reference,
            do_spawn=do_spawn,
            do_listen=do_listen,
            parent=parent,
            client=client,
            *args,
            **kwargs,
        )
        self.calibration = calibration.ImuCalibration(self.reference)

    @staticmethod
    def _on_sensor_event(weak_self, imu):
        self = weak_self()
        acc = imu.accelerometer
        gyr = imu.gyroscope
        agc = {
            "accelerometer": [acc.x, acc.y, acc.z],
            "gyroscope": [gyr.x, gyr.y, gyr.z],
            "compass": imu.compass,
        }
        self._make_data_class(imu.timestamp, imu.frame, agc)


class CarlaCamera(CarlaSensor):
    def __init__(self, attributes, *args, **kwargs):
        w = int(attributes["image_size_x"])
        h = int(attributes["image_size_y"])
        fov_h = float(attributes["fov"]) / 2.0  # half horizontal FOV
        f = (w / 2) / (np.tan(fov_h * math.pi / 180.0))
        self.P = np.array([[f, 0, w / 2.0, 0], [0, f, h / 2.0, 0], [0, 0, 1, 0]])
        self.imsize = (h, w)
        super().__init__(attributes=attributes, *args, **kwargs)


@CARLA.register_module()
class CarlaRgbCamera(CarlaCamera):
    next_id = itertools.count()
    blueprint_name = "sensor.camera.rgb"
    name = "camera"
    base_data = avstack.sensors.ImageData
    converter = cc.Raw

    def __init__(
        self,
        parent: "CarlaActor",
        client: "CarlaClient",
        name: str = "camera-0",
        mode: str = "standard",
        sensor_tick: float = 0.10,
        fov: float = 90,
        image_size_x: int = 800,
        image_size_y: int = 600,
        noise: dict = {},
        reference: ConfigDict = {
            "type": "CarlaReferenceFrame",
            "location": [1.0, 0, 1.6],
            "camera": True,
        },
        do_spawn: bool = True,
        do_listen: bool = True,
        *args,
        **kwargs,
    ):
        attributes = {
            "image_size_x": image_size_x,
            "image_size_y": image_size_y,
            "fov": fov,
            "sensor_tick": sensor_tick,
        }
        super().__init__(
            name=name,
            attributes=attributes,
            mode=mode,
            noise=noise,
            reference=reference,
            do_spawn=do_spawn,
            do_listen=do_listen,
            parent=parent,
            client=client,
            *args,
            **kwargs,
        )
        self.calibration = calibration.CameraCalibration(
            self.reference, self.P, self.imsize, channel_order="rgb"
        )

    @staticmethod
    def _on_sensor_event(weak_self, image):
        self = weak_self()
        image.convert(self.converter)
        # I guess we need to do the conversion here....
        np_img = np.reshape(
            np.array(image.raw_data, dtype=np.float32), (image.height, image.width, 4)
        )[
            :, :, :3
        ]  # BGRA
        self._make_data_class(image.timestamp, image.frame, np_img)


@CARLA.register_module()
class CarlaSemanticSegmentation(CarlaCamera):
    next_id = itertools.count()
    blueprint_name = "sensor.camera.semantic_segmentation"
    name = "semseg_camera"
    base_data = avstack.sensors.SemanticSegmentationImageData

    def __init__(
        self,
        parent: "CarlaActor",
        client: "CarlaClient",
        name: str = "semsegcamera-0",
        mode: str = "standard",
        sensor_tick: float = 0.10,
        fov: float = 90,
        image_size_x: int = 800,
        image_size_y: int = 600,
        noise: dict = {},
        reference: ConfigDict = {"type": "CarlaReferenceFrame", "camera": True},
        do_spawn: bool = True,
        do_listen: bool = True,
        *args,
        **kwargs,
    ):
        attributes = {
            "image_size_x": image_size_x,
            "image_size_y": image_size_y,
            "fov": fov,
            "sensor_tick": sensor_tick,
        }
        super().__init__(
            name=name,
            attributes=attributes,
            mode=mode,
            noise=noise,
            reference=reference,
            do_spawn=do_spawn,
            do_listen=do_listen,
            parent=parent,
            client=client,
            *args,
            **kwargs,
        )
        self.calibration = calibration.SemanticSegmentationCalibration(
            self.reference, self.P, self.imsize, channel_order="rgb"
        )

    @staticmethod
    def _on_sensor_event(weak_self, image):
        self = weak_self()
        self._make_data_class(image.timestamp, image.frame, image)


@CARLA.register_module()
class CarlaDepthCamera(CarlaCamera):
    next_id = itertools.count()
    blueprint_name = "sensor.camera.depth"
    name = "depth_camera"
    base_data = avstack.sensors.DepthImageData

    def __init__(
        self,
        parent: "CarlaActor",
        client: "CarlaClient",
        name: str = "depthcamera-0",
        mode: str = "standard",
        sensor_tick: float = 0.10,
        fov: float = 90,
        image_size_x: int = 800,
        image_size_y: int = 600,
        noise: dict = {},
        reference: ConfigDict = {"type": "CarlaReferenceFrame", "camera": True},
        do_spawn: bool = True,
        do_listen: bool = True,
        *args,
        **kwargs,
    ):
        attributes = {
            "image_size_x": image_size_x,
            "image_size_y": image_size_y,
            "fov": fov,
            "sensor_tick": sensor_tick,
        }
        super().__init__(
            name=name,
            attributes=attributes,
            mode=mode,
            noise=noise,
            reference=reference,
            do_spawn=do_spawn,
            do_listen=do_listen,
            parent=parent,
            client=client,
            *args,
            **kwargs,
        )
        self.calibration = calibration.DepthCameraCalibration(
            self.reference, self.P, self.imsize, channel_order="rgb"
        )

    @staticmethod
    def _on_sensor_event(weak_self, image):
        self = weak_self()
        self._make_data_class(image.timestamp, image.frame, image)


@CARLA.register_module()
class CarlaLidar(CarlaSensor):
    next_id = itertools.count()
    blueprint_name = "sensor.lidar.ray_cast"
    name = "lidar"
    base_data = avstack.sensors.LidarData

    def __init__(
        self,
        parent: "CarlaActor",
        client: "CarlaClient",
        name: str = "lidar-0",
        mode: str = "standard",
        sensor_tick: float = 0.10,
        channels: int = 32,
        rotation_frequency: float = 50,  # needs to match client rate
        range: float = 70.0,
        points_per_second: int = 1120000,  # 1750 * 32 * 20
        upper_fov: float = 2.4,
        lower_fov: float = -17.6,
        horizontal_fov: float = 360.0,
        noise: dict = {},
        reference: ConfigDict = {
            "type": "CarlaReferenceFrame",
            "location": [0, 0, 1.6],
        },
        do_spawn: bool = True,
        do_listen: bool = True,
        *args,
        **kwargs,
    ):
        attributes = {
            "channels": channels,
            "rotation_frequency": rotation_frequency,
            "range": range,
            "points_per_second": points_per_second,
            "upper_fov": upper_fov,
            "lower_fov": lower_fov,
            "horizontal_fov": horizontal_fov,
            "sensor_tick": sensor_tick,
        }
        super().__init__(
            name=name,
            attributes=attributes,
            mode=mode,
            noise=noise,
            reference=reference,
            do_spawn=do_spawn,
            do_listen=do_listen,
            parent=parent,
            client=client,
            *args,
            **kwargs,
        )
        self.calibration = calibration.LidarCalibration(self.reference)

    @staticmethod
    def _on_sensor_event(weak_self, pc):
        self = weak_self()
        self._make_data_class(pc.timestamp, pc.frame, pc, flipy=True)


@CARLA.register_module()
class CarlaRadar(CarlaSensor):
    next_id = itertools.count()
    blueprint_name = "sensor.other.radar"
    name = "radar"
    base_data = avstack.sensors.RadarDataRazelRRT

    def __init__(
        self,
        parent: "CarlaActor",
        client: "CarlaClient",
        name: str = "lidar-0",
        mode: str = "standard",
        sensor_tick: float = 0.10,
        range: float = 100.0,
        points_per_second: int = 1500,
        horizontal_fov: float = 30.0,
        vertical_fov: float = 30.0,
        noise: dict = {},
        reference: ConfigDict = {"type": "CarlaReferenceFrame"},
        do_spawn: bool = True,
        do_listen: bool = True,
        *args,
        **kwargs,
    ):
        attributes = {
            "range": range,
            "points_per_second": points_per_second,
            "horizontal_fov": horizontal_fov,
            "vertical_fov": vertical_fov,
            "sensor_tick": sensor_tick,
        }
        super().__init__(
            name=name,
            attributes=attributes,
            mode=mode,
            noise=noise,
            reference=reference,
            do_spawn=do_spawn,
            do_listen=do_listen,
            parent=parent,
            client=client,
            *args,
            **kwargs,
        )
        fov_h = float(attributes["horizontal_fov"]) * np.pi / 180
        fov_v = float(attributes["vertical_fov"]) * np.pi / 180
        self.calibration = calibration.RadarCalibration(self.reference, fov_h, fov_v)

    @staticmethod
    def _on_sensor_event(weak_self, razelrrt):
        self = weak_self()
        self._make_data_class(razelrrt.timestamp, razelrrt.frame, razelrrt)
