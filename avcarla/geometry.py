import math
from typing import Any, Tuple, Union

import carla
import numpy as np
from avstack.environment.objects import VehicleState
from avstack.geometry import (
    Acceleration,
    AngularVelocity,
    Attitude,
    GlobalOrigin3D,
    Pose,
    Position,
    ReferenceFrame,
    Velocity,
    bbox,
    q_cam_to_stan,
    q_stan_to_cam,
)
from avstack.geometry import transformations as tforms
from carla import Transform

from .bootstrap import CarlaClient
from .config import CARLA
from .utils import get_obj_type_from_actor


def parse_reference(
    reference: Union[int, str, ReferenceFrame], client: CarlaClient
) -> ReferenceFrame:
    if isinstance(reference, str):
        if reference == "GlobalOrigin3D":
            ref = GlobalOrigin3D
        else:
            raise NotImplementedError(reference)
    elif isinstance(reference, int):
        ref = client.spawn_points[reference]
    elif isinstance(reference, ReferenceFrame):
        ref = reference
    else:
        raise NotImplementedError(reference)

    return ref


@CARLA.register_module()
class CarlaReferenceFrame(ReferenceFrame):
    def __init__(
        self,
        reference: Tuple[str, Any],
        location: Tuple[float, float, float] = (0, 0, 0),
        rotation: Tuple[float, float, float] = (0, 0, 0),
        camera: bool = False,
    ):
        x = np.asarray(location)
        rpy = np.asarray(rotation) * np.pi / 180
        q = tforms.transform_orientation(rpy, "euler", "quat")
        self.camera = camera
        if self.camera:
            q = q_stan_to_cam * q
        super().__init__(x, q, reference=reference)

    @staticmethod
    def from_reference(reference: ReferenceFrame):
        loc = reference.x
        rot = tforms.transform_orientation(reference.q, "quat", "euler")
        return CarlaReferenceFrame(loc, rot)

    def as_carla_transform(self, local: bool):
        ref = self if local else self.integrate(start_at=GlobalOrigin3D)
        loc = numpy_vector_to_carla_location(ref.x)
        q = q_cam_to_stan * ref.q if self.camera else ref.q
        rot = quaternion_to_carla_rotation(q)
        return Transform(location=loc, rotation=rot)


def wrap_actor_to_vehicle_state(t, actor):
    """Location is the bottom of the box"""
    obj_type = get_obj_type_from_actor(actor)
    h = 2 * actor.bounding_box.extent.z
    w = 2 * actor.bounding_box.extent.y
    l = 2 * actor.bounding_box.extent.x
    VS = VehicleState(obj_type, actor.id)
    tf = actor.get_transform()
    v = actor.get_velocity()
    ac = actor.get_acceleration()
    pos = Position([tf.location.x, -tf.location.y, tf.location.z], GlobalOrigin3D)
    vel = Velocity([v.x, -v.y, v.z], GlobalOrigin3D)
    acc = Acceleration([ac.x, -ac.y, ac.z], GlobalOrigin3D)
    att = Attitude(
        tforms.transform_orientation(
            carla_rotation_to_RPY(tf.rotation), "euler", "quat"
        ),
        GlobalOrigin3D,
    )
    av = actor.get_angular_velocity()
    ang = AngularVelocity(np.quaternion(av.x, -av.y, av.z), GlobalOrigin3D)
    box = bbox.Box3D(pos, att, [h, w, l], where_is_t="bottom")
    VS.set(t, pos, box, vel, acc, att, ang)
    return VS


def carla_rotation_to_RPY(carla_rotation):
    """
    Convert a carla rotation to a roll, pitch, yaw tuple
    Considers the conversion from left-handed system (unreal) to right-handed
    system (ROS).
    Considers the conversion from degrees (carla) to radians (ROS).
    :param carla_rotation: the carla rotation
    :type carla_rotation: carla.Rotation
    :return: a tuple with 3 elements (roll, pitch, yaw)
    :rtype: tuple
    """
    roll = math.radians(carla_rotation.roll)
    pitch = -math.radians(carla_rotation.pitch)
    yaw = -math.radians(carla_rotation.yaw)

    return [roll, pitch, yaw]


def carla_rotation_to_quaternion(carla_rotation):
    rpy = carla_rotation_to_RPY(carla_rotation)
    return tforms.transform_orientation(rpy, "euler", "quat")


def carla_location_to_numpy_vector(carla_location):
    """
    Convert a carla location to a ROS vector3
    Considers the conversion from left-handed system (unreal) to right-handed
    system (ROS)
    :param carla_location: the carla location
    :type carla_location: carla.Location
    :return: a numpy.array with 3 elements
    :rtype: numpy.array
    """
    return np.array([carla_location.x, -carla_location.y, carla_location.z])


def numpy_vector_to_carla_location(x):
    return carla.Location(float(x[0]), float(-x[1]), float(x[2]))


def quaternion_to_carla_rotation(q):
    rpy = tforms.transform_orientation(q, "quat", "euler")
    return rpy_to_carla_rotation(180 / np.pi * rpy)
    # return carla.Rotation(
    #     pitch=-180 / np.pi * rpy[1],
    #     yaw=-180 / np.pi * rpy[2],
    #     roll=180 / np.pi * rpy[0],
    # )


def rpy_to_carla_rotation(rpy):
    return carla.Rotation(roll=float(rpy[0]), pitch=-float(rpy[1]), yaw=-float(rpy[2]))


def reference_to_carla_transform(reference: ReferenceFrame) -> carla.Transform:
    return carla.Transform(
        location=numpy_vector_to_carla_location(reference.x),
        rotation=quaternion_to_carla_rotation(reference.q),
    )


def carla_transform_to_pose(tf):
    q = tforms.transform_orientation(
        carla_rotation_to_RPY(tf.rotation), "euler", "quat"
    )
    pos = Position([tf.location.x, -tf.location.y, tf.location.z], GlobalOrigin3D)
    att = Attitude(q, GlobalOrigin3D)
    return Pose(pos, att)
