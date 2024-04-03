import random
from typing import TYPE_CHECKING, List, Union


if TYPE_CHECKING:
    from .client import CarlaClient

import itertools

# import carla
import numpy as np
from avstack.config import PIPELINE, ConfigDict
from avstack.datastructs import DataContainer, DataManager
from avstack.geometry import Attitude, GlobalOrigin3D, Pose, Position, ReferenceFrame
from avstack.geometry import transformations as tforms
from avstack.modules import BaseModule
from avstack.utils.decorators import apply_hooks

from avcarla.geometry import (
    carla_location_to_numpy_vector,
    carla_rotation_to_RPY,
    wrap_mobile_actor_to_object_state,
    wrap_static_actor_to_object_state,
)

from .config import CARLA
from .geometry import CarlaReferenceFrame, carla_transform_to_pose


def parse_vehicle_blueprint(vehicle: Union[str, int], vehicle_bps):
    if isinstance(vehicle, str):
        if vehicle in ["random", "randint"]:
            bp = random.choice(vehicle_bps)
        else:
            bp = vehicle_bps.filter(vehicle)[0]
    elif isinstance(vehicle, int):
        bp = vehicle_bps[vehicle]
    else:
        raise NotImplementedError
    return bp


def parse_spawn(
    spawn: Union[ConfigDict, str, int],
    spawn_points: List,
    reference_to_spawn: ConfigDict,
):
    # parse the original spawn point
    if isinstance(spawn, str):
        if spawn in ["random", "randint"]:
            tf = random.choice(spawn_points)
        else:
            raise NotImplementedError(spawn)
    elif isinstance(spawn, int):
        tf = spawn_points[spawn]
    elif isinstance(spawn, ConfigDict):
        tf = CARLA.build(spawn).as_carla_transform()
    else:
        raise NotImplementedError(type(spawn))

    # apply the additional transformation
    tf_as_pose = carla_transform_to_pose(tf)
    tf_as_reference = ReferenceFrame(
        x=tf_as_pose.position.x, q=tf_as_pose.attitude.q, reference=GlobalOrigin3D
    )
    spawn_to_object = CARLA.build(
        reference_to_spawn, default_args={"reference": tf_as_reference}
    )

    # bring back to carla point
    tf_spawn = spawn_to_object.as_carla_transform(local=False)

    return tf_spawn


def parse_destination(
    destination: Union[ConfigDict, str, int],
    spawn_points: List,
    reference: Union[CarlaReferenceFrame, None] = None,
):
    if destination is None:
        dest = None
    elif isinstance(destination, str):
        if destination in ["random", "randint"]:
            dest = random.choice(spawn_points)
        else:
            raise NotImplementedError(destination)
    elif isinstance(destination, int):
        dest = spawn_points[destination].location
        dest = carla_location_to_numpy_vector(dest)
    elif isinstance(destination, np.ndarray):
        dest = destination
    elif isinstance(destination, dict):
        dkeys = sorted(list(destination.keys()))
        if dkeys == ["forward", "right", "up"]:
            delta = np.array([destination[k] for k in dkeys])
            dest = reference.location + reference.rotation @ delta
        elif dkeys == ["x", "y", "z"]:
            raise
        else:
            raise NotImplementedError(dkeys)
    else:
        raise NotImplementedError("Cannot understand destination type")
    return dest


def try_spawn_actor(world, bp, tf):
    n_att = 10
    d_inc = 3
    i = 0
    while i < n_att:
        actor = world.try_spawn_actor(bp, tf)
        if actor is None:
            fv = tf.rotation.get_forward_vector()
            tf.location.x += d_inc * fv.x
            tf.location.y += d_inc * fv.y
            # tf.location.z += d_inc * fv.z
            i += 1
        else:
            break
    else:
        raise RuntimeError(f"Could not spawn actor after {i} attempts")
    return actor


@CARLA.register_module()
class CarlaObjectManager(BaseModule):
    def __init__(
        self, objects, subname: str, client: "CarlaClient", *args, **kwargs
    ) -> None:
        super().__init__(name="CarlaObjectManager", *args, **kwargs)
        self.objects = [
            CARLA.build(obj, default_args={"client": client}) for obj in objects
        ]
        self.subname = subname

    def destroy(self):
        for obj in self.objects:
            obj.destroy()

    def initialize(self, t0: float, frame0: int):
        for obj in self.objects:
            obj.initialize(t0=t0, frame0=frame0)
        self.t0 = t0
        self.frame0 = frame0

    @apply_hooks
    def on_world_tick(self, world_snapshot):
        self.timestamp = world_snapshot.timestamp.elapsed_seconds - self.t0
        self.frame = world_snapshot.frame - self.frame0
        for obj in self.objects:
            debug = obj.tick(timestamp=self.timestamp, frame=self.frame)
        return DataContainer(
            source_identifier=self.subname,
            frame=self.frame,
            timestamp=self.timestamp,
            data=self.objects,
        )


class CarlaObject(BaseModule):
    def __init__(
        self,
        name,
        spawn,
        client: "CarlaClient",
        reference_to_spawn: "ConfigDict",
        *args,
        **kwargs,
    ):
        super().__init__(name=name, *args, **kwargs)
        self.world = client.world
        self.map = self.world.get_map()
        self.spawn = spawn
        self.reference_to_spawn = reference_to_spawn

    def destroy(self):
        if self.actor is not None:
            try:
                self.actor.destroy()
            except RuntimeError as e:
                pass  # usually because already destroyed
        try:
            for s_name, sensor in self.sensors.items():
                try:
                    sensor.destroy()
                except (KeyboardInterrupt, Exception) as e:
                    print(f"Could not destroy sensor {s_name}...continuing")
        except AttributeError:
            pass

    def encode(self):
        return self.get_object_state().encode()

    def get_object_state(self):
        try:
            if self.mobile:
                obj = wrap_mobile_actor_to_object_state(self, self.timestamp)
            else:
                obj = wrap_static_actor_to_object_state(self, self.timestamp)
        except RuntimeError:
            obj = None
        return obj


@CARLA.register_module()
class CarlaNpc(CarlaObject):
    mobile = True
    id_iter_global = itertools.count()

    def __init__(
        self,
        spawn,
        npc_type: str,
        client: "CarlaClient",
        reference_to_spawn: ConfigDict = {"type": "CarlaReferenceFrame"},
        *args,
        **kwargs,
    ):
        self.ID_npc_global = next(self.id_iter_global)
        name = f"npc-{self.ID_npc_global}"
        bp = np.random.choice(client.world.get_blueprint_library().filter(npc_type))
        tf = parse_spawn(
            spawn=spawn,
            spawn_points=client.spawn_points,
            reference_to_spawn=reference_to_spawn,
        )
        self.actor = try_spawn_actor(client.world, bp, tf)
        self.actor.set_autopilot(True)
        super().__init__(
            name=name,
            spawn=spawn,
            reference_to_spawn=reference_to_spawn,
            client=client,
            *args,
            **kwargs,
        )

    @property
    def ID(self):
        return self.ID_npc_global

    def initialize(self, t0, frame0):
        self.t0 = t0
        self.frame0 = frame0

    @apply_hooks
    def tick(self, timestamp: float, frame: int):
        self.timestamp = timestamp
        self.frame = frame


class CarlaActor(CarlaObject):
    id_iter_global = itertools.count()

    def __init__(
        self,
        spawn,
        pipeline: ConfigDict,
        sensors: List[ConfigDict],
        client: "CarlaClient",
        reference_to_spawn: ConfigDict,
        *args,
        **kwargs,
    ):
        self.ID_actor_global = next(self.id_iter_global)
        self.ID_actor_type = next(self.id_iter_type)
        if self.mobile:
            name = f"mobileactor-{self.ID_actor_type}"
        else:
            name = f"staticactor-{self.ID_actor_type}"
        super().__init__(
            name=name,
            spawn=spawn,
            reference_to_spawn=reference_to_spawn,
            client=client,
            *args,
            **kwargs,
        )

        # pose init
        actor_pose = self.get_pose()
        self.reference = ReferenceFrame(
            actor_pose.position.x, actor_pose.attitude.q, GlobalOrigin3D
        )

        # build sensors -- if no actor, use spawn as the reference point
        self.sensors = {}
        try:
            for s in sensors:
                self.sensors[s["name"]] = CARLA.build(
                    s, default_args={"parent": self, "client": client}
                )
        except (KeyboardInterrupt, Exception) as e:
            for s in self.sensors.values():
                s.destroy()
            raise e
        self.sensor_data_manager = DataManager()

        # build pipeline
        self._pipeline_template = pipeline
        self.pipeline = PIPELINE.build(self._pipeline_template)

    @property
    def ID(self):
        return self.ID_actor_global

    @apply_hooks
    def tick(self, timestamp: float, frame: int):
        actor_pose = self.get_pose()
        self.reference.x = actor_pose.position.x
        self.reference.q = actor_pose.attitude.q
        self.timestamp = timestamp
        self.frame = frame
        out = self._tick()
        return out

    def initialize(self, t0, frame0):
        self.t0 = t0
        self.frame0 = frame0
        for k1, sens in self.sensors.items():
            sens.initialize(t0, frame0)


@CARLA.register_module()
class CarlaStaticActor(CarlaActor):
    mobile = False
    id_iter_type = itertools.count()

    def __init__(
        self,
        spawn: Union[ConfigDict, str, int],
        sensors: List[ConfigDict],
        pipeline: ConfigDict,
        client: "CarlaClient",
        reference_to_spawn: ConfigDict = {"type": "CarlaReferenceFrame"},
    ) -> None:
        """Initialize sensors under this static actor container"""
        self.actor = None
        self.tform = parse_spawn(
            spawn=spawn,
            spawn_points=client.spawn_points,
            reference_to_spawn=reference_to_spawn,
        )
        super().__init__(
            spawn=spawn,
            reference_to_spawn=reference_to_spawn,
            pipeline=pipeline,
            sensors=sensors,
            client=client,
        )

    def _tick(self):
        data = self.sensor_data_manager.pop()
        out = self.pipeline(data)

    def get_pose(self):
        return carla_transform_to_pose(self.tform)


@CARLA.register_module()
class CarlaMobileActor(CarlaActor):
    mobile = True
    id_iter_type = itertools.count()

    def __init__(
        self,
        spawn: Union[ConfigDict, str, int],
        vehicle: Union[str, int],
        sensors: List[ConfigDict],
        pipeline: ConfigDict,
        autopilot: bool,
        destination: Union[ConfigDict, str, int, None],
        client: "CarlaClient",
        reference_to_spawn: ConfigDict = {"type": "CarlaReferenceFrame"},
    ) -> None:
        """Initialize the vehicle and attach sensors to it"""

        self.vehicle_bps = client.world.get_blueprint_library().filter("vehicle")
        bp = parse_vehicle_blueprint(vehicle=vehicle, vehicle_bps=self.vehicle_bps)
        tf = parse_spawn(
            spawn=spawn,
            spawn_points=client.spawn_points,
            reference_to_spawn=reference_to_spawn,
        )
        self.actor = try_spawn_actor(client.world, bp, tf)

        super().__init__(
            spawn=spawn,
            reference_to_spawn=reference_to_spawn,
            pipeline=pipeline,
            sensors=sensors,
            client=client,
        )

        try:
            # provide initialization
            self.timestamp = 0  # HACK
            ego_init = self.get_object_state()

            # initialize algorithms
            self.destination = parse_destination(
                destination=destination,
                spawn_points=client.spawn_points,
                reference=ego_init.reference,
            )
            self.pipeline.initialize(
                self.timestamp,
                ego_init,
                destination=self.destination,
                map_data=self.map,
            )
            self.autopilot = autopilot
            if self.autopilot:
                print("Enabling ego autopilot")
                self.actor.set_autopilot(True)
        except (KeyboardInterrupt, Exception) as e:
            self.destroy()
            raise e
        else:
            print("Spawned ego actor at {}".format(ego_init.position.x))

    def _tick(self):
        data = self.sensor_data_manager.pop()
        ctrl = self.pipeline(data)
        if not self.autopilot:
            self.apply_control(ctrl)

    def get_pose(self):
        tf = self.actor.get_transform()
        q = tforms.transform_orientation(
            carla_rotation_to_RPY(tf.rotation), "euler", "quat"
        )
        # center of the vehicle
        pos = Position([tf.location.x, -tf.location.y, tf.location.z], GlobalOrigin3D)
        att = Attitude(q, GlobalOrigin3D)
        return Pose(pos, att)

    # def apply_control(self, ctrl):
    #     VC = carla.VehicleControl(
    #         ctrl.throttle, ctrl.steer, ctrl.brake, ctrl.hand_brake, ctrl.reverse
    #     )
    #     self.actor.apply_control(VC)

    # def set_control_mode(self, mode):
    #     assert mode in ["autopilot", "manual"]
    #     if self.control_mode != mode:
    #         print(f"Setting control to: {mode} mode")
    #     else:
    #         print(f"Control already in {mode} mode")
    #     self.control_mode = mode

    # def _parse_vehicle_keys(self, keys, milliseconds):
    #     if keys[K_UP]:
    #         throttle = min(self.last_control.throttle + 0.01, 1)
    #     else:
    #         throttle = 0.0
    #     if keys[K_DOWN]:
    #         brake = min(self.last_control.brake + 0.2, 1)
    #     else:
    #         brake = 0
    #     steer_increment = 5e-4 * milliseconds  # to adjust sensitivity of steering
    #     if keys[K_LEFT]:
    #         if self.last_control.steer > 0:
    #             steer = 0
    #         else:
    #             steer = self.last_control.steer - steer_increment
    #     elif keys[K_RIGHT]:
    #         if self.last_control.steer < 0:
    #             steer = 0
    #         else:
    #             steer = self.last_control.steer + steer_increment
    #     else:
    #         steer = 0.0
    #     steer = round(min(0.7, max(-0.7, steer)), 2)
    #     hand_brake = keys[K_SPACE]
    #     reverse = keys[K_q]  # hold down q to go into reverse
    #     gear = 1 if reverse else -1
    #     return carla.VehicleControl(
    #         throttle, steer, brake, hand_brake, reverse, gear=gear
    #     )

    # def restart(self, t0, frame0, save_folder):
    #     from .client import client_ego_sensor

    #     self.destroy()
    #     self._spawn_actor()
    #     # --- make sensors attached to ego
    #     try:
    #         # # TODO: MOVE THE SENSOR OPTIONS TO A HIGHER LEVEL SOMEWHERE
    #         # sensor_options = {
    #         #     "camera": sensors.RgbCameraSensor,
    #         #     "gnss": sensors.GnssSensor,
    #         #     "gps": sensors.GnssSensor,
    #         #     "depthcam": sensors.DepthCameraSensor,
    #         #     "imu": sensors.ImuSensor,
    #         #     "lidar": sensors.LidarSensor,
    #         # }
    #         # for sens in sensor_options.values():
    #         #     sens.reset_next_id()
    #         for i, cfg_sensor in enumerate(self.cfg["sensors"]):
    #             client_ego_sensor(self, i, cfg_sensor, save_folder)
    #     except (KeyboardInterrupt, Exception) as e:
    #         self.destroy()
    #         raise e
    #     self.initialize(t0, frame0)

    # def tick(self, t_elapsed):
    #     # -- update ground truth
    #     ground_truth = self.get_ground_truth(t_elapsed, frame)
    #     self.reference.x = ground_truth.ego_state.position.x
    #     self.reference.q = ground_truth.ego_state.attitude.q

    #     # -- apply algorithms
    #     ctrl, alg_debug = self.algorithms.tick(
    #         frame,
    #         t_elapsed,
    #         self.sensor_data_manager,
    #         infrastructure=infrastructure,
    #         ground_truth=ground_truth,
    #     )
    #     if ctrl is not None:
    #         self.apply_control(ctrl)

    #     # -- check if we need to set new destination
    #     done = False
    #     if self.destination is not None:
    #         d_dest = ground_truth.ego_state.position.distance(self.destination)
    #         if (self.destination is not None) and d_dest < 20:
    #             if self.roaming:
    #                 dest = parse_destination("random", self.client.spawn_points)
    #                 dest = np.array(
    #                     [dest.x, -dest.y, dest.z]
    #                 )  # put into avstack coordinates
    #                 dest_true = self.algorithms.set_destination(
    #                     dest, coordinates="avstack"
    #                 )
    #                 self.destination = dest_true
    #             else:
    #                 done = True
    #     debug = {
    #         "algorithms": alg_debug,
    #     }
    #     return done, debug

    # def add_sensor(self, sensor_name, sensor):
    #     assert sensor_name not in self.sensors
    #     self.sensors[sensor_name] = sensor
    #     self.sensor_IDs[sensor_name] = sensor.ID
    #     print(f"Added {sensor_name} sensor")

    # def draw_waypoint(self, plan):
    #     wpt = plan.top()[1]
    #     loc = Location(wpt.location.x, -wpt.location.y, wpt.location.z)
    #     self.world.debug.draw_point(loc, size=1 / 2, life_time=1 / 2)

    # def get_lane_lines(self, debug=False):
    #     """Gets lane lines in local coordinates of ego"""
    #     pose_g2l = self.get_pose()
    #     wpt_init = self.map.get_waypoint(
    #         self.actor.get_location(), project_to_road=True
    #     )
    #     wpts = wpt_init.next_until_lane_end(distance=1)
    #     if (wpts is None) or (len(wpts) < 3):
    #         # lanes = [None, None]
    #         lanes = []
    #     else:
    #         wpts_local = [
    #             [
    #                 Position(
    #                     carla_location_to_numpy_vector(wpt.transform.location),
    #                     GlobalOrigin3D,
    #                 ).change_reference(self.reference, inplace=False),
    #                 wpt.lane_width,
    #             ]
    #             for wpt in wpts
    #         ]
    #         pts_left = [
    #             Position([wpt[0], wpt[1] + lane_width / 2, wpt[2]], GlobalOrigin3D)
    #             for wpt, lane_width in wpts_local
    #         ]
    #         pts_right = [
    #             Position([wpt[0], wpt[1] - lane_width / 2, wpt[2]], GlobalOrigin3D)
    #             for wpt, lane_width in wpts_local
    #         ]
    #         lane_left = detections.LaneLineInSpace(pts_left)
    #         lane_right = detections.LaneLineInSpace(pts_right)
    #         if debug:
    #             T_l2g = pose_g2l.matrix
    #             for i in range(len(wpts) - 1):
    #                 # draw center
    #                 self.world.debug.draw_line(
    #                     wpts[i].transform.location,
    #                     wpts[i + 1].transform.location,
    #                     life_time=0.5,
    #                 )
    #                 # draw left and right
    #                 p1l = T_l2g @ pts_left[i]
    #                 p2l = T_l2g @ pts_left[i + 1]
    #                 self.world.debug.draw_line(
    #                     Location(p1l.x, -p1l.y, p1l.z),
    #                     Location(p2l.x, -p2l.y, p2l.z),
    #                     life_time=0.5,
    #                 )
    #                 p1r = T_l2g @ pts_right[i]
    #                 p2r = T_l2g @ pts_right[i + 1]
    #                 self.world.debug.draw_line(
    #                     Location(p1r.x, -p1r.y, p1r.z),
    #                     Location(p2r.x, -p2r.y, p2r.z),
    #                     life_time=0.5,
    #                 )
    #         lanes = [lane_left, lane_right]
    #     return lanes

    # def get_ground_truth(self, t_elapsed, frame, speed_limit=8):
    #     environment = EnvironmentState()
    #     environment.speed_limit = speed_limit
    #     if self._spd_temp is not None:
    #         if self._i_spd_temp < self._n_spd_temp:
    #             print(f"setting speed limit to {self._spd_temp} for now")
    #             environment.speed_limit = self._spd_temp
    #             self._i_spd_temp += 1
    #         else:
    #             self._spd_temp = None
    #             self._i_spd_temp = 0
    #     ego_state = self.get_vehicle_data_from_actor(t_elapsed)
    #     assert ego_state is not None
    #     objects = self.get_object_data_from_world(t_elapsed)
    #     lane_lines = self.get_lane_lines()
    #     return GroundTruthInformation(
    #         frame=frame,
    #         timestamp=t_elapsed,
    #         ego_state=ego_state,
    #         objects=objects,
    #         lane_lines=lane_lines,
    #         environment=environment,
    #     )

    # def get_object_data_from_world(self, t):
    # objects = []
    # for act in self.world.get_actors():
    #     if "vehicle" in act.type_id:
    #         obj_type = "Vehicle"
    #     elif "walker" in act.type_id:
    #         obj_type = "Pedestrian"
    #     elif (
    #         (act.type_id in ["spectator"])
    #         or ("traffic" in act.type_id)
    #         or ("sensor" in act.type_id)
    #     ):
    #         continue
    #     else:
    #         raise NotImplementedError(f"{act.type_id}, {act}")
    #     if act.get_location().distance(self.actor.get_location()) > 1 / 2:
    #         obj_data = wrap_actor_to_vehicle_state(t, act)
    #         if obj_data is not None:
    #             objects.append(obj_data)
    # return objects
