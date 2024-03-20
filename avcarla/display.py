"""
Welcome to the Heads Up Display HELP for AVstack

Use Keys to toggle modes:
    
    V          : toggle index of the camera type for viewing within the category
    S          : toggle the type of cameras used for viewing (e.g., hud, ego, infrastructure)
    R          : change the representation of objects on the screen
    SHIFT + M  : enter manual control mode
    SHIFT + A  : enter autopilot control mode


If the ego is in MANUAL mode, use ARROWS for control

    UP     : throttle
    DOWN   : brake
    LEFT   : steer left
    RIGHT  : steer right
    Q      : hold to enter reverse

"""
import datetime
import math
import os
import re
import weakref
from typing import TYPE_CHECKING, List, Tuple


if TYPE_CHECKING:
    from avcarla.actor import CarlaActorManager
    from avcarla.client import CarlaClient
    from avstack.config import ConfigDict

import carla
import cv2
import numpy as np
import pygame
from avapi.visualize.snapshot import show_image_with_boxes
from avstack.sensors import ImageData
from pygame.locals import (
    K_ESCAPE,
    K_SLASH,
    KMOD_CTRL,
    KMOD_SHIFT,
    K_a,
    K_c,
    K_h,
    K_m,
    K_q,
    K_r,
    K_v,
)

from .config import CARLA


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")
    name = lambda x: " ".join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match("[A-Z].+", x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[: truncate - 1] + "\u2026") if len(name) > truncate else name


# ==============================================================================
# -- Camera Display Manager ----------------------------------------------------
# ==============================================================================


@CARLA.register_module()
class CarlaDisplay(object):
    def __init__(
        self,
        enabled: bool,
        display_size: Tuple[int, int],
        hud_cameras: List["ConfigDict"],
        manager: "CarlaActorManager",
        client: "CarlaClient",
        gamma_correction: float = 2.2,
    ):
        self.client = client
        self.recording = False
        self.surface = None
        self.sensor = None
        self.gamma_correction = gamma_correction
        self.R_UE2C = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]]).T
        self.objects = {"ground_truth": {}, "actors": []}
        self.object_representations = [
            "off",
            "gt_3d",  # "gt_2d" follows from "gt_3d"
            "object_3d",
            "object_2d",
            "track_2d",
            "track_3d",
        ]
        self.object_representation_colors = {
            "gt_3d": (0, 0, 0),
            "gt_2d": (124, 0, 0),
            "objects_3d": (0, 0, 255),
            "objects_2d": (0, 255, 0),
            "tracks_2d": (124, 124, 124),
            "tracks_3d": (255, 255, 255),
        }

        if enabled:
            # initialize hud
            self.pg_display = pygame.display.set_mode(
                (display_size[0], display_size[1]), pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            self.hud = HUD(display_size[0], display_size[1])

            # initialize keyboard control
            try:
                self.keyboard_control = KeyboardControl(self)
            except (KeyboardInterrupt, Exception) as e:
                self.destroy()
                raise e
            self.client.on_tick(self.hud.on_world_tick)

            # set the camera view parameters
            assert len(hud_cameras) > 0
            assert len(manager.objects) > 0
            self._hud_cameras = hud_cameras
            self._actors = manager.objects
            self._actors_cameras = [
                [s for k, s in act.sensors.items() if "camera" in k]
                for act in manager.objects
            ]
            self.actor_host_idx = 0
            self.actor_camera_index = 0
            self.object_representation_index = 0
            self.set_camera()
        else:
            self.pg_display = None
            self.hud = None
            self.keyboard_control = None

    @property
    def actor(self):
        return self._actors[self._actor_host_idx]

    @property
    def actor_cameras(self):
        return self._actors_cameras[self._actor_host_idx]

    @property
    def hud_cameras(self):
        return [] if "static" in self.actor.name else self._hud_cameras

    @property
    def object_representation_name(self):
        return self.available_object_representations[self.object_representation_index]

    @property
    def actor_host_idx(self):
        return self._actor_host_idx

    @actor_host_idx.setter
    def actor_host_idx(self, index):
        self._actor_host_idx = index % len(self._actors)

    @property
    def actor_camera_index(self):
        return self._actor_camera_index

    @actor_camera_index.setter
    def actor_camera_index(self, index):
        self._actor_camera_index = index % (
            len(self.hud_cameras) + len(self.actor_cameras)
        )

    @property
    def object_representation_index(self):
        return self._object_representation_index

    @object_representation_index.setter
    def object_representation_index(self, index):
        if index == 0:
            self._object_representation_index = index
        else:
            self._object_representation_index = index % len(
                self.available_object_representations
            )

    @property
    def available_object_representations(self):
        # try:
        #     reprs = ["off"] + [
        #         rep
        #         for rep in self.object_representations
        #         if (rep != "off")
        #         and (
        #             self.objects[self.object_camera_host_name].get(rep, None)
        #             is not None
        #         )
        #     ]
        #     if "gt_3d" in reprs:
        #         reprs.insert(reprs.index("gt_3d") + 1, "gt_2d")
        # except KeyError as e:
        #     reprs = ["off"]
        # return reprs
        return ["off"]

    def print_init(self):
        print(
            "\nDisplay Manager ready with the following view hosts and sensors:\n{}".format(
                "\n".join(
                    "  {} - {} HUD views, {} actor sensors".format(
                        host.name,
                        0 if "static" in host.name else len(self._hud_cameras),
                        len(self._actors_cameras[i]),
                    )
                    for i, host in enumerate(self._actors)
                )
            )
        )

    def tick(self, client, clock, debug):
        # self.set_objects("hud", debug["ground_truth"]["objects"])
        # self.set_objects("ego", debug["ego"]["algorithms"]["objects"])
        self.hud.tick(client, self.actor, clock, debug)

    def destroy(self):
        try:
            self.sensor.destroy()
        except RuntimeError as e:
            # usually because it was already destroyed somehow
            pass
        finally:
            self.sensor = None

    def restart(self):
        self.destroy()
        self.set_camera()

    def render(self):
        if self.surface is not None:
            self.pg_display.blit(self.surface, (0, 0))
        self.hud.render(self.pg_display)
        pygame.display.flip()

    def toggle_camera_index(self):
        """Swap between cameras within a class of views"""
        self.actor_camera_index += 1
        self.set_camera(
            notify=True,
            force_respawn=True,
        )

    def toggle_actor_index(self):
        """Swap the class of views"""
        if len(self._actors) > 1:
            self.actor_host_idx += 1
            self.actor_camera_index = 0
            self.set_camera(
                notify=True,
                force_respawn=True,
            )

    def set_camera(self, notify=True, force_respawn=True):
        respawn = (self.sensor is None) or force_respawn
        if respawn:
            # Must destroy the old viewing sensor
            if self.sensor is not None:
                self.sensor.destroy()

            # get the actor and camera
            n_total_cameras = len(self.hud_cameras) + len(self.actor_cameras)
            if self.actor_camera_index < len(self.hud_cameras):
                self.sensor = CARLA.build(
                    self.hud_cameras[self.actor_camera_index],
                    default_args={"client": self.client, "parent": self.actor},
                )
            else:
                camera = self.actor_cameras[
                    self.actor_camera_index - len(self.hud_cameras)
                ]
                self.sensor = camera.factory(do_spawn=False, do_listen=False)
            self.sensor.spawn()

            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.object.listen(
                lambda image: CarlaDisplay._parse_image(weak_self, image)
            )
        if notify:
            notify_str = "Set display sensor to {} of host {}/{}, camera {}/{}".format(
                self.sensor.name,
                self.actor_host_idx + 1,
                len(self._actors),
                self.actor_camera_index + 1,
                n_total_cameras,
            )
            self.hud.notification(notify_str, seconds=2)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification("Recording %s" % ("On" if self.recording else "Off"))

    def toggle_object_representation(self):
        self.object_representation_index += 1
        print(
            f"\n\nRepresentation set as: {self.object_representation_name} from choices {self.available_object_representations}\n"
        )

    def add_objects_to_image(self, img, d_thresh=100):
        # pull off values in case of changes
        repr_name = self.object_representation_name

        img1 = img.astype(np.uint8).copy()
        if repr_name != "off":
            cam_calib = self.sensor.calibration

            # # Pull off objects
            # if ("2d" not in repr_name) or (repr_name == "gt_2d"):
            #     objects = [
            #         obj
            #         for obj in self.objects["actors"][self.actor_host_idx]["gt_3d"]
            #         if maskfilters.box_in_fov(obj.box, cam_calib, d_thresh=d_thresh)
            #     ]
            # else:
            #     objects = self.objects[obj_cam_host_name][repr_name]

            # # Project gt_3d for gt_2d
            # if repr_name == "gt_2d":
            #     objects = [
            #         obj.box3d.project_to_2d_bbox(self.sensor.calibration)
            #         for obj in objects
            #     ]

            # # 3D can always be shown
            # if "3d" in repr_name:
            #     pass
            # # 2D can only be shown in the same camera
            # elif "2d" in repr_name:
            #     if repr_name == "gt_2d":
            #         # can always convert gt 3d to the viewing camera
            #         pass
            #     else:
            #         objects = self.objects[repr_name]
            # else:
            #     raise NotImplementedError(repr_name)
            objects = []

            # Augment the image with the representations
            img_data = ImageData(
                timestamp=0,
                frame=0,
                data=img1,
                calibration=cam_calib,
                source_ID=self.camera_view_index,
            )

            img1 = show_image_with_boxes(
                img=img_data,
                boxes=objects,
                box_colors=self.object_representation_colors[repr_name],
                show=False,
                return_images=True,
                show_IDs=True,
            )
        return img1

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensor.blueprint_name.startswith("sensor.lidar"):
            points = np.frombuffer(image.raw_data, dtype=np.dtype("f4"))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensor.blueprint_name.startswith("sensor.camera.dvs"):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(
                image.raw_data,
                dtype=np.dtype(
                    [
                        ("x", np.uint16),
                        ("y", np.uint16),
                        ("t", np.int64),
                        ("pol", np.bool),
                    ]
                ),
            )
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[
                dvs_events[:]["y"], dvs_events[:]["x"], dvs_events[:]["pol"] * 2
            ] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        else:
            image.convert(self.sensor.converter)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            # -- add object boxes
            array = array[:, :, ::-1]  # HACK for now
            array = self.add_objects_to_image(array)
            # -- scale the image to fit the HUD, preserving aspect and padding
            if array.shape != self.hud.dim:
                array = scale_image_with_padding(
                    array, width=self.hud.dim[0], height=self.hud.dim[1]
                )
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk("_out/%08d" % image.frame)


def scale_image_with_padding(img, width, height):
    """Scale image with consistent aspect ratio and add padding"""
    # -- scale with constant aspect
    scale_width = width / img.shape[1]
    scale_height = height / img.shape[0]
    scale = min(scale_width, scale_height)  # take min to ensure no cropping needed
    dim = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

    # -- fill the remainder
    dw = width - img.shape[1]
    assert dw >= 0, dw
    dh = height - img.shape[0]
    assert dh >= 0, dh
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left
    black = [0, 0, 0]
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=black
    )

    return img


class KeyboardControl(object):
    """class that handles keyboard input"""

    def __init__(self, display_manager):
        display_manager.hud.notification("Press 'H' for help", seconds=3.0)
        self._display_manager = display_manager

    def parse_events(self, world):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_r:
                    self._display_manager.toggle_object_representation()
                elif event.key == K_c:  # camera
                    self._display_manager.toggle_camera_index()
                elif event.key == K_v:  # vehicle
                    self._display_manager.toggle_actor_index()
                elif event.key == K_h or (
                    event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT
                ):
                    self._display_manager.hud.help.toggle()
                elif event.key == K_m and pygame.key.get_mods() & KMOD_SHIFT:
                    self._display_manager._parent.set_control_mode("manual")
                elif event.key == K_a and pygame.key.get_mods() & KMOD_SHIFT:
                    self._display_manager._parent.set_control_mode("autopilot")

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- Heads Up Display ----------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width=800, height=600):
        pygame.init()
        pygame.font.init()

        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = "courier" if os.name == "nt" else "mono"
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = "ubuntumono"
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == "nt" else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self.simulation_time_elapsed = 0
        self.world_map = None
        self.disp_name = None
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        if self.simulation_time == 0:
            self.simulation_time = timestamp.elapsed_seconds
        self.simulation_time_elapsed += timestamp.elapsed_seconds - self.simulation_time
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, client, actor, clock, debug):
        self._notifications.tick(client.world, clock)
        if not self._show_info:
            return
        if self.world_map is None:
            self.world_map = client.map.name
        if self.disp_name is None:
            try:
                self.disp_name = get_actor_display_name(actor.actor, truncate=20)
            except AttributeError:
                pass

        act = actor.actor if actor.actor else list(actor.sensors.values())[0].object

        t_truth = act.get_transform()
        v_truth = act.get_velocity()
        try:
            c_truth = act.get_control()
        except AttributeError:
            c_truth = None

        # Set IMU fields
        try:
            accel = act.imu_sensor.accelerometer
            gyro = act.imu_sensor.gyroscope
            compass = act.imu_sensor.compass
        except AttributeError:
            accel = (-1, -1, -1)
            gyro = (-1, -1, -1)
            compass = 0
        heading = "N" if compass > 270.5 or compass < 89.5 else ""
        heading += "S" if 90.5 < compass < 269.5 else ""
        heading += "E" if 0.5 < compass < 179.5 else ""
        heading += "W" if 180.5 < compass < 359.5 else ""

        # Set GNSS fields
        try:
            lat, lon = (act.gnss_sensor.lat, act.gnss_sensor.lon)
        except AttributeError:
            lat, lon = (0, 0)

        # TODO: FIX THIS
        # colhist = ego.collision_sensor.get_collision_history()
        # collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        collision = [0]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = client.world.get_actors().filter("vehicle.*")

        # Creat the test for the display
        self._info_text = [
            "Server:  % 16.0f FPS" % self.server_fps,
            "Client:  % 16.0f FPS" % clock.get_fps(),
            "",
            "Vehicle: % 20s" % self.disp_name,
            "Map:     % 20s" % self.world_map,
            "Simulation time: % 12s"
            % datetime.timedelta(seconds=int(self.simulation_time)),
            "",
            "Speed:   % 15.0f m/s"
            % (math.sqrt(v_truth.x**2 + v_truth.y**2 + v_truth.z**2)),
            "Location (avstack):% 20s"
            % ("(% 5.1f, % 5.1f)" % (t_truth.location.x, -t_truth.location.y)),
            "Height:  % 18.0f m" % t_truth.location.z,
            "",
        ]

        if isinstance(c_truth, carla.VehicleControl):
            self._info_text += [
                ("Throttle:", c_truth.throttle, 0.0, 1.0),
                ("Steer:", c_truth.steer, -1.0, 1.0),
                ("Brake:", c_truth.brake, 0.0, 1.0),
                ("Reverse:", c_truth.reverse),
                ("Hand brake:", c_truth.hand_brake),
                ("Manual:", c_truth.manual_gear_shift),
                "Gear:        %s" % {-1: "R", 0: "N"}.get(c_truth.gear, c_truth.gear),
            ]
        elif isinstance(c_truth, carla.WalkerControl):
            self._info_text += [
                ("Speed:", c_truth.speed, 0.0, 5.556),
                ("Jump:", c_truth.jump),
            ]
        self._info_text += [
            "",
            "Collision:",
            collision,
            "",
            "Number of vehicles: % 8d" % len(vehicles),
        ]
        if len(vehicles) > 1:
            self._info_text += ["Nearby vehicles:"]
            distance = lambda l: math.sqrt(
                (l.x - t_truth.location.x) ** 2
                + (l.y - t_truth.location.y) ** 2
                + (l.z - t_truth.location.z) ** 2
            )
            vehicles = [
                (distance(x.get_location()), x) for x in vehicles if x.id != act.id
            ]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append("% 4dm %s" % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text("Error: %s" % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [
                            (x + 8, v_offset + 8 + (1.0 - y) * 30)
                            for x, y in enumerate(item)
                        ]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(
                            display, (255, 255, 255), rect, 0 if item[1] else 1
                        )
                    else:
                        rect_border = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (bar_width, 6)
                        )
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + f * (bar_width - 6), v_offset + 8),
                                (6, 6),
                            )
                        else:
                            rect = pygame.Rect(
                                (bar_h_offset, v_offset + 8), (f * bar_width, 6)
                            )
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""

    def __init__(self, font, width, height):
        lines = __doc__.split("\n")
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)
