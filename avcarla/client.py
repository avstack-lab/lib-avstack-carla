import carla
import numpy as np

from .config import CARLA


@CARLA.register_module()
class CarlaClient:
    def __init__(
        self,
        connect_ip: str,
        connect_port: int,
        traffic_manager_port: int,
        traffic_manager_seed: int,
        synchronous: bool,
        rate: float,
        disable_static_actors: bool = True,
        randomize_lights: bool = True,
        prob_light_green: float = 0.35,
        seed: int = None,
        rng: np.random.RandomState = None,
    ) -> None:
        self.rng = rng if rng is not None else np.random.RandomState(seed)
        self._prob_light_green = prob_light_green
        self.client = carla.Client(connect_ip, connect_port)
        self.client.set_timeout(4.0)
        self.traffic_manager = self.client.get_trafficmanager(traffic_manager_port)
        self.traffic_manager.set_synchronous_mode(synchronous)
        if (traffic_manager_seed is None) and (seed is not None):
            traffic_manager_seed = seed
        if traffic_manager_seed is not None:
            print(f"Setting traffic manager seed as {traffic_manager_seed}")
            self.traffic_manager.set_random_device_seed(traffic_manager_seed)
        self.world = self.client.get_world()
        self._orig_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous
        settings.fixed_delta_seconds = 1.0 / rate
        self.world.apply_settings(settings)
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.spawns_chosen = []
        if randomize_lights:
            self.set_traffic_lights()
        if disable_static_actors:
            self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)

    def set_traffic_lights(self):
        list_actor = self.world.get_actors()
        for actor_ in list_actor:
            if isinstance(actor_, carla.TrafficLight):
                # for any light, first set the light state, then set time. for yellow it is
                # carla.TrafficLightState.Yellow and Red it is carla.TrafficLightState.Red
                if self.rng.rand() < self._prob_light_green:
                    actor_.set_state(carla.TrafficLightState.Green)
                else:
                    actor_.set_state(carla.TrafficLightState.Red)
                actor_.set_green_time(5.0)
                actor_.set_yellow_time(1.0)
                actor_.set_red_time(4.0)

    def client_npcs(self):
        raise

    def tick(self):
        self.world.tick()

    def on_tick(self, func):
        self.world.on_tick(func)
