import carla

from .config import CARLA


@CARLA.register_module()
class CarlaClient:
    def __init__(
        self,
        connect_ip: str,
        connect_port: int,
        traffic_manager_port: int,
        synchronous: bool,
        rate: float,
    ) -> None:
        self.client = carla.Client(connect_ip, connect_port)
        self.client.set_timeout(2.0)
        self.traffic_manager = self.client.get_trafficmanager(traffic_manager_port)
        self.traffic_manager.set_synchronous_mode(synchronous)
        self.world = self.client.get_world()
        self._orig_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous
        settings.fixed_delta_seconds = 1.0 / rate
        self.world.apply_settings(settings)
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()

    def client_npcs(self):
        raise

    def tick(self):
        self.world.tick()

    def on_tick(self, func):
        self.world.on_tick(func)
