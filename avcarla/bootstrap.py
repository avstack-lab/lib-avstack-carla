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

    def bootstrap_npcs(self):
        raise

    def tick(self):
        self.world.tick()

    def on_tick(self, func):
        self.world.on_tick(func)


def bootstrap_npcs(world, cfg):
    npcs_cfg = []
    # -- get walker info
    blueprints = world.get_blueprint_library().filter("walker.pedestrian.*")
    n_walkers = cfg["n_random_walkers"]
    spawn_points = []
    for i in range(n_walkers):
        spawn_points.append(
            carla.Transform(
                world.get_random_location_from_navigation(), carla.Rotation()
            )
        )
    npcs_walk = _spawn_agents_randomly(world, blueprints, spawn_points, n_walkers)

    # -- get vehicle info
    blueprints = world.get_blueprint_library().filter("vehicle.*")
    spawn_points = world.get_map().get_spawn_points()
    n_vehicles = cfg["n_random_vehicles"]
    npcs_veh_random = _spawn_agents_randomly(
        world, blueprints, spawn_points, n_vehicles
    )

    blueprints = world.get_blueprint_library().filter("vehicle.*")
    spawn_points = world.get_map().get_spawn_points()
    npcs_agents_placed = _spawn_agents_placed(
        world, blueprints, spawn_points, cfg["placed_agents"]
    )

    time.sleep(2)
    return npcs_walk + npcs_veh_random + npcs_agents_placed, npcs_cfg


def _spawn_agents_placed(world, blueprints, spawn_points, placed_vehicles):
    npcs = []
    try:
        for name, cfg in placed_vehicles.items():
            # -- get object blueprint
            if cfg["type"] == "vehicle":
                if cfg["idx_vehicle"] is None:
                    bp = random.choice(blueprints)
                else:
                    bp = blueprints[cfg["idx_vehicle"]]
            elif cfg["type"] == "walker":
                raise NotImplementedError
            else:
                raise NotImplementedError(cfg["type"])

            # -- get object spawn points
            if cfg["idx_spawn"] in [None, "randint"]:
                spawn_point = random.choice(spawn_points)
            else:
                spawn_point = spawn_points[cfg["idx_spawn"]]
            if cfg["delta_spawn"]:
                dspawn = carla.Vector3D(
                    x=cfg["delta_spawn"]["x"],
                    y=cfg["delta_spawn"]["y"],
                    z=cfg["delta_spawn"]["z"],
                )
                spawn_point.location += dspawn

            # -- spawn and set attributes
            npc = world.try_spawn_actor(bp, spawn_point)
            if npc is not None:
                npc.set_autopilot(cfg["autopilot"])
                npcs.append(npc)
    except (KeyboardInterrupt, Exception) as e:
        for npc in npcs:
            npc.destroy()
        raise e
    print("Successfully spawned %i npcs in placed locations" % len(npcs))

    return npcs


def _spawn_agents_randomly(world, blueprints, spawn_points, n_agents):
    # Check the number of agents
    if n_agents < len(spawn_points):
        random.shuffle(spawn_points)
    elif n_agents > len(spawn_points):
        msg = "requested %d agents, but could only find %d spawn points"
        logging.warning(msg, n_agents, len(spawn_points))
        n_agents = len(spawn_points)
    # -- spawn all agents
    walker_controller_bp = world.get_blueprint_library().find("controller.ai.walker")
    npcs = []
    i_succ = 0
    try:
        print("Spawning %d npcs randomly" % n_agents)
        for i in range(n_agents):
            bp = np.random.choice(blueprints)
            if bp.has_attribute("color"):
                color = random.choice(bp.get_attribute("color").recommended_values)
                bp.set_attribute("color", color)
            if bp.has_attribute("driver_id"):
                driver_id = random.choice(
                    bp.get_attribute("driver_id").recommended_values
                )
                bp.set_attribute("driver_id", driver_id)
            npc = world.try_spawn_actor(bp, spawn_points[i])
            if npc is not None:
                if "walker" in npc.type_id:
                    ai_controller = world.try_spawn_actor(
                        walker_controller_bp, carla.Transform(), npc
                    )
                    ai_controller.start()
                    ai_controller.go_to_location(
                        world.get_random_location_from_navigation()
                    )
                    ai_controller.set_max_speed(
                        1 + random.random()
                    )  # Between 1 and 2 m/s (default is 1.4 m/s).
                else:
                    npc.set_autopilot(True)
                npcs.append(npc)
                i_succ += 1
    except (KeyboardInterrupt, Exception) as e:
        for npc in npcs:
            npc.destroy()
        raise e
    print("Successfully spawned %i npcs randomly" % i_succ)
    return npcs


# def bootstrap_ego(
#     world, ego_stack, cfg=None, config_file="./default_ego.yml", save_folder=""
# ):
#     if cfg is None:
#         cfg = config.read_config(config_file)

#     # --- make ego
#     ego = CarlaEgoActor(world, ego_stack, cfg)

#     # --- make sensors attached to ego
#     try:
#         for sens in sensor_options.values():
#             sens.reset_next_id()
#         for i, cfg_sensor in enumerate(cfg["sensors"]):
#             bootstrap_ego_sensor(ego, i, cfg_sensor, save_folder)
#     except (KeyboardInterrupt, Exception) as e:
#         ego.destroy()
#         raise e

#     # --- make other sensors

#     return ego


# def bootstrap_infrastructure(
#     world, cfg, ego, config_file="./default_infrastructure.yml", save_folder=""
# ):
#     if cfg is None:
#         cfg = config.read_config(config_file)

#     # -- infrastructure class to act like "parent" actor
#     infra = InfrastructureManager(world)

#     # -- make sensors
#     n_infra_spawn = 0
#     try:
#         for k in cfg:
#             prev_spawns = []
#             for idx in range(cfg[k]["n_spawn"]):
#                 x_spawn = bootstrap_infra_sensor(
#                     infra, idx, prev_spawns, cfg[k], ego, save_folder=save_folder
#                 )
#                 prev_spawns.append(x_spawn)
#                 n_infra_spawn += 1
#     except (KeyboardInterrupt, Exception) as e:
#         infra.destroy()
#         raise e
#     print("Spawned %i infrastructure elements" % n_infra_spawn)
#     return infra


# def bootstrap_infra_sensor(infra, idx, prev_spawns, cfg, ego, save_folder):
#     rng = random.Random(int(cfg["seed"]) + idx)

#     # -- find spawn point
#     spawn_points = infra.map.get_spawn_points()
#     if cfg["idx_spawn"] == "random":
#         spawn_point = random.choice(spawn_points)
#     elif cfg["idx_spawn"] == "in_order":
#         spawn_point = spawn_points[cfg["idx_spawn_list"][idx]]
#     elif cfg["idx_spawn"].startswith("within"):
#         lower = int(cfg["idx_spawn"].split("-")[1])
#         upper = int(cfg["idx_spawn"].split("-")[2])
#         i_trial = 0
#         while True:
#             spawn_point = rng.choice(spawn_points)
#             x_spawn = utils.carla_location_to_numpy_vector(spawn_point.location)
#             if (
#                 lower
#                 <= np.linalg.norm(ego.get_ego_pose().position.x - x_spawn)
#                 <= upper
#             ) and (not any([np.all(x_spawn == x_prev) for x_prev in prev_spawns])):
#                 break
#             i_trial += 1
#             if i_trial >= 100:
#                 raise RuntimeError("Cannot find suitable spawn point")
#     else:
#         spawn_point = (
#             spawn_points[cfg["idx_spawn"]]
#             if cfg["idx_spawn"]
#             else random.choice(spawn_points)
#         )
#     x_spawn = utils.carla_location_to_numpy_vector(
#         spawn_point.location
#     ) + utils.carla_location_to_numpy_vector(
#         carla.Location(
#             x=cfg["transform"]["location"]["x"],
#             y=cfg["transform"]["location"]["y"],
#             z=cfg["transform"]["location"]["z"],
#         )
#     )
#     x_spawn[2] -= spawn_point.location.z  # only allow for the manual z component
#     if cfg["add_random_yaw"]:
#         random_yaw_1 = 0 if random.random() < 0.5 else np.pi  # add a random yaw flip
#         random_yaw_2 = np.random.randn() * np.pi / 8  # add small amount of random yaw
#     q_spawn = utils.carla_rotation_to_quaternion(
#         carla.Rotation(
#             pitch=cfg["transform"]["rotation"]["pitch"],
#             yaw=cfg["transform"]["rotation"]["yaw"] + random_yaw_1 + random_yaw_2,
#             roll=cfg["transform"]["rotation"]["roll"],
#         )
#     ) * utils.carla_rotation_to_quaternion(spawn_point.rotation)

#     tform_spawn = carla.Transform(
#         utils.numpy_vector_to_carla_location(x_spawn),
#         utils.quaternion_to_carla_rotation(q_spawn),
#     )

#     # -- spawn sensor
#     save_folder = os.path.join(save_folder, "sensor_data")
#     sens = sensor_options[cfg["sensor_name"]]
#     source_name = cfg["name_prefix"] + f"_{idx+1:03d}"
#     pos_covar = cfg["position_uncertainty"]
#     sens = sens(
#         source_name=source_name,
#         parent=infra,
#         tform=tform_spawn,
#         attr=cfg["attributes"],
#         mode=cfg["mode"],
#         noise=cfg["noise"],
#         save=cfg["save"],
#         save_folder=save_folder,
#     )

#     # -- add to infra
#     infra.add_sensor(
#         source_name, sens, comm_range=cfg["comm_range"], pos_covar=pos_covar
#     )
#     return utils.carla_location_to_numpy_vector(spawn_point.location)


# def bootstrap_ego_sensor(ego, ID, cfg, save_folder):
#     # --- make sensor
#     assert isinstance(cfg, dict)
#     if len(cfg) != 1:
#         import pdb

#         pdb.set_trace()
#         raise RuntimeError
#     k1 = list(cfg.keys())[0]
#     source_name = cfg[k1]["name"]
#     save_folder = os.path.join(save_folder, "sensor_data")
#     tform = carla.Transform(
#         carla.Location(
#             cfg[k1]["transform"]["location"]["x"],
#             cfg[k1]["transform"]["location"]["y"],
#             cfg[k1]["transform"]["location"]["z"],
#         ),
#         carla.Rotation(
#             cfg[k1]["transform"]["rotation"]["pitch"],
#             cfg[k1]["transform"]["rotation"]["yaw"],
#             cfg[k1]["transform"]["rotation"]["roll"],
#         ),
#     )
#     for k, sens in sensor_options.items():
#         if k in k1:
#             break
#     else:
#         raise NotImplementedError(k1)
#     ego.add_sensor(
#         k1,
#         sens(
#             source_name=source_name,
#             parent=ego,
#             tform=tform,
#             attr=cfg[k1]["attributes"],
#             mode=cfg[k1]["mode"],
#             noise=cfg[k1]["noise"],
#             save=cfg[k1]["save"],
#             save_folder=save_folder,
#         ),
#     )


# def bootstrap_hud_sensors(ego, cfg):
#     hud_sensors = []
#     for scfg in cfg:
#         k1 = list(scfg.keys())[0]
#         scfg = scfg[k1]
#         tform = carla.Transform(
#             carla.Location(
#                 scfg["transform"]["location"]["x"],
#                 scfg["transform"]["location"]["y"],
#                 scfg["transform"]["location"]["z"],
#             ),
#             carla.Rotation(
#                 scfg["transform"]["rotation"]["pitch"],
#                 scfg["transform"]["rotation"]["yaw"],
#                 scfg["transform"]["rotation"]["roll"],
#             ),
#         )
#         sens = sensors.RgbCameraSensor(
#             source_name=k1,
#             parent=ego,
#             tform=tform,
#             attr=scfg["attributes"],
#             mode=scfg["mode"],
#             noise=scfg["noise"],
#             listen=False,
#             spawn=False,
#             save=False,
#         )
#         hud_sensors.append(sens)
#     return hud_sensors


# # -------------------------------------------------------------
# # Bootstrap for test cases
# # -------------------------------------------------------------


# def bootstrap_standard(world, traffic_manager, ego_stack, cfg, save_folder):
#     # if cfg is None:
#     #     cfg = config.read_config(config_file)

#     # -- unload parked cars!
#     try:
#         world.unload_map_layer(carla.MapLayer.ParkedVehicles)
#     except AttributeError as e:
#         pass  # some version do not support this

#     ego = bootstrap_ego(world, ego_stack, cfg["ego"], save_folder=save_folder)
#     npcs, npcs_cfg = bootstrap_npcs(world, cfg["world"])
#     # try:
#     #     npcs_set, npc_cfgs = bootstrap_npcs(world, cfg)
#     # except (KeyboardInterrupt, Exception) as e:
#     #     ego.destroy()
#     #     raise e
#     try:
#         infra = bootstrap_infrastructure(
#             world, cfg["infrastructure"], ego, save_folder=save_folder
#         )
#     except (KeyboardInterrupt, Exception) as e:
#         ego.destroy()
#         for npc in npcs:
#             npc.destroy()
#         raise e

#     manager = CarlaManager(
#         world,
#         traffic_manager,
#         record_truth=cfg["recorder"]["record_truth"],
#         record_folder=save_folder,
#         cfg=cfg,
#     )
#     manager.ego = ego
#     manager.npcs = npcs
#     manager.infrastructure = infra
#     try:
#         manager.schedule_npc_events(npcs_cfg)
#     except (KeyboardInterrupt, Exception) as e:
#         manager.destroy()
#         raise e
#     return manager
