import time
import warnings
from queue import PriorityQueue

import numpy as np
from avstack.datastructs import DataManager
from avstack.geometry import GlobalOrigin3D

from . import recorder


class CarlaManager:
    def __init__(
        self,
        world,
        traffic_manager,
        cfg,
        record_truth=True,
        record_folder="sim-results/",
        buffer_dump=3,
    ):
        self._world = world
        self.traffic_manager = traffic_manager
        self._npc_manager = NpcManager(traffic_manager)
        self._ego = None
        self._running = False
        self._infrastructure = None
        self.buffer_dump = buffer_dump
        self.t0 = None
        self.frame0 = None
        self.t_elapsed = 0
        self.frame = 0
        self.sensor_data = {}
        self.cfg = cfg
        if record_truth:
            self.recorder = recorder.CarlaTruthRecorder(record_folder)
        else:
            self.recorder = None

    @property
    def ego(self):
        if self._ego is None:
            raise RuntimeError("Ego has not yet been set!")
        return self._ego

    @ego.setter
    def ego(self, ego):
        assert ego is not None, "Cannot set ego as None!"
        self._ego = ego

        # -- try initialization here...
        snap = self._world.get_snapshot()
        if not self._running:
            self.t0 = snap.timestamp.elapsed_seconds
            self.frame0 = snap.frame
            self.ego.initialize(self.t0, self.frame0)
            self._running = True
            time.sleep(2)

    @property
    def npcs(self):
        return self._npc_manager.npcs

    @npcs.setter
    def npcs(self, npcs):
        self._npc_manager.npcs = npcs

    @property
    def infrastructure(self):
        return self._infrastructure

    @infrastructure.setter
    def infrastructure(self, infra):
        self._infrastructure = infra
        self.infrastructure.initialize(self.t0, self.frame0)

    def schedule_npc_events(self, npc_cfgs):
        for i, cfg in enumerate(npc_cfgs):
            self._npc_manager.schedule(cfg, i)

    def tick(self):
        snap = self._world.get_snapshot()
        self.frame = snap.frame - self.frame0
        self.t_elapsed = snap.timestamp.elapsed_seconds - self.t0
        ground_truth = self.ego.get_ground_truth(self.t_elapsed, self.frame)
        events = self._npc_manager.retrieve(self.t_elapsed)
        self._npc_manager.apply(events)
        self.infrastructure.tick()
        done, ego_debug = self.ego.tick(self.t_elapsed, self.frame, self.infrastructure)
        if self.recorder is not None:
            self.recorder.record(self.t_elapsed, self.frame, self.ego, self.npcs)
        debug = {
            "ego": ego_debug,
            "ground_truth": {"objects": {"gt_3d": ground_truth.objects}},
        }
        return done, debug

    def _print_last_record_path(self):
        if self.recorder is not None:
            print("Last results were saved in: %s" % self.recorder.folder)
            with open("last_run.txt", "w") as f:
                str_write = "\n".join(self.recorder.folder)
                f.write(str_write)

    def destroy(self):
        self._print_last_record_path()
        self.ego.destroy()
        self._npc_manager.destroy()
        self.infrastructure.destroy()

    def restart(self, save_folder=""):
        from .bootstrap import bootstrap_infrastructure, bootstrap_npcs

        time.sleep(self.buffer_dump)  # allow time for buffers to dump...
        if self.recorder is not None:
            self._print_last_record_path()
            self.recorder.restart(save_folder=save_folder)
        snap = self._world.get_snapshot()
        self.t0 = snap.timestamp.elapsed_seconds
        self.frame0 = snap.frame
        self.t_elapsed = 0
        self.sensor_data = {}

        # -- remove existing elements:
        self._npc_manager.destroy()
        self.infrastructure.destroy()

        # -- reset EGO
        self.ego.restart(t0=self.t0, frame0=self.frame0, save_folder=save_folder)

        # -- reset NPCs
        self.npcs, npc_cfgs = bootstrap_npcs(self._world, self.cfg["world"])
        self.schedule_npc_events(npc_cfgs)

        # -- reset infrastructure sensors
        self.infrastructure = bootstrap_infrastructure(
            self._world, self.cfg["infrastructure"], self.ego, save_folder=save_folder
        )
        time.sleep(2)


class NpcManager:
    def __init__(self, traffic_manager, verbose=True):
        self._traffic_manager = traffic_manager
        self._events = PriorityQueue()
        self._npcs = []
        self._n_event_count = -1
        self.verbose = verbose

    @property
    def npcs(self):
        if not self._npcs:
            warnings.warn("NPCS were queried, but none have been set yet")
        for npc in self._npcs:
            if not npc.is_alive:
                npc.destroy()
        return [npc for npc in self._npcs if npc.is_alive]

    @npcs.setter
    def npcs(self, npcs):
        if isinstance(npcs, list):
            self._npcs.extend(npcs)
        else:
            self._npcs.append(npcs)
        self._npcs = [npc for npc in self._npcs if npc.is_alive]

    def destroy(self):
        for i, npc in enumerate(self.npcs):
            try:
                npc.destroy()
            except (KeyboardInterrupt, Exception) as e:
                print(f"Could not destroy npc {i}...continuing")

    def force_npc_lane_change(self, idx_npc: int, direction: bool):
        """Forces lane change of a particular NPC
        :idx_npc - index of NPC to force
        :direction - true=right, false=left
        """
        direct = True if direction == "right" else False
        self._traffic_manager.force_lane_change(self._npcs[idx_npc], direct)
        if self.verbose:
            print(f"::NPC Manager::Forced lane change {direction} for NPC {idx_npc}")

    def npc_speed_limit_pct_global(self, percentage: float):
        """Forces all NPCs to stay below/above a fraction of speed limit
        :percentage - pct to be below (>0) or above (<0) speed limit
        """
        self._traffic_manager.global_percentage_speed_difference(percentage)
        if self.verbose:
            print(f"::NPC Manager::Set pct speed limit to {percentage} globally")

    def npc_speed_limit_pct_single(self, idx_npc: int, percentage: float):
        """Forces a single NPC to stay below/above a fraction of speed limit
        :percentage - pct to be below (>0) or above (<0) speed limit
        """
        self._traffic_manager.vehicle_percentage_speed_difference(
            self._npcs[idx_npc], percentage
        )
        if self.verbose:
            print(
                f"::NPC Manager::Set pct speed limit to {percentage} for NPC {idx_npc}"
            )

    def apply(self, events: list):
        """Apply a list of events"""
        for event in events:
            if event["behavior"] == "pct_speed":
                self.npc_speed_limit_pct_single(event["idx_npc"], event["data"])
            elif event["behavior"] == "lane_change":
                self.force_npc_lane_change(event["idx_npc"], event["data"])
            else:
                raise NotImplementedError

    def retrieve(self, t: float):
        """Get the set of events to apply"""
        current_events = []
        while (not self._events.empty()) and (self._events.queue[0][0] <= t):
            current_events.append(self._events.get()[2])
        return current_events

    def schedule(self, cfg_npc, idx_npc):
        """add an npc's config to the schedule"""
        events = self.parse_events(cfg_npc, idx_npc)
        for event in events:
            self._events.put(event)
            if self.verbose:
                print(f"::NPC Manager::Scheduled {event[2]} for NPC {idx_npc}")

    def parse_events(self, cfg_npc, idx_npc):
        """parse a config into the events"""
        events = []
        for behavior, events_list in cfg_npc["events"].items():
            if (events_list is not None) and (len(events_list) > 0):
                for data in events_list:
                    ev = {"idx_npc": idx_npc, "behavior": behavior, "data": data[1]}
                    self._n_event_count += 1
                    events.append((data[0], self._n_event_count, ev))
        return events


class InfrastructureManager:
    def __init__(self, world):
        self.world = world
        self.map = world.get_map()
        self.actor = None  # for any method trying to use "parent" actor..
        self.sensors = {}
        self.covars = {}
        self.comm_range = {}
        self.sensor_data_manager = DataManager(max_size=5)

    @property
    def reference(self):
        return GlobalOrigin3D

    def add_sensor(
        self, source_name, sens, comm_range=50, pos_covar=np.array([0, 0, 0])
    ):
        assert source_name not in self.sensors
        self.sensors[source_name] = sens
        self.covars[source_name] = pos_covar
        self.comm_range[source_name] = comm_range

    def initialize(self, t0, frame0):
        for sens in self.sensors.values():
            sens.initialize(t0, frame0)

    def tick(self):
        pass  # sensors will automatically push data to data manager

    def destroy(self):
        for k, sens in self.sensors.items():
            try:
                sens.destroy()
            except (KeyboardInterrupt, Exception) as e:
                print(f"Could not destroy sensor {k}...continuing")
