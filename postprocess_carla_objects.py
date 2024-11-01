import argparse
import cProfile
import logging
import os
from functools import partial
from multiprocessing import Pool

from avapi.carla import CarlaScenesManager
from avstack.datastructs import DataContainer
from avstack.environment.objects import Occlusion
from avstack.maskfilters import box_in_fov
from tqdm import tqdm


def get_objects_global(CDM, frames, with_multi, chunksize=10, n_max_proc=4):
    if with_multi:
        print("Getting global objects from all frames")
        nproc = max(1, min(n_max_proc, int(len(frames) / chunksize)))
        with Pool(nproc) as p:
            part_func = partial(get_obj_glob_by_frames, CDM, True)
            objects_global = dict(
                zip(
                    frames,
                    tqdm(
                        p.imap(part_func, frames, chunksize=chunksize),
                        position=0,
                        leave=True,
                        total=len(frames),
                    ),
                )
            )
    else:
        print("Getting global objects from all frames")
        objects_global = {
            i_frame: get_obj_glob_by_frames(CDM, include_agents=True, i_frame=i_frame)
            for i_frame in tqdm(frames)
        }

    assert len(objects_global) == len(frames), "{} {}".format(
        len(objects_global, len(frames))
    )
    return objects_global


def process_func_agents(
    CDM, frames, timestamps, agents, agent_ID, objects_global, n_max_proc=4
):
    # -- remove this agent from the set of objects
    objects_global_filter = {
        frame: [obj for obj in objects if obj.ID != agent_ID]
        for frame, objects in objects_global.items()
    }

    # -- remove all infrastructure agents from the set of objects
    objects_global_filter = {
        frame: [obj for obj in objects if obj.box]
        for frame, objects in objects_global_filter.items()
    }

    # -- in agent frame
    agent_in_frames = {
        frame: [a for a in agents[frame] if a.ID == agent_ID][0] for frame in agents
    }
    process_func_sensors(
        CDM,
        f"agent-{agent_ID}",
        None,
        agent_in_frames,
        objects_global_filter,
        frames,
        timestamps,
        args.data_dir,
        with_multi=args.multi,
        n_max_proc=n_max_proc,
    )

    # -- in sensor frame
    print("Putting objects into sensor frames")
    for i_sens, (sens, sensor_frames) in enumerate(
        reversed(CDM.sensor_frames[agent_ID].items())
    ):
        print(
            "Processing {} of {} - sensor {}".format(
                i_sens + 1, len(CDM.sensor_frames[agent_ID]), sens
            )
        )
        timestamps_this = [CDM.get_timestamp(frame=frame) for frame in frames]
        frames_this = [frame for frame in sensor_frames if frame in frames]
        agent_this = {
            frame: agents
            for frame, agents in agent_in_frames.items()
            if frame in frames_this
        }
        objects_global_this = {
            frame: objects_global_filter[frame] for frame in frames_this
        }
        with_multi = False  # args.multi
        sens_save = sens + f"-{agent_ID}"
        process_func_sensors(
            CDM,
            sens,
            sens_save,
            agent_this,
            objects_global_this,
            frames_this,
            timestamps_this,
            args.data_dir,
            with_multi=with_multi,
            n_max_proc=n_max_proc,
        )


def get_obj_glob_by_frames(CDM, include_agents, i_frame):
    return CDM.get_objects_global(i_frame, include_agents=include_agents)


def process_func_sensors(
    CDM,
    sens,
    sens_save,
    agent_in_frames,
    objects_global,
    frames,
    timestamps,
    data_dir,
    with_multi,
    n_max_proc=10,
):
    """
    Post-process frames for a sensor for an agent
    """
    # no postprocessing for non-perception data
    if "imu" in sens.lower():
        return
    elif "gnss" in sens.lower():
        return

    # check number of files
    assert (
        len(agent_in_frames) == len(objects_global) == len(frames)
    ), "{}, {}, {} for {}".format(
        len(agent_in_frames), len(objects_global), len(frames), sens
    )

    # make the folder to save
    obj_sens_folder = os.path.join(
        data_dir, CDM.scene, "objects_sensor", sens_save if sens_save else sens
    )
    os.makedirs(obj_sens_folder, exist_ok=True)

    # run postprocessing function
    func = partial(process_func_frames, CDM, sens, obj_sens_folder)
    chunksize = 20
    nproc = max(1, min(n_max_proc, int(len(frames) / chunksize)))
    if with_multi:
        with Pool(nproc) as p:
            res = list(
                tqdm(
                    p.istarmap(
                        func,
                        zip(
                            agent_in_frames.values(),
                            objects_global.values(),
                            frames,
                            timestamps,
                        ),
                        chunksize=chunksize,
                    ),
                    position=0,
                    leave=True,
                    total=len(frames),
                )
            )
    else:
        for i_frame, ts in tqdm(zip(frames, timestamps), total=len(frames)):
            func(agent_in_frames[i_frame], objects_global[i_frame], i_frame, ts)


def process_func_frames(
    CDM, sens, obj_sens_folder, agent, objects_global, i_frame, timestamp
):
    # process objects into frame
    if "agent" in sens:
        agent_ref = agent.as_reference()
        objects_local = [
            obj.change_reference(agent_ref, inplace=False) for obj in objects_global
        ]
    else:
        calib = CDM.get_calibration(i_frame, agent=agent.ID, sensor=sens)

        # -- change to sensor origin
        objects_local = [
            obj.change_reference(calib.reference, inplace=False)
            for obj in objects_global
        ]

        # -- filter in view of sensors
        if ("cam" in sens.lower()) or ("radar" in sens.lower()):
            objects_local = [
                obj
                for obj in objects_local
                if box_in_fov(obj.box, calib, d_thresh=150, check_reference=True)
            ]

        # -- get depth information from dephtcam or lidar
        check_reference = True
        # prefer to get lidar data
        try:
            pc = CDM.get_lidar(
                frame=i_frame,
                sensor="lidar-0",
                agent=agent.ID,
            )  # HACK this for now....
        except Exception as e:
            pc = None
            if "cam" in sens.lower():
                # otherwise, check for depth camera
                try:
                    if "depth" in sens.lower():
                        depth_camera = sens
                    elif "semseg" in sens.lower():
                        depth_camera = sens.replace("semseg", "depth")
                    else:
                        depth_camera = sens.replace("camera", "depthcamera")
                    depth_img = CDM.get_depth_image(
                        frame=i_frame,
                        sensor=depth_camera,
                        agent=agent.ID,
                    )
                except Exception as e:
                    depth_img = None
            else:
                depth_img = None

        # -- set occlusion
        for obj in objects_local:
            if pc is not None:
                obj.set_occlusion_by_lidar(pc, check_reference=check_reference)
            elif depth_img is not None:
                obj.set_occlusion_by_depth(depth_img, check_reference=check_reference)
            else:
                print("Could not set occlusion!")

        # -- filter to only non-complete, known occlusions
        objects_local = [
            obj
            for obj in objects_local
            if obj.occlusion not in [Occlusion.COMPLETE, Occlusion.UNKNOWN]
        ]

    # -- save objects to sensor files
    objects_local = DataContainer(
        frame=i_frame,
        timestamp=timestamp,
        data=objects_local,
        source_identifier="objects",
    )
    obj_file = CDM.npc_files["frame"][i_frame].replace("npcs", "objects")
    CDM.save_objects(None, objects_local, obj_sens_folder, obj_file)


def main(args, frame_start=4, frame_end_trim=4, n_frames_max=100000, n_max_proc=4):
    CSM = CarlaScenesManager(args.data_dir)
    print(
        "Postprocessing carla dataset from {}{}".format(
            args.data_dir, "" if not args.multi else " with multiprocessing"
        )
    )
    for i_scene, CDM in enumerate(CSM):
        print("Scene {} of {}".format(i_scene + 1, len(CSM)))
        with_multi = args.multi
        frames = [f for f in CDM.frames if f >= frame_start]
        frames = frames[: max(1, min(n_frames_max, len(frames)) - frame_end_trim)]
        timestamps = [CDM.get_timestamp(frame=frame) for frame in frames]
        agents = {i: CDM.get_agents(frame=i) for i in frames}

        # get objects in global frame
        objects_global = get_objects_global(
            CDM=CDM, frames=frames, with_multi=with_multi, n_max_proc=n_max_proc
        )

        # put objects in local frames
        print("Putting objects into frames")
        for agent_ID in CDM.agent_IDs:
            process_func_agents(
                CDM=CDM,
                frames=frames,
                timestamps=timestamps,
                agents=agents,
                agent_ID=agent_ID,
                objects_global=objects_global,
                n_max_proc=n_max_proc,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument(
        "--multi", action="store_true", help="Enable for multiprocessing"
    )
    args = parser.parse_args()

    pr = cProfile.Profile()
    pr.enable()
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    finally:
        pr.disable()
        pr.dump_stats("last_run.prof")
    print("done")
