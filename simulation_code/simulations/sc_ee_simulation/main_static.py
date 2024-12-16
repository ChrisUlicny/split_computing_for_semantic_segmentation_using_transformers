import os
import time

from simulation_code.mec.mec_types import EntityID
from typing import Dict, cast
import random
import numpy as np
import json
from pathlib import Path

from simulation_code.simulations.sc_ee_simulation.common.stats_collector import StaticSplitStatsCollector
from simulation_code.simulations.sc_ee_simulation.common.task_events import DDPGTaskEventsListener, TaskEventsProtocol
from simulation_code.simulations.sc_ee_simulation.simulation.static_simulation import ServerOnlySimulation
from simulation_code.simulations.sc_ee_simulation.task.task_size_generator import OursTaskSizeGenerator
from simulation_code.src.city.Map import Map
from simulation_code.src.IISMotion import IISMotion
from simulation_code.mec.entities import BaseStationIISMotion
from simulation_code.src.movement.movementStrategies.MovementStrategyType import MovementStrategyType
# from simulation_code.simulations.gnn_resource_allocation.common.task_events import TaskEventsProtocol
from simulation_code.mec.sinr.sinr_map import SINRMap
from simulation_code.src.placeable.Placeable import Placeable, PlaceableFactory
from simulation_code.mec.iismotion_extension.locations_loader import LocationsLoader
from simulation_code.src.movement.ActorCollection import ActorCollection
from simulation_code.simulations.sc_ee_simulation.task.task_executor import TaskExecutor
from simulation_code.simulations.sc_ee_simulation.config import SimulationConfig, VehicleData, BaseStationData, SimulationType
from simulation_code.simulations.sc_ee_simulation.common.entities import SimulationVehicle
from simulation_code.simulations.sc_ee_simulation.task.task_generator import PeriodTaskGenerator
from simulation_code.simulations.sc_ee_simulation.common.mec import MECSimulation

import matplotlib.pyplot as plt

# with open(os.getcwd() + "/data/computer_vision/layer_data.json") as f:
#     layer_data = json.load(f, object_pairs_hook=OrderedDict)
# with open(os.getcwd() + "/data/computer_vision/output_data_ae.json") as f:
#     output_data = json.load(f, object_pairs_hook=OrderedDict)
# with open(os.getcwd() + "/data/computer_vision/ae_data.json") as f:
#     ae_data = json.load(f, object_pairs_hook=OrderedDict)


def get_layer_data(filepath, split_index, split_channels):
    if os.path.exists(filepath):
        with open(filepath, 'r') as json_file:
            results = json.load(json_file)
    else:
        raise FileNotFoundError(filepath)
    split_index = str(split_index)
    split_channels = str(split_channels)
    # Check if the specified index and split_channels exist in the results
    if split_index in results and split_channels in results[split_index]:
        return results[split_index][split_channels]
    else:
        raise ValueError(f"No entry for split index={split_index} and/or channels={split_channels}")

def _add_vehicles(
    iismotion: IISMotion,
    vehicle_config: VehicleData,
    movement_dt: float,
    event_listener: TaskEventsProtocol,
    split_index: int,
    split_channels: int,
    # strategy: Strategy,
) -> Dict[EntityID, SimulationVehicle]:
    task_generators = []
    vehicles = {}

    vehicles_collection = iismotion.createActorCollection(
        "vehicles",
        ableOfMovement=True,
        movementStrategy=MovementStrategyType.RANDOM_INTERSECTION_WAYPOINT_CITY_CUDA,
    )
    idx = 0

    for vehicle_config in vehicle_config:
        task_config = vehicle_config.task
        vehicles_movables = vehicles_collection.generateMovables(vehicle_config.count)

        for m in vehicles_movables:
            # img_generator = RandomIdGenerator()
            # strategy = Strategy()
            # strategy.update_strategy(0, -1, None, True, 4)
            # requirement = Requirements(vehicle_config.requirements.latency)

            model_data = get_layer_data('../layer_data/transformer_splits.json', split_index, split_channels)
            task_executor = TaskExecutor(
                model_data
            )
            task_size_generator = OursTaskSizeGenerator(
                task_executor
            )

            t_generator = PeriodTaskGenerator(
                period=task_config.generation_period,
                task_ttl=task_config.ttl,
                size_generator=task_size_generator,
                start_delay=random.random() * task_config.generation_delay_range,
            )
            t_generator.task_owner_id = m.id
            m.setSpeed((vehicle_config.speed_ms + (vehicle_config.speed_variation * random.random())) * movement_dt)
            task_generators.append(t_generator)
            vehicles[m.id] = SimulationVehicle(
                iismotion_movable=m,
                task_generator=t_generator,
                event_listener=event_listener,
                # requirements=requirement,
                # strategy=strategy,
                ddpg_event_listener=DDPGTaskEventsListener(),
                ma_agent=None,
                ips=vehicle_config.ips,
                prev_state=None,
                selected_action=None,
                log_idx=idx,
                weights=None,
            )
            cast(PeriodTaskGenerator, vehicles[m.id].task_generator).set_owner(vehicles[m.id])
            idx += 1

    return vehicles


class BaseStationPlaceableFactory(PlaceableFactory):
    def createPlaceable(self, locationsTable, map: Map) -> Placeable:
        return Placeable(locationsTable, map)


def create_base_station(
    placeable: Placeable, config: BaseStationData
) -> BaseStationIISMotion:
    return BaseStationIISMotion(
        ips=int(config.ips),
        bandwidth=int(config.bandwidth),
        resource_blocks=config.resource_blocks,
        tx_frequency=int(config.tx_frequency),
        tx_power=config.tx_power,
        coverage_radius=config.coverage_radius,
        iismotion_placeable=placeable,
    )


def init_locations_for_base_stations(
    map: Map, bs_collection: ActorCollection, min_radius: int
) -> None:
    loc_loader = LocationsLoader(path=os.path.join(os.getcwd() ,'simulation_code/cellCache/smallcellCache/'))
    base_stations = list(bs_collection.actorSet.values())
    locs = loc_loader.load_locations_from_file(map, f"bs_{len(base_stations)}")
    for bs, loc in zip(base_stations, locs):
        bs.setLocation(loc)


def _add_base_stations(
    iismotion: IISMotion, bs_config: BaseStationData
) -> Dict[EntityID, BaseStationIISMotion]:
    base_stations_collection = iismotion.createActorCollection(
        "base_stations",
        ableOfMovement=False,
        movementStrategy=MovementStrategyType.DRONE_MOVEMENT_CUDA,
    ).addPlaceableFactory(BaseStationPlaceableFactory(), bs_config.count)

    init_locations_for_base_stations(
        iismotion.map, base_stations_collection, bs_config.min_radius
    )

    base_stations = {}
    for placeable in base_stations_collection.actorSet.values():
        base_stations[placeable.id] = create_base_station(placeable, bs_config)

    return base_stations


def execute_whole_simulation(config: SimulationConfig, split_index, split_channels):
    for i in range(config.repeat_start_idx, config.repeat):
        for ep in range(config.episodes):
            main(config, split_index, split_channels, None, i, ep)


def main(config, split_index, split_channels, config_data = None, idx = None, ep_idx=None):
    task_event_listener = DDPGTaskEventsListener()
    map = Map.fromFile(
        os.path.join(os.getcwd(), "simulation_code/simulations/sc_ee_simulation/data/maps/map_3x3_manhattan.pkl"), radius=110, contains_deadends=False
    )

    iismotion = IISMotion(
        map,
        guiEnabled=config.simulation.gui_enabled,
        gridRows=22,
        secondsPerTick=config.simulation.dt,
    )

    vehicles = _add_vehicles(
        iismotion, config.vehicles, config.simulation.dt, task_event_listener, split_index, split_channels
    )

    base_stations = _add_base_stations(iismotion, config.base_stations)

    map_grid = iismotion.map.mapGrid

    sinr_map = SINRMap(
        x_size=40,
        y_size=40,
        latmin=map_grid.latmin,
        latmax=map_grid.latmax,
        lonmin=map_grid.lonmin,
        lonmax=map_grid.lonmax,
    )

    mec = MECSimulation(
        ue_dict=vehicles,
        bs_dict=base_stations,
        sinr_map=sinr_map,
        backhaul_datarate_capacity_Mb=10,  # not used
        task_event_listener=task_event_listener
    )

    ddpg_stats = StaticSplitStatsCollector()
    simulation = ServerOnlySimulation(
        stats_collector=ddpg_stats,
        iismotion=iismotion,
        number_of_ticks=config.simulation.steps,
        simulation_dt=config.simulation.dt,
        mec=mec,
        gui_enabled=config.simulation.gui_enabled,
        gui_timeout=config.simulation.gui_timeout,
        stepping=False,
        name=SimulationType(config.simulation.simulation_type).name,
        # plot_map=config.simulation.plot_map,
    )

    key = 0
    result_dir = config.result_dir if idx is None or "{}" not in config.result_dir else (
        config.result_dir.format(str(split_index), str(split_channels),
                                 SimulationType(config.simulation.simulation_type).name, str(idx), str(ep_idx)))
    result_name = config.result_name if key is None or "{}" not in config.result_name else config.result_name.format(
        str(key))
    # result_dir_path = os.getcwd() + result_dir + result_name
    # I want the absolute path here
    result_dir_path = result_dir + result_name
    print("Checking results dir:", result_dir_path)

    start = time.time()
    simulation.run_simulation()
    end = time.time()


    for key, stat_data in ddpg_stats.statistics.items():
        timestamps = [i.timestamp for i in stat_data]
        latency = [i.latency for i in stat_data]
        # device_latency = [i.device_latency for i in stat_data]
        # server_latency = [i.server_latency for i in stat_data]
        device_processing = [i.device_processing for i in stat_data]
        server_processing = [i.server_processing for i in stat_data]
        device_wait = [i.device_wait for i in stat_data]
        server_wait = [i.server_wait for i in stat_data]
        data_latency = [i.data_latency for i in stat_data]

        device_latency = [i.device_wait + i.device_processing for i in stat_data]
        server_latency = [i.server_wait + i.server_processing for i in stat_data]
        # exits = [i.exit for i in stat_data]
        # splits = [i.split for i in stat_data]

        # accuracy = [i.accuracy for i in stat_data]
        dropped = [i.dropped for i in stat_data]
        accepted = [i.accepted for i in stat_data]
        bandwidth = [i.bandwidth for i in stat_data]
        server_utilization = [i.server_utilization for i in stat_data]
        device_utilization = [i.device_utilization for i in stat_data]
        simulation_data = {
            "start_time": start,
            "end_time": end,
            "execution_time": end - start
        }

        result_dir = config.result_dir if idx is None or "{}" not in config.result_dir else (
            config.result_dir.format(str(split_index), str(split_channels),  str(idx)))
        result_name = config.result_name if key is None or "{}" not in config.result_name else config.result_name.format(str(key))
        # result_dir_path = os.getcwd() + result_dir + result_name
        # I want the absolute path here
        result_dir_path = result_dir + result_name
        Path(result_dir_path).mkdir(parents=True, exist_ok=True)
        # if not os.path.exists(result_dir_path + "plots"):
        #     os.makedirs(result_dir_path + "plots")
        if not os.path.exists(result_dir_path + "data"):
            os.makedirs(result_dir_path + "data")

        # plt.plot(timestamps, exits)
        # plt.savefig(result_dir_path + "plots/exits.png")
        # plt.clf()
        # plt.plot(timestamps, splits)
        # plt.savefig(result_dir_path + "plots/splits.png")
        # plt.clf()
        # plt.plot(timestamps, latency)
        # plt.savefig(result_dir_path + "latency.png")
        # plt.plot(timestamps, device_latency)
        # plt.savefig(result_dir_path + "device_latency.png")
        # plt.plot(timestamps, server_latency)
        # plt.savefig(result_dir_path + "server_latency.png")
        # plt.plot(timestamps, data_latency)
        # plt.savefig(result_dir_path + "data_latency.png")
        # plt.close()


        # statistics displayed separately for each vehicle

        if config_data is not None:
            with open(result_dir_path + "config.json", 'w') as f:
                json.dump(config_data, f, indent=4)
        with open(result_dir_path + "data/timestamps.txt", 'w') as filehandle:
            np.savetxt(filehandle, timestamps)
        # with open(result_dir_path + "data/exits.txt", 'w') as filehandle:
        #     np.savetxt(filehandle, exits)
        # with open(result_dir_path + "data/splits.txt", 'w') as filehandle:
        #     np.savetxt(filehandle, splits)
        with open(result_dir_path + "data/latency.txt", 'w') as filehandle:
            np.savetxt(filehandle, latency)
        with open(result_dir_path + "data/device_latency.txt", 'w') as filehandle:
            np.savetxt(filehandle, device_latency)
        with open(result_dir_path + "data/server_latency.txt", 'w') as filehandle:
            np.savetxt(filehandle, server_latency)
        with open(result_dir_path + "data/device_processing.txt", 'w') as filehandle:
            np.savetxt(filehandle, device_processing)
        with open(result_dir_path + "data/server_processing.txt", 'w') as filehandle:
            np.savetxt(filehandle, server_processing)
        with open(result_dir_path + "data/device_wait.txt", 'w') as filehandle:
            np.savetxt(filehandle, device_wait)
        with open(result_dir_path + "data/server_wait.txt", 'w') as filehandle:
            np.savetxt(filehandle, server_wait)

        with open(result_dir_path + "data/data_latency.txt", 'w') as filehandle:
            np.savetxt(filehandle, data_latency)
        with open(result_dir_path + "data/bandwidth.txt", 'w') as filehandle:
            np.savetxt(filehandle, bandwidth)
        # with open(result_dir_path + "data/accuracy.txt", 'w') as filehandle:
        #     np.savetxt(filehandle, accuracy)
        with open(result_dir_path + "data/dropped.txt", 'w') as filehandle:
            np.savetxt(filehandle, dropped)
        with open(result_dir_path + "data/accepted.txt", 'w') as filehandle:
            np.savetxt(filehandle, accepted)
        with open(result_dir_path + "data/device_utilization.txt", 'w') as filehandle:
            np.savetxt(filehandle, device_utilization)
        with open(result_dir_path + "data/server_utilization.txt", 'w') as filehandle:
            np.savetxt(filehandle, server_utilization)
        with open(result_dir_path + "data/simulation_data.json", 'w') as filehandle:
            json.dump(simulation_data, filehandle, indent = 4)


# if __name__ == "__main__":
#     main(SimulationConfig(**config))
#


def generate_split_graphs_separate(cloud_flops, mobile_flops, t_uplink_avg, t_downlink_avg, file):
    t_data_transfer = []

    # Nvidia GeForce 40 series (in FLOPS per second)
    avg_cloud_computational_power = 1.32 * pow(10, 15)

    avg_mobile_computational_power = 1.2 * pow(10, 12)


    # Data for different splits and their total latency
    splits = ['On Cloud', 'Encoder1', 'Encoder2', 'Encoder3', 'Encoder4', 'Decoder1', 'Decoder2', 'Decoder3',
              'On Mobile']

    # for i in range(0, len(cloud_flops)):
    #     t_data_transfer.append(t_uplink_avg[i] + t_downlink_avg[i])

    t_data_transfer = [sum(x) for x in zip(t_uplink_avg, t_downlink_avg)]

    t_mobile = [i/avg_mobile_computational_power for i in mobile_flops]
    print("Mobile processing:", t_mobile)
    t_cloud = [i/avg_cloud_computational_power for i in cloud_flops]
    print("Cloud processing:", t_cloud)
    print("Data processing:", t_data_transfer)

    # Calculate the sum of each set of values
    t_total = [sum(x) for x in zip(t_mobile, t_cloud, t_data_transfer)]

    # Create a stacked bar chart
    bar_width = 0.35
    index = np.arange(len(t_total))
    plt.figure(dpi=1200)
    plt.bar(index, t_mobile, bar_width, label='Vehicle', color='skyblue')
    plt.bar(index, t_cloud, bar_width, bottom=t_mobile, label='MEC', color='#fabc37')
    plt.bar(index, t_data_transfer, bar_width, bottom=np.array(t_mobile) + np.array(t_cloud), label='Data', color='#54f066')
    # plt.figure(figsize=(10, 6))


    plt.xlabel('Split Location')
    plt.xticks(rotation=45)
    plt.ylabel('Total Time (s)')
    plt.title('Total Time w.r.t Split Location')
    plt.xticks(index, splits)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{file}.png')
    plt.close()