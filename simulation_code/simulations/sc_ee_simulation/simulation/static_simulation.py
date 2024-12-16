from typing import List
import attr

from simulation_code.mec.simulation import IISMotionSimulation, SimulationContext
from simulation_code.mec.tasks.task import TaskComputationEntity
from simulation_code.mec.mec import MEC
from simulation_code.mec.utils.plot_utils import plot_map_and_nodes
from simulation_code.mec.entities import MecEntity, UEsMap
from simulation_code.simulations.sc_ee_simulation.common.entities import SimulationVehicle
from simulation_code.simulations.sc_ee_simulation.common.stats_collector import StaticSplitStatsCollector
from simulation_code.simulations.sc_ee_simulation.common.task_events import DDPGTaskEventsListener
from simulation_code.mec.datarate import RadioDataRate
from typing import cast

import osmnx as ox
import matplotlib.pyplot as plt
import copy
import numpy as np
import random


@attr.s(kw_only=True)
class ServerOnlySimulation(IISMotionSimulation):
    mec: MEC = attr.ib()
    stats_collector: StaticSplitStatsCollector = attr.ib()

    def plott_map_and_nodes(self, G_map, nodes: list[MecEntity], filename: str):
        node_id = max(G_map.nodes) + 1
        G = copy.deepcopy(G_map)
        for node in nodes:
            location = node.location
            G.add_node(node_id, y=location.latitude, x=location.longitude)
            node_id += 1

        fig, ax = ox.plot_graph(G, save=True, filepath=filename, show=False)
        plt.close(fig)

    def log_data(self, context: SimulationContext):
        for bs in self.mec.bs_dict.values():
            connected_ues_mapping = cast(UEsMap, self.mec.network.connected_ues(bs.id))
            connected_ues = list(connected_ues_mapping.values())
            for ue in connected_ues:
                ue = cast(SimulationVehicle, ue)
                self.stats_collector.timestamp = context.sim_info.simulation_step
                # self.stats_collector.split = self.get_split_idx(prev_action[0])
                self.stats_collector.latency = ue.ddpg_event_listener.get_average_latency()
                # self.stats_collector.device_latency = (ue.ddpg_event_listener.get_average_device_processing() +
                #                                        ue.ddpg_event_listener.get_average_wait_ue())
                # self.stats_collector.server_latency = (ue.ddpg_event_listener.get_average_server_processing() +
                #                                        ue.ddpg_event_listener.get_average_wait_server())

                self.stats_collector.device_processing = ue.ddpg_event_listener.get_average_device_processing()
                self.stats_collector.device_wait = ue.ddpg_event_listener.get_average_wait_ue()
                self.stats_collector.server_processing = ue.ddpg_event_listener.get_average_server_processing()
                self.stats_collector.server_wait = ue.ddpg_event_listener.get_average_wait_server()
                self.stats_collector.data_latency = ue.ddpg_event_listener.get_average_data_latency()
                self.stats_collector.bandwidth = ue.ddpg_event_listener.get_average_transmission_rate()
                self.stats_collector.accepted = ue.ddpg_event_listener.accepted_tasks
                self.stats_collector.dropped = ue.ddpg_event_listener.discarded_tasks
                self.stats_collector.server_utilization = ue.ddpg_event_listener.server_utilization
                self.stats_collector.device_utilization = ue.ddpg_event_listener.device_utilization
                self.stats_collector.create_statistic(ue.log_idx)
                # print(ue.used_timestep_interval)
                ue.ddpg_event_listener.reset()

    def reset_used_timestamp(self, context: SimulationContext):
        tasks: List[TaskComputationEntity] = []
        sim_dt = context.sim_info.simulation_dt
        for ue in self.mec.ue_dict.values():
            tasks += ue.tasks_computing + ue.tasks_sending + ue.tasks_to_compute
            ue.used_timestep_interval = 0
        for bs in self.mec.bs_dict.values():
            tasks += bs.tasks_computing + bs.tasks_sending
            bs.used_timestep_interval = 0
        for task in tasks:
            if round(task.task.used_timestep_interval, 10) > sim_dt:
                raise Exception(
                    "used more timestamp for task then is the simulation step"
                )
            task.task.used_timestep_interval = 0
            task.task.used_timestep += sim_dt
            if task.task.lived_timestep >= task.task.ttl:
                task.task.used_timestep = task.task.ttl + task.task.created_timestep
                self.mec.on_deadline(task.task)

    def generate_tasks(self, context: SimulationContext) -> None:
        for ue in self.mec.ue_dict.values():
            ue.generate_tasks(context.sim_info.simulation_dt)

            # offloading decision -> Offload all tasks
            ue.tasks_computing += [t.computing_ue for t in ue.tasks_to_compute if
                                   t.computing_ue.remaining_computation > 0]
            ue.tasks_sending += [t.request for t in ue.tasks_to_compute if
                                 t.computing_ue.remaining_computation == 0]
            ue.tasks_to_compute.clear()

    def simulation_step(self, context: SimulationContext) -> None:
        if self.gui_enabled:
            self.plott_map_and_nodes(
                self.iismotion.map.driveGraph,
                list(self.mec.ue_dict.values()) + list(self.mec.bs_dict.values()),
                "mecmap.png",
            )

        self.reset_used_timestamp(context)
        self.generate_tasks(context)
        self.log_data(context)
        # self.update_strategy(context)

        # update mec and network params
        self.mec.update_network(context)

        self.mec.process_ue_computation(context)
        self.mec.update_network(context)
        self.mec.process_uplink(context)
        self.mec.process_computation(context)
        self.mec.process_downlink(context)
        # self.mec.process_ue_computation(context)
        # print(np.average(self.rewards))

        # move with vehicles
        self.iismotion.stepAllCollections(context.sim_info.new_day)
