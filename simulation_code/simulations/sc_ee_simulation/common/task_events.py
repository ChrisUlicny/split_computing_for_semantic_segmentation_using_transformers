from typing import Protocol

import numpy as np

from simulation_code.mec.tasks.task import Task


class TaskEventsProtocol(Protocol):
    def accepted_task(self, task: Task) -> None:
        pass

    def generated_task(self, task: Task) -> None:
        pass

    def discarded_task(self, task: Task) -> None:
        pass


class DummyTaskEventsListener(TaskEventsProtocol):
    def accepted_task(self, task: Task) -> None:
        pass

    def generated_task(self, task: Task) -> None:
        pass

    def discarded_task(self, task: Task) -> None:
        pass


class DDPGTaskEventsListener(TaskEventsProtocol):
    def __init__(self):
        self.execution_time = 0
        self.executed_tasks = 0
        self.transmission_data_size = 0
        self.transmission_data_time = 0
        self.discarded_tasks = 0
        self.accepted_tasks = 0
        self.generated_tasks = 0
        # self.sum_exits = 0
        self.device_utilization = 0
        self.server_utilization = 0
        self.correct_predictions = []
        # self.device_latency = 0
        # self.server_latency = 0
        self.wait_latency_tasks_ue = 0
        self.wait_latency_tasks_server = 0
        self.device_processing = 0
        self.server_processing = 0
        self.n_waited_tasks_ue = 0
        self.n_waited_tasks_server = 0

    def reset(self):
        self.execution_time = 0
        self.executed_tasks = 0
        self.transmission_data_size = 0
        self.transmission_data_time = 0
        self.discarded_tasks = 0
        self.accepted_tasks = 0
        self.generated_tasks = 0
        # self.sum_exits = 0
        self.device_utilization = 0
        self.server_utilization = 0
        # self.device_latency = 0
        # self.server_latency = 0
        self.wait_latency_tasks_ue = 0
        self.wait_latency_tasks_server = 0
        self.n_waited_tasks_ue = 0
        self.n_waited_tasks_server = 0
        self.device_processing = 0
        self.server_processing = 0
        # self.correct_predictions.clear()


    def get_average_latency(self) -> float:
        return 0 if self.executed_tasks == 0 else self.execution_time / self.executed_tasks

    def get_average_data_latency(self) -> float:
        return 0 if self.executed_tasks == 0 else self.transmission_data_time / self.executed_tasks

    def get_average_server_processing(self) -> float:
        return 0 if self.executed_tasks == 0 else self.server_processing / self.executed_tasks

    def get_average_device_processing(self) -> float:
        return 0 if self.executed_tasks == 0 else self.device_processing / self.executed_tasks

    def get_average_wait_ue(self) -> float:
        return 0 if self.n_waited_tasks_ue == 0 else self.wait_latency_tasks_ue / self.n_waited_tasks_ue

    def get_average_wait_server(self) -> float:
        return 0 if self.n_waited_tasks_server == 0 else self.wait_latency_tasks_server / self.n_waited_tasks_server

    def get_average_transmission_rate(self) -> float:
        return 0 if self.transmission_data_time == 0 else self.transmission_data_size / self.transmission_data_time

    def get_accuracy(self) -> float:
        if len(self.correct_predictions) == 0:
            return 0
        return np.average(self.correct_predictions)

    def get_average_exit(self) -> float:
        return 0 if self.executed_tasks == 0 else self.sum_exits / self.executed_tasks

    def accepted_task(self, task: Task) -> None:
        self.correct_predictions.append(1 if "prediction" in task.data and task.data["prediction"] == task.data["target"] else 0)
        self.save(task)
        self.accepted_tasks += 1

    def generated_task(self, task: Task) -> None:
        self.generated_tasks += 1

    def discarded_task(self, task: Task) -> None:
        self.correct_predictions.append(0)
        self.save(task)
        self.discarded_tasks += 1

    def save(self, task: Task) -> None:
        if round(task.ttl, 7) < round(task.lived_timestep, 7):
            raise Exception("Task went over time to live {}".format(task.lived_timestep))
        self.execution_time += task.lived_timestep
        self.executed_tasks += 1
        self.transmission_data_time += 0 if "uplink" not in task.data.keys() else task.data["uplink"]
        self.transmission_data_size += task.request_size - task.request.remaining_computation
        self.transmission_data_time += 0 if "downlink" not in task.data.keys() else task.data["downlink"]
        self.transmission_data_size += task.response_size - task.response.remaining_computation
        # self.sum_exits += task.data["chosen_exit"] if "chosen_exit" in task.data else -100
        self.device_utilization += 0 if "computation" not in task.data.keys() else 0 if "vehicle" not in task.data["computation"].keys() else task.data["computation"]["vehicle"]
        self.server_utilization += 0 if "computation" not in task.data.keys() else 0 if "server" not in task.data["computation"].keys() else task.data["computation"]["server"]

        self.wait_latency_tasks_ue += 0 if "latencies" not in task.data.keys() else (
            0 if "waiting" not in task.data["latencies"].keys() else (
                0 if "vehicle" not in task.data["latencies"]["waiting"].keys() else (
                    0 if task.data["latencies"]["waiting"]["vehicle"] == 0 else task.data["latencies"]["waiting"][
                        "vehicle"])))
        self.wait_latency_tasks_server += 0 if "latencies" not in task.data.keys() else (
            0 if "waiting" not in task.data["latencies"].keys() else (
                0 if "server" not in task.data["latencies"]["waiting"].keys() else (
                    0 if task.data["latencies"]["waiting"]["server"] == 0 else task.data["latencies"]["waiting"][
                        "server"])))


        self.device_processing += 0 if "latencies" not in task.data.keys() else (
            0 if "processing" not in task.data["latencies"].keys() else (
                0 if "vehicle" not in task.data["latencies"]["processing"].keys() else (
                    0 if task.data["latencies"]["processing"]["vehicle"] == 0 else task.data["latencies"]["processing"][
                        "vehicle"])))
        self.server_processing += 0 if "latencies" not in task.data.keys() else (
            0 if "processing" not in task.data["latencies"].keys() else (
                0 if "server" not in task.data["latencies"]["processing"].keys() else (
                    0 if task.data["latencies"]["processing"]["server"] == 0 else task.data["latencies"]["processing"][
                        "server"])))

        # self.device_processing += 0 if "computation_time" not in task.data.keys() else 0 if "vehicle" not in task.data[
        #     "computation_time"].keys() else task.data["computation_time"]["vehicle"]
        # self.server_processing += 0 if "computation_time" not in task.data.keys() else 0 if "server" not in task.data[
        #     "computation_time"].keys() else task.data["computation_time"]["server"]


        self.n_waited_tasks_ue += 0 if "latencies" not in task.data.keys() else ( 0 if "waiting" not in task.data["latencies"].keys() else ( 0 if "vehicle" not in task.data["latencies"]["waiting"].keys() else (0 if task.data["latencies"]["waiting"]["vehicle"] == 0 else 1)))
        self.n_waited_tasks_server += 0 if "latencies" not in task.data.keys() else ( 0 if "waiting" not in task.data["latencies"].keys() else ( 0 if "server" not in task.data["latencies"]["waiting"].keys() else (0 if task.data["latencies"]["waiting"]["server"] == 0 else 1)))

        # self.device_latency += self.wait_latency_tasks_ue + self.device_processing
        # self.server_latency += self.wait_latency_tasks_server + self.server_processing

        # print(f"device_latency: {self.device_latency} = waiting ue: {self.wait_latency_tasks_ue} + device_processing: {self.device_processing}")
        # print(f"device_latency: {self.server_latency} = waiting ue: {self.wait_latency_tasks_server} + device_processing: {self.server_latency}")
        # print(f"Task lived: {task.lived_timestep}")


        if "latencies" in task.data.keys() and "waiting" in task.data["latencies"].keys() and "processing" in task.data[
            "latencies"].keys() and "uplink" in task.data.keys():
            if round(task.data["latencies"]["processing"]["server"] + task.data["uplink"] +
                     task.data["latencies"]["processing"]["vehicle"] + task.data["latencies"]["waiting"]["server"] +
                     task.data["latencies"]["waiting"]["vehicle"], 8) != round(task.lived_timestep, 8):
                raise Exception("Something went wrong: {} | {} | {} | {} | {} | {}".format(task.lived_timestep,
                                                                                                task.data["latencies"][
                                                                                                    "processing"][
                                                                                                    "server"],
                                                                                                task.data["uplink"],
                                                                                                task.data["latencies"][
                                                                                                    "processing"][
                                                                                                    "vehicle"],
                                                                                                task.data["latencies"][
                                                                                                    "waiting"][
                                                                                                    "server"],
                                                                                                task.data["latencies"][
                                                                                                    "waiting"][
                                                                                                    "vehicle"]))

        if len(self.correct_predictions) > 200:
            self.correct_predictions.pop(0)
