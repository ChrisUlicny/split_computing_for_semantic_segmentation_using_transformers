from typing import NamedTuple


class DDPGStatistic(NamedTuple):
    timestamp: float
    exit: float
    split: float
    autoencoder: float
    reward: float
    actor_loss: float
    critic_loss: float
    latency: float
    dropped: int
    accepted: int
    accuracy: float
    bandwidth: float
    bandwidth_2: float
    device_utilization: float
    server_utilization: float

class EdgeAIStatistic(NamedTuple):
    timestamp: float
    exit: float
    split: float
    latency: float
    dropped: int
    accepted: int
    accuracy: float
    bandwidth: float
    bandwidth_2: float
    device_utilization: float
    server_utilization: float


class StaticSplitStatistic(NamedTuple):
    timestamp: float
    latency: float
    dropped: int
    accepted: int
    bandwidth: float
    bandwidth_2: float
    device_utilization: float
    server_utilization: float
    # device_latency: float
    # server_latency: float
    server_processing: float
    server_wait: float
    device_processing: float
    device_wait: float
    data_latency: float



class StatsCollector:
    def __init__(self):
        self.timestamp = 0
        self.exit = 0
        self.split = 0
        self.autoencoder = 0
        self.dropped = 0
        self.accepted = 0
        self.accuracy = 0
        self.bandwidth = 0
        self.bandwidth_2 = 0
        self.latency = 0
        self.device_utilization = 0
        self.server_utilization = 0
        self.statistics = {}


class DDPGStatsCollector(StatsCollector):
    def __init__(self):
        super().__init__()
        self.reward = 0
        self.actor_loss = 0
        self.critic_loss = 0

    def create_statistic(self, id):
        if str(id) not in self.statistics:
            self.statistics[str(id)] = []
        self.statistics[str(id)].append(DDPGStatistic(self.timestamp, self.exit, self.split, self.autoencoder, self.reward, self.actor_loss,
                                             self.critic_loss, self.latency, self.dropped, self.accepted, self.accuracy,
                                             self.bandwidth, self.bandwidth_2, self.device_utilization, self.server_utilization))
        self.exit = 0
        self.split = 0
        self.autoencoder = 0
        self.reward = 0
        self.actor_loss = 0
        self.critic_loss = 0
        self.latency = 0
        self.timestamp = 0
        self.dropped = 0
        self.accepted = 0
        self.accuracy = 0
        self.bandwidth = 0
        self.bandwidth_2 = 0
        self.device_utilization = 0
        self.server_utilization = 0

class EdgeAIStatsCollector(StatsCollector):
    def __init__(self):
        super().__init__()

    def create_statistic(self, id):
        if str(id) not in self.statistics:
            self.statistics[str(id)] = []
        self.statistics[str(id)].append(EdgeAIStatistic(self.timestamp, self.exit, self.split, self.latency, 
                                             self.dropped, self.accepted, self.accuracy, self.bandwidth, self.bandwidth_2,
                                             self.device_utilization, self.server_utilization))
        self.exit = 0
        self.split = 0
        self.latency = 0
        self.timestamp = 0
        self.dropped = 0
        self.accepted = 0
        self.accuracy = 0
        self.bandwidth = 0
        self.bandwidth_2 = 0
        self.device_utilization = 0
        self.server_utilization = 0


class StaticSplitStatsCollector(StatsCollector):
    def __init__(self):
        super().__init__()
        # self.device_latency = 0
        # self.server_latency = 0
        self.server_processing = 0
        self.server_wait = 0
        self.device_processing = 0
        self.device_wait = 0
        self.data_latency = 0

    def create_statistic(self, id):
        if str(id) not in self.statistics:
            self.statistics[str(id)] = []
        # self.statistics[str(id)].append(StaticSplitStatistic(self.timestamp, self.latency,
        #                                      self.dropped, self.accepted, self.bandwidth, self.bandwidth_2,
        #                                      self.device_utilization, self.server_utilization,
        #                                      self.device_latency, self.server_latency, self.data_latency))
        self.statistics[str(id)].append(StaticSplitStatistic(self.timestamp, self.latency,
                                             self.dropped, self.accepted, self.bandwidth, self.bandwidth_2,
                                             self.device_utilization, self.server_utilization,
                                             self.server_processing, self.server_wait,
                                             self.device_processing, self.device_wait, self.data_latency))
        self.latency = 0
        self.timestamp = 0
        self.dropped = 0
        self.accepted = 0
        self.bandwidth = 0
        self.bandwidth_2 = 0
        self.device_utilization = 0
        self.server_utilization = 0
        # self.device_latency = 0
        # self.server_latency = 0
        self.server_processing = 0
        self.server_wait = 0
        self.device_processing = 0
        self.device_wait = 0
        self.data_latency = 0


