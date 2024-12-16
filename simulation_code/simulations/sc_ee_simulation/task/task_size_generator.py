from simulation_code.mec.tasks.task_generators.task_size_generator import TaskSizeGenerator
from simulation_code.simulations.sc_ee_simulation.task.task_executor import TaskExecutor
from simulation_code.mec.tasks.task_size import TaskSize
from typing import NamedTuple


# @attr.s(kw_only=True)
class TaskData(NamedTuple):
    prediction: int
    target: int


class OursTaskSizeGenerator(TaskSizeGenerator):
    def __init__(self, task_executor: TaskExecutor):
        self.task_executor = task_executor

    def get_size(self) -> (TaskSize, TaskData):
        ed, es, size = self.task_executor.execute_model()
        output_size = 0
        return TaskSize(size/1048576, output_size/1048576, es, ed)
