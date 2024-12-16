from simulation_code.src.city.Map import Map
from simulation_code.src.placeable.stationary.BaseStation import BaseStation


class FemtoCell(BaseStation):
    def __init__(self, locationsTable, map:Map):
        super(FemtoCell, self).__init__(locationsTable, map)
