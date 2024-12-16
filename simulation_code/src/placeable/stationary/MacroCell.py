from simulation_code.src.city.Map import Map
from simulation_code.src.placeable.stationary.BaseStation import BaseStation


class MacroCell(BaseStation):
    def __init__(self, locationsTable, map:Map):
        super(MacroCell, self).__init__(locationsTable, map)
