from typing import Any, Callable
from simulation_code.src.movement.movementStrategies.DroneMovementCuda import DroneMovementCuda
from simulation_code.src.movement.movementStrategies.MovementStrategy import MovementStrategy
from simulation_code.src.movement.movementStrategies.PreloadedLocationsStrategy import PreloadedLocationsStrategy
from simulation_code.src.movement.movementStrategies.RandomWaypointBlankEnvCuda import RandomWaypointBlankEnvCuda
from simulation_code.src.movement.movementStrategies.RandomWaypointCity import RandomWaypointCity
from simulation_code.src.movement.movementStrategies.RandomWaypointCityCuda import RandomWaypointCityCuda
from simulation_code.src.movement.movementStrategies.RandomIntersectionWaypointCityCuda import RandomIntersectionWaypointCityCuda

from simulation_code.src.movement.movementStrategies.PersonBehaviorCityCuda import PersonBehaviourCityCuda
from simulation_code.src.movement.movementStrategies.MovementStrategyType import MovementStrategyType

MovementStrategyFactoryFunctionType = Callable[[Any, Any, Any, Any, Any], MovementStrategy]

class MovementStrategyFactory:
    def getStrategy(self, type: MovementStrategyType, locationsTable, actorSet, map, mapGrid):

        if (type == MovementStrategyType.RANDOM_WAYPOINT_CITY_CUDA):
            return RandomWaypointCityCuda(locationsTable, actorSet, map, mapGrid, type)

        if (type == MovementStrategyType.RANDOM_WAYPOINT_CITY):
            return RandomWaypointCity(locationsTable, actorSet, map, mapGrid, type)

        if (type == MovementStrategyType.RANDOM_WAYPOINT_BLANK_ENV_CUDA):
            return RandomWaypointBlankEnvCuda(locationsTable, actorSet, map, mapGrid, type)

        if (type == MovementStrategyType.DRONE_MOVEMENT_CUDA):
            return DroneMovementCuda(locationsTable, actorSet, map, mapGrid, type)

        if (type == MovementStrategyType.PERSON_BEHAVIOUR_CITY_CUDA):
            return PersonBehaviourCityCuda(locationsTable, actorSet, map, mapGrid, type)

        if (type == MovementStrategyType.RANDOM_INTERSECTION_WAYPOINT_CITY_CUDA):
            return RandomIntersectionWaypointCityCuda(locationsTable, actorSet, map, mapGrid, type)

        if (type == MovementStrategyType.PRELOADED_LOCATIONS_STRATEGY):
            return PreloadedLocationsStrategy(locationsTable, actorSet, map, mapGrid, type)
        return None
