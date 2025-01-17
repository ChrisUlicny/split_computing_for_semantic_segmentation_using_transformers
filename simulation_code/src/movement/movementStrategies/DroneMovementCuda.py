from __future__ import division
from numba import cuda
import math

from simulation_code.src.common.Location import Location
from simulation_code.src.movement.movementStrategies.MovementStrategy import MovementStrategy


# CUDA kernel
from simulation_code.src.placeable.movable.Person import Person


@cuda.jit
def walking_kernel(io_array):
    def toRadians(degrees):
        return degrees * math.pi / 180

    def toDegrees(radians):
        return radians * 180 / math.pi

    pos = cuda.grid(1)
    latStep = io_array[pos, 11]
    latMin = io_array[pos, 12]

    lonStep = io_array[pos, 13]
    lonMin = io_array[pos, 14]

    if pos < io_array.size:
        lat1 = io_array[pos, 0]
        lon1 = io_array[pos, 1]
        lat2 = io_array[pos, 3]
        lon2 = io_array[pos, 4]
        distanceToWalk = io_array[pos, 6]  # speed actually

        d = distanceToWalk
        R = 6371000
        lat1rad = math.radians(lat1)
        lat2rad = math.radians(lat2)
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)

        a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(lat1rad) * math.cos(lat2rad) * math.sin(
            dLon / 2) * math.sin(dLon / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distanceBetweenPoints = R * c

        if (distanceToWalk > distanceBetweenPoints):
            io_array[pos, 0] = lat2
            io_array[pos, 1] = lon2
            io_array[pos, 7] = True
        else:
            io_array[pos, 7] = False
            lat1rad = toRadians(lat1)
            lon1rad = toRadians(lon1)

            lat2rad = toRadians(lat2)
            diffLong = toRadians(lon2 - lon1)

            x = math.sin(diffLong) * math.cos(lat2rad)
            y = math.cos(lat1rad) * math.sin(lat2rad) - (math.sin(lat1rad) * math.cos(lat2rad) * math.cos(diffLong))
            initial_bearing = math.atan2(x, y)
            initial_bearing = toDegrees(initial_bearing)
            brng = toRadians((initial_bearing + 360) % 360)

            lat3rad = math.asin(
                math.sin(lat1rad) * math.cos(d / R) + math.cos(lat1rad) * math.sin(d / R) * math.cos(brng))
            lon3rad = lon1rad + math.atan2(math.sin(brng) * math.sin(d / R) * math.cos(lat1rad),
                                           math.cos(d / R) - math.sin(lat1rad) * math.sin(lat2rad))
            lat3 = toDegrees(lat3rad)
            lon3 = toDegrees(lon3rad)

            io_array[pos, 0] = lat3
            io_array[pos, 1] = lon3

            io_array[pos, 8] = int((lat3 - latMin) // latStep)
            io_array[pos, 9] = int((lon3 - lonMin) // lonStep)


class DroneMovementCuda(MovementStrategy):

    def __init__(self, locationsTable, movableSet, map, mapGrid, strategyType):
        super(DroneMovementCuda, self).__init__(locationsTable, movableSet, map, mapGrid, strategyType)

    def move(self):
        threadsperblock = 256
        blockspergrid = math.ceil(len(self.locationsTable.table) / threadsperblock)
        walking_kernel[blockspergrid, threadsperblock](self.locationsTable.table)

    def getNewRoute(self, movable):
        newRoute = self.map.getRouteBetweenPoints(movable.getLocation(), movable.getLocation())
        return newRoute

    def getRouteTo(self, movable, location: Location):
        newRoute = self.map.getRouteBetweenPoints(movable.getLocation(), movable.getLocation())
        return newRoute

    def onDayChange(self, person: Person):
        '''
        No activity is needed on day change for this Strategy
        :param person:
        :return:
        '''
        return
