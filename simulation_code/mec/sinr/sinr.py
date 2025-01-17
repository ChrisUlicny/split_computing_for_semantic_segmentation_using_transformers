import math
from simulation_code.src.common.Location import Location
from typing import Any, List, Protocol, Tuple, Union
from simulation_code.src.common.CommonFunctions import CommonFunctions


__com = CommonFunctions()
__m_kHZ = -87.55


class BS(Protocol):
    id: Any
    location: Location
    coverage_radius: float
    tx_power: float
    tx_frequency: float


def calculate_sinr(
    location: Location,
    gNB: BS,
    other_gNBs: List[BS]
) -> float:
    eps = 1
    distance = max(eps, __com.getReal2dDistance(location, gNB.location))
    # fspl = free_space_path_loss(distance, gNB.Tx_frequency)
    cpl = city_path_loss(distance/1000)
    # print('fspl', fspl)
    # print('cpl', cpl)

    S = (w_to_dbm(gNB.tx_power) - cpl)
    # I = w_to_dbm(1e-19)
    I = 0

    for gNB_inter in other_gNBs:
        if gNB_inter.id == gNB.id:
            continue #skip the one we use

        if gNB_inter.tx_frequency != gNB.tx_frequency:
            #skip even if the frequency is not the same - interference is ~0
            continue

        distance = max(
            eps,
            __com.getReal2dDistance(location, gNB_inter.location)
        )

        if distance <= gNB_inter.coverage_radius:
            # fspl = free_space_path_loss(distance, gNB_inter.Tx_frequency)
            cpl = city_path_loss(distance/1000)

            # I += w_to_dbm(gNB_inter.Tx_power) - cpl
            I += dbm_to_w(w_to_dbm(gNB_inter.tx_power) - cpl)

    # k * T  : k - boltzmann constant, T - room temperature in kelvin
    kT = 4.002e-21
    N = w_to_dbm(kT * gNB.bandwidth)
    I = w_to_dbm(I+1e-26)
    return sinr_ltecalc(S, N, I)


def dbm_to_w(inp):
    return math.pow(10,(inp/10))/1000


def w_to_dbm(inp):
    return 10*math.log10((inp*1000))


def sinr_ltecalc(rsrp, noise, interference):
    result = dbm_to_w(rsrp)/(dbm_to_w(noise)+dbm_to_w(interference))
    return w_to_dbm(result/1000)


def free_space_path_loss(distance: float, frequency: float):
    return (20*math.log10(distance) + 20*math.log10(frequency/1000) + __m_kHZ)


def city_path_loss(distance: float):
    #TODO use frequency
    return 128.1+37.6*math.log10(distance)


def calculate_highest_sinr(
    l: Location,
    base_stations: List[BS]
) -> Tuple[float, Union[BS, None]]:
    """
    Calculate the highest SINR value at specified location by specified
    base stations.

    Args:
        l (Location): Location of interest.
        base_stations (List[BaseStationVirtual]): List of BSs to use

    Returns:
        Tuple[float, BaseStationVirtual]: SINR value, respective BS
    """
    best = -1000, None
    for b in base_stations:
        dist = __com.getCuda2dDistance(l, b.location)
        if b.coverage_radius < dist:
            continue

        s = calculate_sinr(l, b, base_stations)

        if s > best[0]:
            best = s, b
    return best