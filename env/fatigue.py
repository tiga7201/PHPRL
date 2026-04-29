import math


def actual_processing_time(base_time: float, fatigue: float, skill: float = 1.0) -> float:
    return base_time * (1.0 + math.log(1.0 + fatigue)) / skill


def update_fatigue_working(fatigue: float, lam: float, delta_u: float = 1.0) -> float:
    return fatigue + (1.0 - fatigue) * (1.0 - math.exp(-lam * delta_u))


def update_fatigue_resting(fatigue: float, mu: float, delta_u: float = 1.0) -> float:
    return fatigue * math.exp(-mu * delta_u)


def physical_to_rates(physical_condition: float) -> tuple[float, float]:
    lam = max(0.05, 0.4 * (1.0 - physical_condition) + 0.05)
    mu = max(0.05, 0.3 * physical_condition + 0.05)
    return lam, mu
