from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class DroneState:
    drone_id: str
    x: float
    y: float
    z: float
    battery: float
    status: str = "active"

    def to_dict(self) -> Dict:
        return asdict(self)