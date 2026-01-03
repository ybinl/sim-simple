
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

@dataclass(frozen=True)
class Transform:
    """
    p_to = R @ p_from + t
    """
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)

    def as_matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T