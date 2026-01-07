from enum import Enum, auto
import numpy as np
from src.planning.path import (
    make_straight_lane_path,
    make_lane_change_path,
    make_turn_path
)

class DriveState(Enum):
    KEEP_LANE   = auto()
    LANE_CHANGE = auto()
    TURN        = auto()


class Planner:
    """
    Rule-based finite state machine planner
    """

    def __init__(self):
        self.state = DriveState.KEEP_LANE
        self.prev_state = None

    def update(self, ego_state, perception):
        """
        ego_state: VehicleState
        perception: dict (idealized perception result)
        """
        self.prev_state = self.state
        
        obj_result = perception.get("object_box_result", None)

        # --- State transition logic ---
        if self.state == DriveState.KEEP_LANE:
            if perception.get("need_lane_change", False):
                self.state = DriveState.LANE_CHANGE
            elif perception.get("approaching_intersection", False):
                self.state = DriveState.TURN

        elif self.state == DriveState.LANE_CHANGE:
            if perception.get("lane_change_done", False):
                self.state = DriveState.KEEP_LANE

        elif self.state == DriveState.TURN:
            if perception.get("turn_done", False):
                self.state = DriveState.KEEP_LANE

        # --- Generate reference path for current state ---
        ref_path = self._make_ref_path(ego_state, obj_result)

        return self.state, ref_path

    def _make_ref_path(self, ego_state, obj_result):
        x0 = ego_state.x
        
        if obj_result is None:

            if self.state == DriveState.KEEP_LANE:
                return make_straight_lane_path(
                    x_start=x0,
                    x_end=x0 + 60.0,
                    y=0.0
                )

            elif self.state == DriveState.LANE_CHANGE:
                return make_lane_change_path(
                    x_start=x0,
                    x_end=x0 + 60.0,
                    y0=0.0,
                    y1=3.5,
                    x_change_start=x0 + 10.0,
                    x_change_end=x0 + 30.0
                )

            elif self.state == DriveState.TURN:
                return make_turn_path(
                    center=(x0 + 15.0, 0.0),
                    radius=10.0,
                    angle_start=0.0,
                    angle_end=np.pi/2
                )
        else:
            # line 그리는 함수를 새로 짬
            return make_detour_path(obj_result)
        
def make_detour_path(obj_result :np.ndarray):
    # obj_result input [0] = x좌표, [1] = y좌표, [2] = z좌표(z=0), dim=(1,3)
    p1,p2,p3,p4 = (0,0), (1,0), (obj_result[0], obj_result[1] + 1.75), (obj_result[0] + 10,0)
    points = [p1,p2,p3,p4]
    points = np.array(points)
    
    x =  points[:,0]
    y = points[:,1]
    
    coeff = np.polyfit(x,y, deg=3)
    xs = np.linspace(0, +100, 100) # x좌표 100개
    ys = np.polyval(coeff, xs) # y좌표 100개
    return list(zip(xs, ys))
    
