import isaaclab.sim as sim_utils

from isaaclab.assets import AssetBaseCfg
from isaaclab.assets import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


# configuration defines objects in the scene
@configclass
class SceneCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    # overhead lighting
    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/light",
        spawn=sim_utils.DomeLightCfg()
    )
    # biped
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot", # in the stage, this is the path
        spawn=sim_utils.UsdFileCfg(
            usd_path="/source/biped.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.45), # relative position of the root
            joint_pos={ # joint position on initialization/reset
                "left_hip_pitch": 0.0,
                "left_hip_roll": 0.0,
                "left_knee": 0.0,
                "right_hip_pitch": 0.0,
                "right_hip_roll": 0.0,
                "right_knee": 0.0,
            },
        ),
    )