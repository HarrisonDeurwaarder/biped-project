import isaaclab.sim as sim_utils

from isaaclab.assets import AssetBaseCfg
from isaaclab.assets import InteractiveSceneCfg
from isaaclab.utils import configclass


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
    '''
    COMPLETE LATER
    '''