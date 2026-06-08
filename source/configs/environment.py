from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg

from configs.scene import SceneCfg


@configclass
class EnvironmentCfg(DirectRLEnvCfg):
    # rl env
    decimation = 2
    episode_length_s = 5.0
    action_space = 6
    observation_space = 14 # joint pos + joint vel + target direction
    
    # sim
    sim = SimulationCfg(
        dt=0.01,
        render_interval=2,
    )
    
    # scene
    scene = SceneCfg(
        num_envs=9,
        env_spacing=5.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    
    # reward coefficients
    vel_alignment = 1.0
    action_rate = 0.01
    joint_vel = 0.01
    
    # other
    min_termination_height = 0.0