from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg

from configs.scene import SceneCfg

# this configuration defines aspects of the environment
@configclass
class EnvironmentCfg(DirectRLEnvCfg):
    # rl env
    decimation = 2 # how many times per step the actions are applied
    episode_length_s = 5.0 # truncation limit
    action_space = 6 # number of actions (joints, in this case)
    observation_space = 14 # joint pos + joint vel + target direction
    
    # sim
    sim = SimulationCfg(
        dt=0.01, # render delta time
        render_interval=2,
    )
    
    # scene
    scene = SceneCfg(
        num_envs=4000, # number of parallel environments; parallelization is handled on the gpu
        env_spacing=5.0, # physical spacing between environments
        replicate_physics=True,
        clone_in_fabric=True,
    )
    
    # reward coefficients
    vel_alignment = 1.0 # reward magnitude of aligning the velocity vectors (via cosine similarity)
    action_rate = 0.01 # smooths the change in actions to prevent jitter
    joint_vel = 0.01 # smooths the velocity to prioritize slow, intentional movements
    
    # other
    min_termination_height = 0.0 # height that the root of the robot (torso) must be to terminate