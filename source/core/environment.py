import torch
from isaaclab.envs import DirectRLEnv

from source.configs.environment import EnvironmentCfg


class Environment(DirectRLEnv):
    def __init__(self):
        super().__init__(EnvironmentCfg(), None)
        # extract entities
        self.robot = self.scene["robot"]
        # define rewards for logging
        self._episode_rewards = {
            key: torch.zeros((self.num_envs), device=self.sim.device)
            for key in ["vel_alignment", "action_rate", "joint_vel"]
        }
        # define empty command tensor
        self.target_direction = torch.zeros((self.num_envs, 2))
        # define empty action tensors
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space))
        self.past_actions = torch.zeros((self.num_envs, self.cfg.action_space))
        
        
    def _pre_physics_step(self, actions: torch.Tensor):
        """
        preprocess actions prior to applying in physics loop
        """
        self.past_actions = self.actions[:]
        self.actions = actions[:]
    
    
    def _apply_action(self):
        """
        apply actions in physics loop
        """
        # write raw joint efforts
        self.robot.set_joint_effort_target(self.actions)
        # update robot buffers
        self.robot.update(self.physics_dt)
        self.robot.write_data_to_sim()
        
        
    def _get_observations(self):
        """
        fetch observations (joint pos, joint vel)
        """
        # get joint localization
        # this includes positions and velocities
        # these are necessary so the robot knows where its joints are and where they are going
        # for example, the joint positions in the ground contact phase are different than the swing phase, and the policy must know how to adjust torque accordingly
        joint_pos = self.robot.data.joint_pos[:]
        joint_vel = self.robot.data.joint_vel[:]
        # concatenate
        return torch.cat((
            joint_pos, joint_vel
        ), dim=1)
        
        
    def _get_rewards(self):
        """
        fetch rewards/penalties (+projected_velocity, -action_rates, -joint_accel)
        """
        # compute velocity term
        robot_vel = self.robot.data.root_lin_vel_w[:]
        # compute cosine similarity
        # this rewards the robot root (the body) for traveling in the direction of the velocity
        vel_sim_score = (self.target_direction * robot_vel).sum(dim=1) / (torch.linalg.norm(robot_vel, dim=1), torch.linalg.norm(self.target_direction, dim=1))
        
        # fetch mean action rate
        # smooths the output to prevent unsafe jitter
        action_rate = (self.actions - self.past_actions).abs().mean(dim=1)
        
        # fetch mean joint accel
        # reward slow, smooth movements
        # fast movements are generally less effective
        joint_vel = self.robot.data.joint_vel[:]
        
        rewards = {
            "vel_alignment": self.cfg.vel_alignment * vel_sim_score,
            "action_rate": self.cfg.action_rate * action_rate,
            "joint_vel": self.cfg.joint_vel * joint_vel,
        }
        # log rewards
        for key, reward in rewards.items():
            self._episode_rewards[key] += reward
        # cumulative reward
        return torch.sum(
            torch.stack(tuple(rewards.values()), dim=1),
            dim=1,
        )
        
        
    def _get_dones(self):
        """
        fetch terminations/truncations
        """
        # get truncations
        # truncated environments are where the maximum time duration has been exceeded
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # get terminations
        # when the root of the robot (the body) is below the specified region, 
        term = self.robot.data.root_lin_vel_w[:, 2] <= self.cfg.min_termination_height
        
        return term, time_out
    
    
    def _reset_idx(self, env_ids):
        """
        reset specified environment indicies
        """
        super().reset_idx(env_ids)
        # resolve ids
        if env_ids is None:
            env_ids = self.robot._ALL_INDICIES
            
        # reset robot state
        # when each environment is reset (i.e. they are truncated/terminated), we want the robot joints to return to their defined native state
        default_joint_pos = self.robot.data.default_joint_pos.clone()
        default_joint_vel = self.robot.data.default_joint_vel.clone()
        # write to the robot buffers
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
        self.robot.write_data_to_sim()
        
        self.robot.reset()
        self.robot.update(self.physics_dt,)