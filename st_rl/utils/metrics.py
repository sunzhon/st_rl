# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure a :class:`ManagerBasedRLEnv` or :class:`DirectRlEnv` instance to RSL-RL vectorized environment.

The following example shows how to wrap an environment for RSL-RL:

.. code-block:: python

    from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

    env = RslRlVecEnvWrapper(env)

"""

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from legged_robots.tasks.utils.wrappers.st_rl import StRlVecEnvWrapper #, StRlMetricEnvWrapper
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat
from isaaclab.utils.math import euler_xyz_from_quat as get_euler_xyz

from .logger import Logger

class StRlMetricEnvWrapper(StRlVecEnvWrapper):
    """Wrapper class to add metrics and logging capabilities to an RL environment."""

    def __init__(self, env, enable_logger=False,log_dir=None):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            enable_logger: Whether to enable logging.
        """
        super().__init__(env)
        self.__dict__.update(env.__dict__)
        asset_cfg = SceneEntityCfg("robot")
        asset = self.unwrapped.scene[asset_cfg.name]
        self.data = asset.data
        if "feet_contact_forces" in self.unwrapped.scene.keys():
            self.contact_sensor: ContactSensor = self.unwrapped.scene["feet_contact_forces"]

        self.joint_names = self.unwrapped.scene["robot"].joint_names
        self.default_body_mass = torch.sum(self.data.default_mass, dim=1).to(self.device)
        self.payloads = torch.zeros_like(self.default_body_mass)
        self.g = 9.8  # Gravitational constant
        self.start_log = False
        self.finish_log = False

        if enable_logger:
            self.logger = Logger(env.unwrapped.sim.cfg.dt, log_dir=log_dir)
        else:
            self.logger = None

    @property
    def base_lin_vel(self) -> torch.Tensor:
        """Get the base linear velocity in the robot's frame."""
        vel_yaw = quat_rotate_inverse(yaw_quat(self.data.root_link_quat_w), self.data.root_com_lin_vel_w[:, :3])
        return vel_yaw
    
    @property
    def base_ang_vel(self) -> torch.Tensor:
        """Get the base angular velocity in the robot's frame."""
        ang_vel = quat_rotate_inverse(yaw_quat(self.data.root_link_quat_w), self.data.root_com_ang_vel_w[:, :3])
        return ang_vel


    @property
    def projected_gravity(self) -> torch.Tensor:
        """Get the gravity vector projected in the robot's base frame."""
        return self.data.projected_gravity_b


    @property
    def commands(self) -> torch.Tensor:
        """Get the commanded velocities."""
        command_name = "base_velocity"
        return self.env.unwrapped.command_manager.get_command(command_name)[:, :]

    @property
    def dof_pos(self) -> torch.Tensor:
        """Get the joint positions."""
        return self.data.joint_pos

    @property
    def dof_vel(self) -> torch.Tensor:
        """Get the joint velocities."""
        return self.data.joint_vel

    @property
    def dof_torque(self) -> torch.Tensor:
        """Get the applied joint torques."""
        return self.data.applied_torque

    @property
    def power_consumption(self) -> torch.Tensor:
        """Calculate the power consumption of the robot."""
        return torch.sum(torch.abs(self.dof_torque) * torch.abs(self.dof_vel), dim=1)

    @property
    def CoT(self) -> torch.Tensor:
        """Calculate the Cost of Transport (CoT)."""
        P = self.power_consumption
        m = self.default_body_mass + self.payloads
        v = torch.norm(self.base_lin_vel[:, 0:2], dim=1)
        v = torch.where(v == 0, torch.ones_like(v), v)  # Avoid division by zero
        return P / (m * self.g * v)

    @property
    def lin_vel_rmsd(self) -> torch.Tensor:
        """Calculate the RMSD of linear velocity."""
        return ((self.base_lin_vel[:, 0] - self.commands[:, 0]) ** 2).cpu() ** 0.5

    def start_logging(self):
        print(f"[Logging] Start logging data")
        self.start_log=True
        self.finish_log=False


    def finish_logging(self):
        print(f"[Logging] Finish logging data")
        self.finish_log=True


    def log_update(self, robot_index=0):
        """Log the states of the robot.

        Args:
            robot_index: The index of the robot to log.
        """

        if self.logger is None:
            return

        states = {
            'command_x': self.commands[robot_index, 0].item(),
            'command_y': self.commands[robot_index, 1].item(),
            'command_yaw': self.commands[robot_index, 2].item(),
            'base_lin_vel_x': self.base_lin_vel[robot_index, 0].item(),
            'base_lin_vel_y': self.base_lin_vel[robot_index, 1].item(),
            'base_lin_vel_z': self.base_lin_vel[robot_index, 2].item(),
            'base_ang_vel_x': self.base_ang_vel[robot_index, 0].item(),
            'base_ang_vel_y': self.base_ang_vel[robot_index, 1].item(),
            'base_ang_vel_z': self.base_ang_vel[robot_index, 2].item(),
            'base_pro_gravity_x': self.projected_gravity[robot_index, 0].item(),
            'base_pro_gravity_y': self.projected_gravity[robot_index, 1].item(),
            'base_pro_gravity_z': self.projected_gravity[robot_index, 2].item(),
            'base_roll': get_euler_xyz(self.data.root_link_quat_w)[0][robot_index].item(),
            'base_pitch': get_euler_xyz(self.data.root_link_quat_w)[1][robot_index].item(),
            'base_yaw': get_euler_xyz(self.data.root_link_quat_w)[2][robot_index].item(),
        }

        # Log joint-specific states
        for idx in range(self.num_actions):
            states[f"{self.joint_names[idx]}_action"] = self.unwrapped.action_manager._action[robot_index,idx].item()
            states[f'{self.joint_names[idx]}_pos_target'] = self.data.joint_pos_target[robot_index, idx].item()
            states[f'{self.joint_names[idx]}_pos'] = self.dof_pos[robot_index, idx].item()
            states[f'{self.joint_names[idx]}_vel'] = self.dof_vel[robot_index, idx].item()
            states[f'{self.joint_names[idx]}_tau'] = self.dof_torque[robot_index, idx].item()

        
        # Log contact forces
        grf = self.contact_sensor.data.net_forces_w_history.mean(dim=1)
        for idx in range(grf.shape[1]):
            states[f'grf_x_{idx}'] = grf[:,idx,0].item()
            states[f'grf_y_{idx}'] = grf[:,idx,1].item()
            states[f'grf_z_{idx}'] = grf[:,idx,2].item()

        self.logger.update_states(states)

    def log_save(self):
        """Save the logger."""
        if self.logger is not None:
            self.logger.save()
            self.logger.reset()
            self.start_log = False
            
