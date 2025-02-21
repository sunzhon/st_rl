import os
import glob
import json
import logging

import torch
import numpy as np
from pybullet_utils import transformations

from st_rl.utils import utils
from st_rl.datasets import motion_util
from st_rl.datasets.pose3d import QuaternionNormalize

class AMPLoader:
    # Define constants for data sizes and indices
    ROOT_POS_SIZE = 3
    ROOT_ROT_SIZE = 4
    JOINT_POS_SIZE = 19
    JOINT_VEL_SIZE = 19


    # root pos
    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + ROOT_POS_SIZE

    # root rot
    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROOT_ROT_SIZE

    # root lin vel
    ROOT_LIN_VEL_START_IDX = ROOT_ROT_END_IDX
    ROOT_LIN_VEL_END_IDX = ROOT_LIN_VEL_START_IDX + ROOT_POS_SIZE

    # root ang vel
    ROOT_ANG_VEL_START_IDX = ROOT_LIN_VEL_END_IDX
    ROOT_ANG_VEL_END_IDX = ROOT_ANG_VEL_START_IDX + ROOT_POS_SIZE

    # dof pos 
    JOINT_POS_START_IDX = ROOT_ANG_VEL_END_IDX
    JOINT_POS_END_IDX = JOINT_POS_START_IDX + JOINT_POS_SIZE

    # dof vel 
    JOINT_VEL_START_IDX = JOINT_POS_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE
    
    def __init__(self, time_between_frames, 
                 num_transition=10000,
                 motion_files=glob.glob(os.getenv("HOME")+"/workspace/kepler_ws/KeplerRobot/datasets/motion_files/*"),
                 field_names=None,
                 field_group=None,
                 switch_quaternion=True,
                 shuffle=False,
                 device="cuda:0", **kwargs):
        """
        Initialize the AMPLoader.

        Args:
            time_between_frames: Amount of time in seconds between transition.
            data_dir: Directory containing motion data.
            num_preload_transitions: Number of transitions to preload.
            motion_files: List of motion files.
            device: Device to run the model on (e.g., 'cpu' or 'cuda').
        """
        self.device = device
        self.time_between_frames = time_between_frames # the duration between two cohenrent frames
        self.switch_quaternion=switch_quaternion
        self.shuffle = shuffle
        self.num_transition=num_transition

        self.field_names = field_names
        self.field_group = field_group
        if self.field_group is None:
            self.field_group = ['root_pos', 'root_rot', 'root_vel', 'root_ang_vel', 'dof_pos']
        
        
        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = [] # the duration of a frame
        self.trajectory_num_frames = []

        self.clock=0.0 # a clock count from 0 to 1.
        self.dt=0.001

        # Load motion data from files
        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split('.')[0])
            with open(motion_file, "r") as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])
                #motion_data = self.reorder_from_pybullet_to_isaac(motion_data)
                self.trajectory_fields = motion_json["Fields"]
                print(f"[Dataset Info] Loaded trajectory fields: {self.trajectory_fields}")

                if self.field_names is not None:
                    assert(set(field_names)==set(self.trajectory_fields)), "The arguments field_names of Amploader should have same elements with the dataset fields in {motion_file}"
                    reorder_field_index = [self.trajectory_fields.index(key) for key in field_names]
                    motion_data = motion_data[:,reorder_field_index]
                    print(f"[Dataset Info] Re-ordered trajectory fields: {field_names}")

                # Normalize and standardize quaternions.
                for f_i in range(motion_data.shape[0]):
                    root_rot = AMPLoader.get_root_rot(motion_data[f_i])
                    root_rot = QuaternionNormalize(root_rot) # unit(normalize) quaternion (x,y,z,w), to make norm to be 1
                    root_rot = motion_util.standardize_quaternion(root_rot) # standlize quaternion, make w > 0
                    motion_data[f_i, AMPLoader.ROOT_ROT_START_IDX : AMPLoader.ROOT_ROT_END_IDX] = root_rot

                self.trajectories.append(torch.tensor(motion_data,dtype=torch.float32, device=device))

                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len) # it is time [s], frame len* duration
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")
        
        # Normalize trajectory weights
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions if specified
        print(f'Preloading {self.num_transition} transitions')
        traj_idxs = self.weighted_traj_idx_sample_batch(size=int(max(1, self.num_transition//min(self.trajectory_num_frames)))) 
        s,s_next=[],[]
        for traj_idx in traj_idxs:
            times = self.traj_time_sample_batch(traj_idx, size= max(self.trajectory_num_frames))
            for frame_time in times:
                s.append(self.sw_quat(self.get_frame_at_time(traj_idx, frame_time)))
                s_next.append(self.sw_quat(self.get_frame_at_time(traj_idx, frame_time + self.time_between_frames)))

        self.preloaded_s = torch.vstack(s)
        self.preloaded_s_next = torch.vstack(s_next)
        print(f'Finished preloading')


        print(f"A trajectory shape with selected fields: {self.trajectories[0].shape}")
        print(f" Trajectory number: {len(self.trajectory_idxs)}")



    def reorder_from_pybullet_to_isaac(self, motion_data):
        """
        Convert from PyBullet ordering to Isaac ordering.

        Rearranges leg and joint order from PyBullet [FR, FL, RR, RL] to
        IsaacGym order [FL, FR, RL, RR].

        Args:
            motion_data: Motion data to be reordered.

        Returns:
            Reordered motion data.
        """
        root_pos = AMPLoader.get_root_pos_batch(motion_data)
        root_rot = AMPLoader.get_root_rot_batch(motion_data)

        jp_fr, jp_fl, jp_rr, jp_rl = np.split(AMPLoader.get_joint_pose_batch(motion_data), 4, axis=1)
        joint_pos = np.hstack([jp_fl, jp_fr, jp_rl, jp_rr])
    
        fp_fr, fp_fl, fp_rr, fp_rl = np.split(AMPLoader.get_tar_toe_pos_local_batch(motion_data), 4, axis=1)
        foot_pos = np.hstack([fp_fl, fp_fr, fp_rl, fp_rr])

        lin_vel = AMPLoader.get_linear_vel_batch(motion_data)
        ang_vel = AMPLoader.get_angular_vel_batch(motion_data)

        jv_fr, jv_fl, jv_rr, jv_rl = np.split(AMPLoader.get_joint_vel_batch(motion_data), 4, axis=1)
        joint_vel = np.hstack([jv_fl, jv_fr, jv_rl, jv_rr])

        fv_fr, fv_fl, fv_rr, fv_rl = np.split(AMPLoader.get_tar_toe_vel_local_batch(motion_data), 4, axis=1)
        foot_vel = np.hstack([fv_fl, fv_fr, fv_rl, fv_rr])

        return np.hstack([root_pos, root_rot, joint_pos, foot_pos, lin_vel, ang_vel, joint_vel, foot_vel])

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size:int):
        """Batch sample traj idxs."""
        if self.shuffle:
            return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)
        else:
            repeated_array = np.tile(self.trajectory_idxs, (size // len(self.trajectory_idxs)) + 1)
            return repeated_array[:size]


    
    # def traj_time_sample(self, traj_idx):
    #     """Sample random time for traj."""
    #     subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx] # frame period
    #     return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idx, size):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx] # frame period
        if self.shuffle:
            time_samples = self.trajectory_lens[traj_idx] * np.random.uniform(size=int(size)) - subst
        else:
            time_samples = self.trajectory_lens[traj_idx] * np.linspace(0,1,int(size)) - subst

        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        """Spherical linear interpolation."""
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx] # percentage of the time on the trajectory
        n = self.trajectories[traj_idx].shape[0] # number of traj_idx trajectory
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        #return self.slerp(frame_start, frame_end, blend)
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        #return self.slerp(all_frame_starts, all_frame_ends, blend)
        return self.blend_frame_pose(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        """Returns full frame for the given trajectories at the specified times."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_pos_starts = torch.zeros(len(traj_idxs), AMPLoader.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), AMPLoader.POS_SIZE, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), AMPLoader.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), AMPLoader.ROT_SIZE, device=self.device)
        all_frame_amp_starts = torch.zeros(len(traj_idxs), AMPLoader.JOINT_VEL_END_IDX - AMPLoader.JOINT_POSE_START_IDX, device=self.device)
        all_frame_amp_ends = torch.zeros(len(traj_idxs),  AMPLoader.JOINT_VEL_END_IDX - AMPLoader.JOINT_POSE_START_IDX, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = AMPLoader.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = AMPLoader.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = AMPLoader.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = AMPLoader.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][:, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_VEL_END_IDX]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][:, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_VEL_END_IDX]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = utils.quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([pos_blend, rot_blend, amp_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)


    def blend_frame_pose(self, frame0, frame1, blend):
        """
        Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between the two frames.

        Returns:
            An interpolation of the two frames.


        There are examples:
            root_pos0, root_pos1 = AMPLoader.get_root_pos(frame0), AMPLoader.get_root_pos(frame1)
            root_rot0, root_rot1 = AMPLoader.get_root_rot(frame0), AMPLoader.get_root_rot(frame1)
            joints0, joints1 = AMPLoader.get_joint_pose(frame0), AMPLoader.get_joint_pose(frame1)
            tar_toe_pos_0, tar_toe_pos_1 = AMPLoader.get_tar_toe_pos_local(frame0), AMPLoader.get_tar_toe_pos_local(frame1)
            linear_vel_0, linear_vel_1 = AMPLoader.get_linear_vel(frame0), AMPLoader.get_linear_vel(frame1)
            angular_vel_0, angular_vel_1 = AMPLoader.get_angular_vel(frame0), AMPLoader.get_angular_vel(frame1)
            joint_vel_0, joint_vel_1 = AMPLoader.get_joint_vel(frame0), AMPLoader.get_joint_vel(frame1)

            blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
            blend_root_rot = transformations.quaternion_slerp(root_rot0.cpu().numpy(), root_rot1.cpu().numpy(), blend)
            blend_root_rot = torch.tensor(motion_util.standardize_quaternion(blend_root_rot), dtype=torch.float32, device=self.device)
            blend_joints = self.slerp(joints0, joints1, blend)
            blend_tar_toe_pos = self.slerp(tar_toe_pos_0, tar_toe_pos_1, blend)
            blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
            blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
            blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        """
        blend_data =[]
        for fg in self.field_group:
            #fd0, fd1 = AMPLoader.get_root_pos(frame0), AMPLoader.get_root_pos(frame1)
            fd0, fd1 = getattr(AMPLoader,"get_"+fg)(frame0), getattr(AMPLoader,"get_"+fg)(frame1)
            if fg != "root_rot":
                blend_fd = self.slerp(fd0, fd1, blend)
            else:
                blend_fd = transformations.quaternion_slerp(fd0.cpu().numpy(), fd1.cpu().numpy(), blend)
                blend_fd = torch.tensor(motion_util.standardize_quaternion(blend_fd), dtype=torch.float32, device=self.device)
            blend_data.append(blend_fd)
        return torch.cat(blend_data)


            
    #def feed_forward_generator(self, num_mini_batch, mini_batch_size):
    #    """Generates a batch of AMP transitions."""
    #    for _ in range(num_mini_batch):
    #        if self.preload_transitions:
    #            idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
    #            s = self.preloaded_s[idxs, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_VEL_END_IDX]
    #            s = torch.cat([s, self.preloaded_s[idxs, AMPLoader.ROOT_POS_START_IDX + 2:AMPLoader.ROOT_POS_START_IDX + 3]], dim=-1)
    #            s_next = self.preloaded_s_next[idxs, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_VEL_END_IDX]
    #            s_next = torch.cat([s_next, self.preloaded_s_next[idxs, AMPLoader.ROOT_POS_START_IDX + 2:AMPLoader.ROOT_POS_START_IDX + 3]], dim=-1)
    #        else:
    #            s, s_next = [], []
    #            traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
    #            times = self.traj_time_sample_batch(traj_idxs)
    #            for traj_idx, frame_time in zip(traj_idxs, times):
    #                s.append(self.get_frame_at_time(traj_idx, frame_time))
    #                s_next.append(self.get_frame_at_time(traj_idx, frame_time + self.time_between_frames))
    #            
    #            s = torch.vstack(s)
    #            s_next = torch.vstack(s_next)
    #        yield s, s_next
    


    def expert_obs_generator(self, num_iterations, iteration_size):
        """Generates a batch of AMP transitions."""
        loop=0
        for _ in range(num_iterations):
            s, s_next = [], []
            if self.shuffle:
                traj_idxs = np.random.choice(self.preloaded_s.shape[0], size=iteration_size) # sample random sequence of frame
            else:
                traj_idxs = iteration_size*loop+np.arange(iteration_size)
                loop+=1
                loop%=int(self.num_transition/iteration_size)
            s = self.preloaded_s[traj_idxs,:] # differet trajectories and frame times
            s_next = self.preloaded_s_next[traj_idxs,:]
            yield torch.cat((s, s_next),dim=-1)


    def get_sequence_obs(self, num_envs):
        """Generates a sequence frame at specify time"""
        s, s_next = [], []
        traj_idxs = self.weighted_traj_idx_sample_batch(num_envs)
        times = self.traj_time_sample_batch(traj_idxs)

        for traj_idx, frame_time in zip(traj_idxs, times):

            subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx] # frame period
            frame_time = max(0, self.clock * self.trajectory_lens[traj_idx])- subst

            s.append(self.sw_quat(self.get_frame_at_time(traj_idx, frame_time)))
            s_next.append(self.sw_quat(self.get_frame_at_time(traj_idx, frame_time + self.time_between_frames)))

        self.clock+=self.dt
        self.clock%=1

        s = torch.vstack(s)
        s_next = torch.vstack(s_next)
        return torch.cat((s, s_next),dim=-1)


    def sw_quat(self, frame_data):
        # switch root-rot quaternion from xyzw to wxyz
        if self.switch_quaternion:
            root_rot = AMPLoader.get_root_rot(frame_data)
            root_rot = torch.cat((root_rot[-1:], root_rot[:-1]))
            frame_data[AMPLoader.ROOT_ROT_START_IDX : AMPLoader.ROOT_ROT_END_IDX] = root_rot
        return frame_data

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_root_pos(pose):
        return pose[AMPLoader.ROOT_POS_START_IDX:AMPLoader.ROOT_POS_END_IDX]

    def get_root_pos_batch(poses):
        return poses[:, AMPLoader.ROOT_POS_START_IDX:AMPLoader.ROOT_POS_END_IDX]

    def get_root_rot(pose):
        return pose[AMPLoader.ROOT_ROT_START_IDX:AMPLoader.ROOT_ROT_END_IDX]

    def get_root_rot_batch(poses):
        return poses[:, AMPLoader.ROOT_ROT_START_IDX:AMPLoader.ROOT_ROT_END_IDX]

    def get_root_vel(pose):
        return pose[AMPLoader.ROOT_LIN_VEL_START_IDX:AMPLoader.ROOT_LIN_VEL_END_IDX]

    def get_root_vel_batch(pose):
        return pose[:,AMPLoader.ROOT_LIN_VEL_START_IDX:AMPLoader.ROOT_LIN_VEL_END_IDX]

    def get_root_ang_vel(pose):
        return pose[AMPLoader.ROOT_ANG_VEL_START_IDX:AMPLoader.ROOT_ANG_VEL_END_IDX]

    def get_root_ang_vel_batch(pose):
        return pose[:,AMPLoader.ROOT_ANG_VEL_START_IDX:AMPLoader.ROOT_ANG_VEL_END_IDX]

    def get_dof_pos(pose):
        return pose[AMPLoader.JOINT_POS_START_IDX:AMPLoader.JOINT_POS_END_IDX]

    def get_dof_pos_batch(poses):
        return poses[:, AMPLoader.JOINT_POS_START_IDX:AMPLoader.JOINT_POS_END_IDX]

    def get_dof_vel(pose):
        return pose[AMPLoader.JOINT_VEL_START_IDX:AMPLoader.JOINT_VEL_END_IDX]

    def get_dof_vel_batch(poses):
        return poses[:, AMPLoader.JOINT_VEL_START_IDX:AMPLoader.JOINT_VEL_END_IDX]  











    def get_tar_toe_pos_local(pose):
        return pose[AMPLoader.TAR_TOE_POS_LOCAL_START_IDX:AMPLoader.TAR_TOE_POS_LOCAL_END_IDX]

    def get_tar_toe_pos_local_batch(poses):
        return poses[:, AMPLoader.TAR_TOE_POS_LOCAL_START_IDX:AMPLoader.TAR_TOE_POS_LOCAL_END_IDX]

    def get_linear_vel(pose):
        return pose[AMPLoader.LINEAR_VEL_START_IDX:AMPLoader.LINEAR_VEL_END_IDX]

    def get_linear_vel_batch(poses):
        return poses[:, AMPLoader.LINEAR_VEL_START_IDX:AMPLoader.LINEAR_VEL_END_IDX]

    def get_angular_vel(pose):
        return pose[AMPLoader.ANGULAR_VEL_START_IDX:AMPLoader.ANGULAR_VEL_END_IDX]  

    def get_angular_vel_batch(poses):
        return poses[:, AMPLoader.ANGULAR_VEL_START_IDX:AMPLoader.ANGULAR_VEL_END_IDX]  


    def get_tar_toe_vel_local(pose):
        return pose[AMPLoader.TAR_TOE_VEL_LOCAL_START_IDX:AMPLoader.TAR_TOE_VEL_LOCAL_END_IDX]

    def get_tar_toe_vel_local_batch(poses):
        return poses[:, AMPLoader.TAR_TOE_VEL_LOCAL_START_IDX:AMPLoader.TAR_TOE_VEL_LOCAL_END_IDX]


    
if __name__ == "__main__":

    # load dataset for demonstration:
    import glob
    from collections import OrderedDict
    import os

    joint_names=['left_hip_yaw_joint', 'right_hip_yaw_joint', 'torso_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_hip_pitch_joint', 'right_hip_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_ankle_joint', 'right_ankle_joint', 'left_elbow_joint', 'right_elbow_joint']
    reordered_fields=["root_pos_"+key for key in ["x","y","z"]] +  ["root_rot_"+key for key in ["x","y","z", "w"]] + \
                     ["root_vel_"+key for key in ["x","y","z"]] + ["root_ang_vel_"+key for key in ["x","y","z"]] + joint_names

    amp_data_cfg = OrderedDict(
        time_between_frames=0.1, # time between two frames
        shuffle=True,
        num_transition=1000,
        motion_files = glob.glob(os.getenv("HOME")+"/workspace/kepler_ws/KeplerRobot/datasets/motion_files/saved_data.txt"),
        field_names = reordered_fields,
        device="cuda:0",
    )
    amp_data = AMPLoader(**amp_data_cfg)
    amp_expert_generator = amp_data.expert_obs_generator(1000,1)
    for _ in range(4):
        print(amp_expert_generator.__next__().shape)
        print(amp_expert_generator.__next__()[:13])
        
