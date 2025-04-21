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
    def __init__(
        self,
        time_between_frames, # the duration between two cohenrent frames (the frames of state and next_state)
        motion_files=glob.glob(
            os.getenv("HOME")
            + "/workspace/kepler_ws/KeplerRobot/datasets/motion_files/*"
        ),
        style_fields=None,
        switch_quaternion=False,
        shuffle=False,
        device="cuda:0",
        frame_begin=None,
        frame_end=None,
        augment_trajectory_num=None,
        augment_frame_num=None,
        init_root_state_fields=None,
        init_dof_state_fields=None,
        expressive_joint_name=None,
        expressive_link_name=None,
        random_start = False,
        amp_obs_history_length=1, # 1 mean no history, only current frame 
        **kwargs,
    ):
        """
        Initialize the AMPLoader.

        Args:
            time_between_frames: Amount of time in seconds between transition.
            data_dir: Directory containing motion data.
            num_preload_transitions: Number of transitions to preload.
            motion_files: List of motion files.
            device: Device to run the model on (e.g., 'cpu' or 'cuda').
        """

        print(f"[Datset Info] Motion files: \n{motion_files}")
        self.time_between_frames = time_between_frames  # the duration between two cohenrent frames
        
        self.switch_quaternion = switch_quaternion
        self.shuffle = shuffle

        self.style_fields = style_fields

        self.random_start = random_start

        self.expressive_joint_name = expressive_joint_name
        self.expressive_link_name = expressive_link_name

        self.amp_obs_history_length=amp_obs_history_length

        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []  # the duration of a frame
        self.trajectory_num_frames = []

        # self.clock=0.0 # a clock count from 0 to 1.
        # self.dt=0.001

        # Load motion data from files
        assert len(motion_files) > 0, "No motion files found."
        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            print(f"[Dataset Info] start to reading {motion_file}")
            with open(motion_file, "r") as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])
                # motion_data = self.reorder_from_pybullet_to_isaac(motion_data)
                self.trajectory_fields = motion_json["Fields"]

                frame_begin = frame_begin if frame_begin is not None else 0
                frame_end = motion_data.shape[0] if frame_end is None else frame_end
                
                print(f"[Dataset Info] Load {motion_file}")
                print(f"[Dataset Info] It has {motion_data.shape[0]} frames.")
                motion_data = motion_data[frame_begin:frame_end, :]
                print(f"[Dataset Info] Select {motion_data.shape[0]} frames from frame_begin is {frame_begin} and frame_end is {frame_end}.")
                #print(f"[Dataset Info] Re-ordered trajectory fields: {self.style_fields}")

                # Normalize and standardize quaternions.
                for f_i in range(motion_data.shape[0]):
                    root_rot = motion_data[f_i, [self.trajectory_fields.index("root_rot_" + key) for key in ["x", "y", "z", "w"]]]
                    root_rot = QuaternionNormalize(root_rot)  # unit(normalize) quaternion (x,y,z,w), to make norm to be 1
                    root_rot = motion_util.standardize_quaternion(root_rot)  # standlize quaternion, make w > 0
                    motion_data[f_i, [self.trajectory_fields.index("root_rot_" + key) for key in ["x", "y", "z", "w"]]] = root_rot

                self.trajectories.append(torch.tensor(motion_data, dtype=torch.float32, device=device))

                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)  # it is time [s], frame len* duration
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

                print(f"[Dataset Info] Select {traj_len} s motion from the trajectory.")
                #print(f"[Dataset Info] The trajectory fields are: {self.trajectory_fields}")

        # Normalize trajectory weights
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(
            self.trajectory_weights
        )
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions if specified
        print(f"[Dataset Info] Preloading data into self.preloaded_s and self.preloaded_s_next")
        if augment_frame_num is None:
            self.augment_frame_num = min(self.trajectory_num_frames)
        else:
            self.augment_frame_num = augment_frame_num
        assert self.augment_frame_num <= min(self.trajectory_num_frames), f"required frame num  {self.augment_frame_num} should less than the loaded trajtectory frame number {self.trajectory_num_frames}"
        if augment_trajectory_num is None:
            self.augment_trajectory_num = self.num_motions
        else:
            self.augment_trajectory_num = augment_trajectory_num


        # Define the field names for the dataset.
        self.root_rot_index = [
            self.style_fields.index("root_rot_" + key) for key in ["x", "y", "z", "w"]
        ]

        traj_idxs = self.weighted_traj_idx_sample_batch(size=self.augment_trajectory_num)
        self.preloaded_s, self.preloaded_s_next = [], []
        for traj_idx in traj_idxs:
            times = self.traj_time_sample_batch(traj_idx, size=self.augment_frame_num + self.amp_obs_history_length)
            s, s_next = [], []
            for frame_time in times:
                s.append(self.sw_quat(self.get_frame_at_time(traj_idx, frame_time)))
                s_next.append(self.sw_quat(self.get_frame_at_time(traj_idx, frame_time + self.time_between_frames)))
            self.preloaded_s.append(torch.vstack(s))
            self.preloaded_s_next.append(torch.vstack(s_next))

        self.preloaded_s = torch.stack(self.preloaded_s, dim=0)
        self.preloaded_s_next = torch.stack(self.preloaded_s_next, dim=0)


        # Select init state and necessary fields
        assert set(init_root_state_fields).issubset(set(self.trajectory_fields)), f"The arguments field {init_root_state_fields} should have same elements with the dataset fields {self.trajectory_fields} in {motion_file}"
        init_state_fields_index = [self.trajectory_fields.index(key) for key in init_root_state_fields]
        init_dof_pos_index = [self.trajectory_fields.index(key) for key in [key+"_dof_pos" for key in init_dof_state_fields]]
        init_dof_vel_index = [self.trajectory_fields.index(key) for key in [key+"_dof_vel" for key in init_dof_state_fields]]
        self.init_states = self.preloaded_s[:,self.amp_obs_history_length, init_state_fields_index + init_dof_pos_index + init_dof_vel_index] # amp_obs_history_length is the start frame
        
        # Select root velocity
        self.expressive_root_vel_index = [self.trajectory_fields.index(key) for key in ["root_vel_x", "root_vel_y","root_ang_vel_z"]]
        self.expressive_root_vel = self.preloaded_s[:,:,self.expressive_root_vel_index][traj_idxs, self.amp_obs_history_length,:]
        
        # Select expressive joint pos
        if expressive_joint_name is not None:
            expressive_joint_pos_fields = [key+"_dof_pos" for key in expressive_joint_name]
            self.expressive_joint_pos_index = [self.trajectory_fields.index(key) for key in expressive_joint_pos_fields] 
            self.expressive_joint_pos = self.preloaded_s[:,:,self.expressive_joint_pos_index][traj_idxs, self.amp_obs_history_length,:]

        # Select expressive link pos
        if expressive_link_name is not None:
            expressive_link_pos_fields = [key1 + key2 for key1 in expressive_link_name for key2 in ["_x", "_y", "_z"]]
            self.expressive_link_pos_index = [self.trajectory_fields.index(key) for key in expressive_link_pos_fields] 
            self.expressive_link_pos = self.preloaded_s[:,:,self.expressive_link_pos_index][traj_idxs, self.amp_obs_history_length,:]

        # Select fields
        #print(f"[Dataset Info] The SELECTED trajectory fields are: {self.style_fields}")
        assert set(self.style_fields).issubset(set(self.trajectory_fields)), f"The arguments field {self.style_fields} should have same elements with the dataset fields {self.trajectory_fields} in {motion_file}"
        reorder_field_index = [self.trajectory_fields.index(key) for key in self.style_fields]
        self.selected_preloaded_s = self.preloaded_s[:,:,reorder_field_index]
        self.selected_preloaded_s_next = self.preloaded_s_next[:,:,reorder_field_index]
    
        print(f"[Dataset Info] self.preloaded_s dim are {len(self.preloaded_s)} and {self.preloaded_s[0].shape} in which augment_frame_num: {self.augment_frame_num} and amp_obs_history_leng: {self.amp_obs_history_length}")
        print(f"[Dataset Info] Trajectory number: {len(self.trajectory_idxs)}")

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size: int):
        """Batch sample traj idxs."""
        if self.shuffle:
            return np.random.choice(
                self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True
            )
        else:
            repeated_array = np.tile(
                self.trajectory_idxs, (size // len(self.trajectory_idxs)) + 1
            )
            return repeated_array[:size]


    def traj_time_sample_batch(self, traj_idx, size):
        """Sample random time for multiple trajectories."""
        subst = (self.time_between_frames + self.trajectory_frame_durations[traj_idx])  # frame period
        if self.shuffle:
            time_samples = (self.trajectory_lens[traj_idx] * np.random.uniform(size=int(size))- subst)
        else:
            if self.random_start:
                #NOTE: data frame num must be large than 2*num_steps_per_env
                random_start_threshold = 1-size/(1+self.trajectories[traj_idx].shape[0])
                start_time = np.random.uniform(high=random_start_threshold, size=1)*self.trajectory_lens[traj_idx]
            else:
                start_time = 0.0
            time_samples = start_time + np.arange(0, size * self.time_between_frames, self.time_between_frames)
            time_samples = np.minimum(time_samples, self.trajectory_lens[traj_idx]-subst)
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        """Spherical linear interpolation."""
        return (1.0 - blend) * val0 + blend * val1


    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = (float(time) / self.trajectory_lens[traj_idx])  # percentage of the time on the trajectory
        n = self.trajectories[traj_idx].shape[0]  # number of traj_idx trajectory
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_begin = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        # return self.slerp(frame_begin, frame_end, blend)
        return self.blend_frame_pose(frame_begin, frame_end, blend)


    def blend_frame_pose(self, frame0, frame1, blend):
        """
        Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between the two frames.

        Returns:
            An interpolation of the two frames.

        """

        if "root_rot" not in self.trajectory_fields:
            blend_fd = self.slerp(frame0, frame1, blend)
        else:
            # Define the field names for the dataset.
            root_rot_index = [self.trajectory_fields.index("root_rot_" + key) for key in ["x", "y", "z", "w"]]
            other_field_index = [
                i for i in range(len(self.trajectory_fields)) if i not in root_rot_index
            ]

            fd0 = frame0[root_rot_index]
            fd1 = frame1[root_rot_index]
            rot_blend_fd = transformations.quaternion_slerp(
                fd0.cpu().numpy(), fd1.cpu().numpy(), blend
            )
            rot_blend_fd = motion_util.standardize_quaternion(rot_blend_fd)

            other_blend_fd = self.slerp(
                frame0[:, other_field_index], frame1[:, other_field_index], blend
            )

            blend_fd = frame0
            blend_fd[:, root_rot_index] = rot_blend_fd
            blend_fd[:, other_field_index] = other_blend_fd

            # blend_fd = torch.cat((other_blend_fd, rot_blend_fd), dim=-1, dtype=torch.float32, device=self.device)
        return blend_fd

    def expert_obs_generator(self):
        """Generates a batch of AMP transitions."""
        if self.shuffle:
            traj_idxs = np.random.choice(self.selected_preloaded_s.shape[0], size=self.selected_preloaded_s.shape[0])  # sample random sequence of frame
        else:
            traj_idxs = np.arange(self.selected_preloaded_s.shape[0])

        for frame_idx in range(self.augment_frame_num):
            s_f = self.selected_preloaded_s[traj_idxs, frame_idx:frame_idx+self.amp_obs_history_length, :]  # differet trajectories and frame times
            self.expressive_root_vel = self.preloaded_s[:,:,self.expressive_root_vel_index][traj_idxs, frame_idx+self.amp_obs_history_length,:]
            if self.expressive_joint_name is not None:
                self.expressive_joint_pos = self.preloaded_s[:,:,self.expressive_joint_pos_index][traj_idxs, frame_idx+self.amp_obs_history_length,:]
            if self.expressive_link_name is not None:
                self.expressive_link_pos = self.preloaded_s[:,:,self.expressive_link_pos_index][traj_idxs, frame_idx+self.amp_obs_history_length,:]
            
            #yield s_f.reshape(s_f.shape[0],-1)
            yield s_f.flatten(1,2)

    def sw_quat(self, frame_data):
        # switch root-rot quaternion from xyzw to wxyz
        if self.switch_quaternion:
            root_rot = frame_data[self.root_rot_index]
            root_rot = torch.cat((root_rot[-1:], root_rot[:-1]))
            frame_data[self.root_rot_index] = root_rot
        return frame_data

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)

        # def get_root_pos(pose):
        #    return pose[AMPLoader.ROOT_POS_START_IDX:AMPLoader.ROOT_POS_END_IDX]

        # def get_root_pos_batch(poses):
        #    return poses[:, AMPLoader.ROOT_POS_START_IDX:AMPLoader.ROOT_POS_END_IDX]

        # def get_root_rot(pose):
        #    return pose[AMPLoader.ROOT_ROT_START_IDX:AMPLoader.ROOT_ROT_END_IDX]

        # def get_root_rot_batch(poses):
        #    return poses[:, AMPLoader.ROOT_ROT_START_IDX:AMPLoader.ROOT_ROT_END_IDX]

        # def get_root_vel(pose):
        #    return pose[AMPLoader.ROOT_LIN_VEL_START_IDX:AMPLoader.ROOT_LIN_VEL_END_IDX]

        # def get_root_vel_batch(pose):
        #    return pose[:,AMPLoader.ROOT_LIN_VEL_START_IDX:AMPLoader.ROOT_LIN_VEL_END_IDX]

        # def get_root_ang_vel(pose):
        #    return pose[AMPLoader.ROOT_ANG_VEL_START_IDX:AMPLoader.ROOT_ANG_VEL_END_IDX]

        # def get_root_ang_vel_batch(pose):
        #    return pose[:,AMPLoader.ROOT_ANG_VEL_START_IDX:AMPLoader.ROOT_ANG_VEL_END_IDX]

        # def get_dof_pos(pose):
        #    return pose[AMPLoader.JOINT_POS_START_IDX:AMPLoader.JOINT_POS_END_IDX]

        # def get_dof_pos_batch(poses):
        #    return poses[:, AMPLoader.JOINT_POS_START_IDX:AMPLoader.JOINT_POS_END_IDX]

        # def get_dof_vel(pose):
        #    return pose[AMPLoader.JOINT_VEL_START_IDX:AMPLoader.JOINT_VEL_END_IDX]

        # def get_dof_vel_batch(poses):
        #    return poses[:, AMPLoader.JOINT_VEL_START_IDX:AMPLoader.JOINT_VEL_END_IDX]

        # def get_tar_toe_pos_local(pose):
        #    return pose[AMPLoader.TAR_TOE_POS_LOCAL_START_IDX:AMPLoader.TAR_TOE_POS_LOCAL_END_IDX]

        # def get_tar_toe_pos_local_batch(poses):
        #    return poses[:, AMPLoader.TAR_TOE_POS_LOCAL_START_IDX:AMPLoader.TAR_TOE_POS_LOCAL_END_IDX]

        # def get_linear_vel(pose):
        #    return pose[AMPLoader.LINEAR_VEL_START_IDX:AMPLoader.LINEAR_VEL_END_IDX]

        # def get_linear_vel_batch(poses):
        #    return poses[:, AMPLoader.LINEAR_VEL_START_IDX:AMPLoader.LINEAR_VEL_END_IDX]

        # def get_angular_vel(pose):
        #    return pose[AMPLoader.ANGULAR_VEL_START_IDX:AMPLoader.ANGULAR_VEL_END_IDX]

        # def get_angular_vel_batch(poses):
        #    return poses[:, AMPLoader.ANGULAR_VEL_START_IDX:AMPLoader.ANGULAR_VEL_END_IDX]

        # def get_tar_toe_vel_local(pose):
        #    return pose[AMPLoader.TAR_TOE_VEL_LOCAL_START_IDX:AMPLoader.TAR_TOE_VEL_LOCAL_END_IDX]

        # def get_tar_toe_vel_local_batch(poses):
        return poses[
            :,
            AMPLoader.TAR_TOE_VEL_LOCAL_START_IDX : AMPLoader.TAR_TOE_VEL_LOCAL_END_IDX,
        ]


if __name__ == "__main__":

    # load dataset for demonstration:
    import glob
    from collections import OrderedDict
    import os
    joint_names = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_joint', 'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_joint', 'torso_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint']

    key_body_names = ["left_ankle_joint","right_ankle_joint"]
    reordered_fields= [
            "root_pos_z",
            "root_rot_x",
            "root_rot_y",
            "root_rot_z",
            "root_rot_w",
            "root_vel_x",
            "root_vel_y",
            "root_vel_z",
            "root_ang_vel_x",
            "root_ang_vel_y",
            "root_ang_vel_z",
        ] +\
        [key + "_dof_pos" for key in joint_names] +\
        [key + "_dof_vel" for key in joint_names] +\
        [k1 + "_" + k2 for k1 in key_body_names for k2 in ["x", "y", "z"]]
    

    amp_data_cfg = OrderedDict(
        time_between_frames=0.02,  # time between two frames
        shuffle=False,
        motion_files=glob.glob(
            os.getenv("HOME")
            + "/workspace/kepler_ws/KeplerRobot/datasets/motion_files/saved_h1_data.txt"
        ),
        style_fields=reordered_fields,
        augment_trajectory_num=100,
        augment_frame_num=10000,
        device="cuda:0",
    )
    amp_data = AMPLoader(**amp_data_cfg)
    amp_expert_generator = amp_data.expert_obs_generator(100, 1)
    for _ in range(4):
        print(amp_expert_generator.__next__().shape)
        print(amp_expert_generator.__next__()[:13])
