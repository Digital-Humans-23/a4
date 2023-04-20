# load necessary modules
import os, sys, glob
sys.path.append(os.getcwd())
import json
import numpy as np
import torch
import time
import heapq
import copy
import pickle
import pathlib

from lisst.utils.vislib import *
from lisst.utils.config_creator import ConfigLoader
from lisst.utils.batch_gen import BatchGeneratorCMUCanonicalized
from lisst.models.baseops import CanonicalCoordinateExtractor
from lisst.models.motion import LISSTGEMP
from lisst.models.body import LISSTPoser
from lisst.models.baseops import RotConverter
from scripts.gen_floor import LISSTGEMPGenFloorOP

# DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE=torch.device('cpu')

class LISSTGEMPGenSpeedOP(LISSTGEMPGenFloorOP):
    def eval_quality(self, motion_primitive):
        """
        scoring function controlling motion synthesis.
        """
        motion_primitive = motion_primitive.view(-1, 1, 31, 9)  # [T, B, J, D], sequence of joint locations and orientations
        floor_height = self.floor_height
        feet_joints = [self.model.body_model.joint_names.index('lfoot'),
                       self.model.body_model.joint_names.index('rfoot'),
                       self.model.body_model.joint_names.index('ltoes'),
                       self.model.body_model.joint_names.index('rtoes')
                       ]
        feet_height = motion_primitive[:, :, feet_joints, 2]
        root_idx = self.model.body_model.joint_names.index('root')
        h = 1 / 40  # time interval between two frames, use it to calculate speed
        speed_goal = self.speed_goal  # [2, ]
        quality_floor = 0
        quality_speed = 0

        """student implementation"""


        quality_dict = {
            'floor': quality_floor,
            'speed': quality_speed,
            'all': quality_floor + quality_speed
        }
        for key in quality_dict:
            quality_dict[key] = quality_dict[key].detach()
        return quality_dict

    def generate(self, data, J_shape, save_dir, n_seqs=1, max_depth=50,
                 n_gens_1frame=16, n_gens_2frame=4, num_expand=4,
                 speed_goal=1.0):
        """motion generation function. Motion seed is from part of the CMU Mocap dataset.

        Args:
            data (_type_): sequence of body joint location and orientations, tensor of shape [T, B, J*(3+6)]
            J_shape (_type_): body bone lengths, tensor of shape [T, B, J]
            n_seqs (_type_): number of sequences to generate
            n_gens (_type_): for each motion seed, how many different sequences to predict
            max_depth (_type_): how many frames to generate. Default is 50
        """

        # use the first frame as the motion seed, repeat the tensor according to desired batch size
        X = data[:1]  # [1, b, J*d]
        # set the floor height as the lowest joint of the motion seed
        self.floor_height = X.view(1, 1, 31, 9)[0, 0, :, 2].amin()
        print('floor height:', self.floor_height)
        self.speed_goal = speed_goal
        J_shape = J_shape[0]  # [1, 10]
        """LISST exporting format"""
        gen_results = {
            'r_locs': [],
            'J_rotmat': [],
            'J_shape': [],
            'J_locs_3d': [],
            'J_locs_2d': []
        }

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for ii in range(n_seqs):
            print('--generate sequence {:03d}'.format(ii))
            r_locs, J_rotmat, J_locs_3d, quality_log = self.search(X, J_shape, max_depth=max_depth,
                                                      n_gens_1frame=n_gens_1frame, n_gens_2frame=n_gens_2frame,
                                                      num_expand=num_expand)
            gen_results['r_locs'].append(r_locs)
            gen_results['J_rotmat'].append(J_rotmat)
            gen_results['J_locs_3d'].append(J_locs_3d)
            gen_results['J_shape'].append(J_shape.detach().cpu().numpy())
            with open(os.path.join(save_dir, 'quality_log{}.pkl'.format(ii)), 'wb') as f:
                pickle.dump(quality_log, f)

        """export"""
        motion_data = {
            'r_locs': np.concatenate(gen_results['r_locs'], axis=1),
            'J_rotmat': np.concatenate(gen_results['J_rotmat'], axis=1),
            'J_shape': np.concatenate(gen_results['J_shape'], axis=0),
            'J_locs_3d': np.concatenate(gen_results['J_locs_3d'], axis=1),
        }
        with open(os.path.join(save_dir, 'motion.pkl'), 'wb') as f:
            pickle.dump(motion_data, f)


if __name__ == '__main__':
    # env setup
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False)
    dtype = torch.float32
    torch.set_default_dtype(dtype)

    # which model versions to use
    shaper_config_name = 'LISST_SHAPER_v0'
    mover_config_name = 'LISST_GEMP_v0'
    poser_config_name = 'LISST_POSER_v0'

    base_dir = str(pathlib.Path(__file__).parent.parent.resolve())
    shaper_config = ConfigLoader('{}/lisst/cfg/{}.yml'.format(base_dir, shaper_config_name))
    shaper_ckpt = '{}/results/lisst/{}/checkpoints/epoch-000.ckp'.format(base_dir, shaper_config_name)
    poser_config = ConfigLoader('{}/lisst/cfg/{}.yml'.format(base_dir, poser_config_name))
    poser_ckpt = '{}/results/lisst/{}/checkpoints/epoch-500.ckp'.format(base_dir, poser_config_name)
    mover_config = ConfigLoader('{}/lisst/cfg/{}.yml'.format(base_dir, mover_config_name))
    mover_ckpt = '{}/results/lisst/{}/checkpoints/epoch-5000.ckp'.format(base_dir, mover_config_name)

    testcfg = {}
    testcfg['gpu_index'] = 0
    testcfg['seed'] = 0
    testcfg['shaper_ckpt_path'] = shaper_ckpt
    testcfg['poser_config'] = poser_config
    testcfg['poser_ckpt_path'] = poser_ckpt
    testcfg['mover_ckpt_path'] = mover_ckpt
    testcfg['body_repr'] = mover_config.modelconfig['body_repr']

    """model and testop"""
    testop = LISSTGEMPGenSpeedOP(mover_config.modelconfig, testcfg)
    testop.build_model()

    # load motion seed data
    with open('data/init_body_1.pkl', 'rb') as f:
        data, J_shape = pickle.load(f)
    data = torch.from_numpy(data).to(device=DEVICE)
    J_shape = torch.from_numpy(J_shape).to(device=DEVICE)

    testop.generate(data, J_shape,
                    save_dir='results/speed_search',
                    speed_goal=1.0,
                    n_seqs=1, max_depth=50,
                    n_gens_1frame=16, n_gens_2frame=4, num_expand=4)
    testop.generate(data, J_shape,
                    save_dir='results/speed_random',
                    speed_goal=1.0,
                    n_seqs=1, max_depth=50,
                    n_gens_1frame=1, n_gens_2frame=1, num_expand=1)