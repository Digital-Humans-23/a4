# load necessary modules
import os, sys, glob
sys.path.append(os.getcwd())
import json
import numpy as np
import torch
import time
import heapq
import copy
import pathlib
import pickle

from lisst.utils.vislib import *
from lisst.utils.config_creator import ConfigLoader
from lisst.utils.batch_gen import BatchGeneratorCMUCanonicalized
from lisst.models.baseops import CanonicalCoordinateExtractor
from lisst.models.motion import LISSTGEMP
from lisst.models.body import LISSTPoser
from lisst.models.baseops import RotConverter

# DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE=torch.device('cpu')

class LISSTGEMPGenOP():

    def __init__(self, moverconfig, testconfig):
        self.dtype = torch.float32
        self.device = DEVICE
        self.moverconfig = moverconfig
        self.testconfig = testconfig

    def build_model(self):
        '''Load pretrained shaper, poser, and mover models'''
        self.model = LISSTGEMP(self.moverconfig)
        self.model.eval()
        self.model.to(self.device)
        self.model.body_model.load(self.testconfig['shaper_ckpt_path'])
        self.model.body_model.eval()
        poser = LISSTPoser(self.testconfig['poser_config'].modelconfig)
        poser.eval()
        poser.to(DEVICE)
        poser.load(self.testconfig['poser_ckpt_path'])
        self.poser = poser
        self.coord_extractor = CanonicalCoordinateExtractor(self.device)
        self.nj = self.model.nj
        self.mover_type = self.moverconfig['predictorcfg']['type']

        ckpt_path = os.path.join(self.testconfig['mover_ckpt_path'])
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print('[INFO] --load pre-trained predictor: {}'.format(ckpt_path))

    def rollout(self, J_seed, J_shape, max_depth=10):
        '''random motion rollout given the initial motion seed, body shape, and maximum rollout depth'''
        _, n_b = J_seed.shape[:2]
        Y_all = []

        '''motion generation'''
        for mp in range(max_depth):
            if mp == 0:
                t_his = 1
                t_pred = self.model.num_frames_primitive - t_his
                ## the first motion primitive is already canonicalized.
                X = J_seed[:t_his]
                if self.mover_type == 'RNN':
                    z = torch.randn(n_b, self.model.predictor.z_dim).to(J_seed.device)
                else:
                    z = torch.randn(t_pred, n_b, self.model.predictor.z_dim).to(J_seed.device)
                Yg_b, Yg = self.model.forward_test(J_shape, z, X)
                Yg = torch.cat([X, Yg])
                Y_all.append(Yg)
            else:
                t_his = 2
                t_pred = self.model.num_frames_primitive - t_his
                '''get new coordinate'''
                J_locs_rec_g = Yg_b.contiguous().view(-1, n_b, 31, 9)[:, :, :, :3].detach()
                pred_jts_g = J_locs_rec_g[-t_his:]
                pelvis_ = pred_jts_g[0, :, 0]
                lhip_ = pred_jts_g[0, :, self.model.body_model.joint_names.index('lhipjoint')]
                rhip_ = pred_jts_g[0, :, self.model.body_model.joint_names.index('rhipjoint')]
                R_curr, T_curr = self.coord_extractor.get_new_coordinate(pelvis_, lhip_, rhip_)
                '''get transformed motion seed X'''
                X = self.model.transform_bone_transf(Yg_b[-t_his:], R_curr, T_curr, to_local=True)
                if self.mover_type == 'RNN':
                    z = torch.randn(n_b, self.model.predictor.z_dim).to(J_seed.device)
                else:
                    z = torch.randn(t_pred, n_b, self.model.predictor.z_dim).to(J_seed.device)
                Y_b, Y = self.model.forward_test(J_shape, z, X)
                Yg = self.model.transform_bone_transf(Y, R_curr, T_curr, to_local=False)
                Yg_b = self.model.transform_bone_transf(Y_b, R_curr, T_curr, to_local=False)

                Y_all.append(Yg)

        Y_all = torch.cat(Y_all).contiguous().view(-1, n_b, self.nj, 9)

        '''decompose bone transform'''
        J_locs_3d = Y_all[:, :, :, :3].detach().cpu().numpy()
        J_rotcont = Y_all[:, :, :, 3:]
        J_rotmat = self.model._cont2rotmat(J_rotcont).detach().cpu().numpy()
        r_locs = J_locs_3d[:, :, :1]

        return r_locs, J_rotmat, J_locs_3d

    def generate(self, data, J_shape, n_seqs, n_gens, max_depth=50):
        """motion generation function. Motion seed is from part of the CMU Mocap dataset.

        Args:
            data (_type_): sequence of body joint location and orientations, tensor of shape [T, B, J*(3+6)]
            J_shape (_type_): body bone lengths, tensor of shape [T, B, J]
            n_seqs (_type_): number of sequences to generate
            n_gens (_type_): for each motion seed, how many different sequences to predict
            max_depth (_type_): how many frames to generate. Default is 50
        """

        # use the first frame as the motion seed, repeat the tensor according to desired batch size
        X = data[:1].repeat(1, n_gens, 1)
        J_shape = J_shape[0].repeat(n_gens, 1)

        """LISST exporting format"""
        gen_results = {
            'r_locs': [],
            'J_rotmat': [],
            'J_shape': [],
            'J_locs_3d': [],
            'J_locs_2d': []
        }

        for ii in range(n_seqs):
            print('--generate sequence {:03d}'.format(ii))
            r_locs, J_rotmat, J_locs_3d = self.rollout(X, J_shape, max_depth=max_depth)
            gen_results['r_locs'].append(r_locs)
            gen_results['J_rotmat'].append(J_rotmat)
            gen_results['J_locs_3d'].append(J_locs_3d)
            gen_results['J_shape'].append(J_shape.detach().cpu().numpy())

        """export"""
        motion_data = {
            'r_locs': np.concatenate(gen_results['r_locs'], axis=1),
            'J_rotmat': np.concatenate(gen_results['J_rotmat'], axis=1),
            'J_shape': np.concatenate(gen_results['J_shape'], axis=0),
            'J_locs_3d': np.concatenate(gen_results['J_locs_3d'], axis=1),
        }
        with open('results/random.pkl', 'wb') as f:
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
    testop = LISSTGEMPGenOP(mover_config.modelconfig, testcfg)
    testop.build_model()

    # load motion seed data
    with open('data/init_body_1.pkl', 'rb') as f:
        data, J_shape = pickle.load(f)
    data = torch.from_numpy(data).to(device=DEVICE)
    J_shape = torch.from_numpy(J_shape).to(device=DEVICE)
    """you can tune the numbers of n_seqs and max_depth"""
    gen_results = testop.generate(data, J_shape, n_seqs=4, max_depth=60,
                                  n_gens=1)