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
from scripts.gen_random import LISSTGEMPGenOP

# DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE=torch.device('cpu')

class MinHeap(object):
    """
    Heap data structure for tree search
    """
    def __init__(self):
        self.data = []

    def push(self, node):
        heapq.heappush(self.data, node)

    def pop(self):
        try:
            node = heapq.heappop(self.data)
        except IndexError as e:
            node=None
        return node

    def clear(self):
        self.data.clear()

    def is_empty(self):
        return True if len(self.data)==0 else False

    def deepcopy(self):
        return copy.deepcopy(self)

    def len(self):
        return len(self.data)

class MPTNodeTorch(object):
    """
    Tree node data structure
    """
    def __init__(self, data, quality, quality_dict):
        '''
        A MPT node contains (data, parent, children list, quality)
        '''
        self.data = data
        self.parent = None
        self.children = []
        self.quality = quality
        self.quality_dict = quality_dict

    def __lt__(self, other):
        '''
        note that this definition is to flip the order in the python heapq (a min heap)
        '''
        return self.quality > other.quality


    def add_child(self, child):
        '''
        child - MPTNode
        '''
        if child.quality != 0:
            child.parent = self
            self.children.append(child)
        # else:
        #     # print('[INFO searchop] cannot add low-quality children. Do nothing.')
        #     pass

    def set_parent(self, parent):
        '''
        parent - MPTNode
        '''
        self.parent = parent
        return True

class LISSTGEMPGenFloorOP(LISSTGEMPGenOP):
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
        quality_floor = 0

        """student implementation"""


        quality_dict = {
            'floor': quality_floor,
            'all': quality_floor
        }
        for key in quality_dict:
            quality_dict[key] = quality_dict[key].detach()
        return quality_dict

    def search(self, J_seed, J_shape, max_depth=10,
               n_gens_1frame=16, n_gens_2frame=4, num_expand=4):
        '''root node generation'''
        mp_heap_root = MinHeap()
        t_his = 1
        t_pred = self.model.num_frames_primitive - t_his
        ## the first motion primitive is already canonicalized.
        X = J_seed[:t_his]
        if self.mover_type == 'RNN':
            z = torch.randn(n_gens_1frame, self.model.predictor.z_dim).to(J_seed.device)
        else:
            z = torch.randn(t_pred, n_gens_1frame, self.model.predictor.z_dim).to(J_seed.device)
        Yg_b, Yg = self.model.forward_test(J_shape.repeat(n_gens_1frame, 1), z, X.repeat(1, n_gens_1frame, 1))  # [t, b, J*d]
        for gen_idx in range(n_gens_1frame):
            motion_primitive = torch.cat([X, Yg_b[:, [gen_idx], :]], dim=0)  # [t, 1, J*d]
            quality_dict = self.eval_quality(motion_primitive)
            quality = quality_dict['all']
            root_node = MPTNodeTorch(data=motion_primitive.detach(), quality=quality, quality_dict=quality_dict)
            mp_heap_root.push(root_node)

        """2 frame"""
        mp_heap_curr = MinHeap()
        mp_heap_prev = mp_heap_root
        t_his = 2
        t_pred = self.model.num_frames_primitive - t_his
        for depth in range(1, max_depth):
            print('[INFO] at level {}'.format(depth))
            idx_node = 0
            while (not mp_heap_prev.is_empty()) and (idx_node < num_expand):
                mp_prev = mp_heap_prev.pop()
                idx_node += 1

                '''get new coordinate'''
                Yg_b = mp_prev.data.detach()
                J_locs_rec_g = Yg_b.contiguous().view(-1, 1, 31, 9)[:, :, :, :3].detach()  # [t, 1, J, 9]
                pred_jts_g = J_locs_rec_g[-t_his:]
                pelvis_ = pred_jts_g[0, :, 0]
                lhip_ = pred_jts_g[0, :, self.model.body_model.joint_names.index('lhipjoint')]
                rhip_ = pred_jts_g[0, :, self.model.body_model.joint_names.index('rhipjoint')]
                R_curr, T_curr = self.coord_extractor.get_new_coordinate(pelvis_, lhip_, rhip_)
                '''get transformed motion seed X'''
                X = self.model.transform_bone_transf(Yg_b[-t_his:], R_curr, T_curr, to_local=True)
                if self.mover_type == 'RNN':
                    z = torch.randn(n_gens_2frame, self.model.predictor.z_dim).to(J_seed.device)
                else:
                    z = torch.randn(t_pred, n_gens_2frame, self.model.predictor.z_dim).to(J_seed.device)
                Y_b, Y = self.model.forward_test(J_shape.repeat(n_gens_2frame, 1), z, X.repeat(1, n_gens_2frame, 1))
                Yg_b = self.model.transform_bone_transf(Y_b, R_curr, T_curr, to_local=False)
                for gen_idx in range(n_gens_2frame):
                    motion_primitive = Yg_b[:, [gen_idx], :]  # [t, 1, J*d]
                    quality_dict = self.eval_quality(motion_primitive)
                    quality = quality_dict['all']
                    node = MPTNodeTorch(data=motion_primitive.detach(), quality=quality, quality_dict=quality_dict)
                    node.set_parent(mp_prev)
                    mp_heap_curr.push(node)

            print('quality:', mp_heap_curr.data[0].quality_dict)

            mp_heap_prev.clear()
            mp_heap_prev = copy.deepcopy(mp_heap_curr)
            mp_heap_curr.clear()

        """iterate tree to collect trajectory"""
        gen_results = []
        quality_dict_list = []
        mp_leaf = mp_heap_prev.pop()
        gen_results.append(mp_leaf.data)
        quality_dict_list.append(mp_leaf.quality_dict)
        while mp_leaf.parent is not None:
            gen_results.append(mp_leaf.parent.data)
            quality_dict_list.append(mp_leaf.parent.quality_dict)
            mp_leaf = mp_leaf.parent
        gen_results.reverse()
        quality_dict_list.reverse()
        quality_log = {}
        for key in quality_dict_list[0].keys():
            quality_log[key] = torch.stack([quality_dict[key] for quality_dict in quality_dict_list]).detach().cpu().numpy()
        print(quality_log)

        '''decompose bone transform'''
        Y_all = torch.cat(gen_results, dim=0).contiguous().view(-1, 1, self.nj, 9)
        J_locs_3d = Y_all[:, :, :, :3].detach().cpu().numpy()
        J_rotcont = Y_all[:, :, :, 3:]
        J_rotmat = self.model._cont2rotmat(J_rotcont).detach().cpu().numpy()
        r_locs = J_locs_3d[:, :, :1]

        return r_locs, J_rotmat, J_locs_3d, quality_log

    def generate(self, data, J_shape, save_dir, n_seqs=1, max_depth=50,
                 n_gens_1frame=16, n_gens_2frame=4, num_expand=4):
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
        J_shape = J_shape[0]
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
    testop = LISSTGEMPGenFloorOP(mover_config.modelconfig, testcfg)
    testop.build_model()

    # load motion seed data
    with open('data/init_body_0.pkl', 'rb') as f:
        data, J_shape = pickle.load(f)
    data = torch.from_numpy(data).to(device=DEVICE)
    J_shape = torch.from_numpy(J_shape).to(device=DEVICE)

    testop.generate(data, J_shape,
                    save_dir='results/floor_search',
                    n_seqs=1, max_depth=50,
                    n_gens_1frame=16, n_gens_2frame=4, num_expand=4)
    testop.generate(data, J_shape,
                    save_dir='results/floor_random',
                    n_seqs=1, max_depth=50,
                    n_gens_1frame=1, n_gens_2frame=1, num_expand=1)