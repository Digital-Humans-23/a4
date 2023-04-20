import torch
import numpy as np
import random
import glob
import os, sys
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
import pickle
import cv2
import math
import tqdm
sys.path.append(os.path.join(os.getcwd(), 'lisst'))
sys.path.append(os.getcwd())

from lisst.models.baseops import RotConverter




class BatchGeneratorCMUCanonicalized(object):
    """batch generator for the canonicalized CMU dataset
    
    - The order of joints follow these labels and orders:
        data['joints_names'] = ['root', 'lhipjoint', 'lfemur', 'ltibia', 'lfoot', 
                            'ltoes', 'rhipjoint', 'rfemur', 'rtibia', 'rfoot', 'rtoes', 
                            'lowerback', 'upperback', 'thorax', 'lowerneck', 'upperneck', 
                            'head', 'lclavicle', 'lhumerus', 'lradius', 'lwrist', 'lhand', 'lfingers', 
                            'lthumb', 'rclavicle', 'rhumerus', 'rradius', 
                            'rwrist', 'rhand', 'rfingers', 'rthumb']
    - for each npz file, it contains
         data = {
                'global_transl': [], #[t,3]
                'global_rotmat': [], #[t,3,3]
                'joints_locs':[], # #[t,J,3]
                'joints_rotmat': [], #[t,J,3,3] # note that all joint locations are w.r.t. the skeleton template. They need to be transformed globally.
                'joints_length': [], #[t,J+1], incl. the root
            }

    """
    def __init__(self,
                data_path,
                sample_rate=3,
                body_repr='bone_transform', #['joint_location', 'bone_transform', 'bone_transform_localrot' ]
                read_to_ram=True
                ):
        self.rec_list = list()
        self.index_rec = 0
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.J_locs = []
        self.J_rotmat = []
        # self.J_rotmat_local = []
        self.J_len = []
        self.body_repr = body_repr
        self.read_to_ram = read_to_ram
        self.max_len = 80 if 'x8' in data_path else 10

    def reset(self):
        self.index_rec = 0
        if self.read_to_ram:
            idx_permute = torch.randperm(self.J_locs.shape[1])
            self.J_locs = self.J_locs[:,idx_permute]
            self.J_rotcont = self.J_rotcont[:,idx_permute]
            # self.J_rotcont_local = self.J_rotcont_local[:,idx_permute]
            self.J_len = self.J_len[:,idx_permute]
            

    def has_next_rec(self):
        if self.read_to_ram:
            if self.index_rec < self.J_locs.shape[1]:
                return True
            return False
        else:
            if self.index_rec < len(self.rec_list):
                return True
            return False



      
      
    def get_rec_list(self, 
                    to_gpu=False):
    
        self.rec_list = os.path.join(self.data_path+'.pkl')
        print('[INFO] read all data to RAM from: '+self.data_path)
        all_data = np.load(self.rec_list, allow_pickle=True)
        self.J_locs = all_data['J_locs'] #[b,t,J,3]
        self.J_rotmat = all_data['J_rotmat'] #[b,t, J, 3, 3]
        self.J_len = all_data['J_len'] #[b,t, J]

        if to_gpu:
            self.J_locs = torch.cuda.FloatTensor(self.J_locs).permute(1,0,2,3) #[t,b,J,3]
            self.J_rotmat = torch.cuda.FloatTensor(self.J_rotmat).permute(1,0,2,3,4) #[t,b,J,3,3]
            self.J_len = torch.cuda.FloatTensor(self.J_len).permute(1,0,2) #[t,b,J]
        else:
            raise NotImplementedError('it has to be on gpus.')
        
        ## convert rotation matrix to 6D representations
        nt, nb, nj = self.J_locs.shape[:3]
        self.J_rotcont = self.J_rotmat[:,:,:,:,:-1].contiguous().view(nt,nb,nj,-1)
            


    def next_batch(self, batch_size=64, return_shape=False):
        if self.body_repr == 'joint_location':
            batch_data_ = self.J_locs[:, self.index_rec:self.index_rec+batch_size] #[t,b,J,3]
        elif self.body_repr == 'bone_transform':
            batch_data_locs = self.J_locs[:, self.index_rec:self.index_rec+batch_size]
            batch_data_rotcont = self.J_rotcont[:, self.index_rec:self.index_rec+batch_size]
            batch_data_ = torch.cat([batch_data_locs, batch_data_rotcont],dim=-1)#[t,b,J,3+6]
        
        batch_shape = self.J_len[:, self.index_rec:self.index_rec+batch_size]

        self.index_rec+=batch_size
        nt, nb = batch_data_.shape[:2]
        
        if not return_shape:
            return batch_data_.contiguous().view(nt,nb,-1).detach()
        else:
            return batch_data_.contiguous().view(nt,nb,-1).detach(), batch_shape


    
    def next_sequence(self):
        rec = self.rec_list[self.index_rec]
        with np.load(rec) as data:
            g_transl = np.expand_dims(data['global_transl'], axis=1)[::self.sample_rate]
            g_rotmat = np.expand_dims(data['global_rotmat'], axis=1)[::self.sample_rate]
            j_transl = data['joints_locs'][::self.sample_rate]
            j_rotmat = data['joints_rotmat'][::self.sample_rate]
            transl = np.concatenate([g_transl, j_transl],axis=1)
            rotmat = np.concatenate([g_rotmat, j_rotmat],axis=1)
        
        transl = torch.cuda.FloatTensor(transl) # [t,j,3]
        rotmat = torch.cuda.FloatTensor(rotmat) # [t,j,3,3]
        nt, nj = rotmat.shape[:2]
        rotcont = rotmat[:,:,:,:-1].contiguous().view(nt,nj,-1)

        if self.body_repr == 'joint_location':
            outdata = transl
        elif self.body_repr == 'bone_transform':
            outdata = torch.cat([transl, rotcont],dim=-1)
        
        self.index_rec+=1
        
        return outdata.detach()





if __name__=='__main__':
    pass




from lisst.models.body import LISSTCore, LISSTPoser
class BodyGenerator:
    """this class is to generate a batch of bodies in the world coordinate, in order to study motion control.
    The generated bodies
        - have random shapes and poses,
        - snap to the ground,
        - placed at the world origin,
        - have random facing directions
    """
    def __init__(self, shaper_cfg, poser_cfg, 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
                ):
        self.shaper = LISSTCore(shaper_cfg)
        self.shaper.eval().to(device)
        self.shaper.load(shaper_cfg['shaper_ckpt_path'])
        
        self.poser = LISSTPoser(poser_cfg)
        self.poser.eval().to(device)
        self.poser.load(poser_cfg['poser_ckpt_path'])

        self.device = device

        self.rotmat_pos_x_90deg = torch.tensor([[1.0000000,  0.0000000,  0.0000000],
         [0.0000000,  0.0000000,  -1.0000000],
         [0.0000000, 1.0000000,  0.0000000]], device=device)
        

    def angles_to_rotmats(self, thetas):
        """change an array of rotation angles about Y axis to rotmats

        Args:
            angles (torch.Tensor): angles in radians, [b,]
        
        Returns:
            rotmats (torch.Tensor): rotmats, [b,3,3]
        """
        batch_size = thetas.shape[0]
        r1 = torch.stack([torch.cos(thetas), torch.zeros_like(thetas), torch.sin(thetas)],dim=-1)
        r2 = torch.tensor([0.,1.,0.]).repeat(batch_size, 1)
        r3 = torch.stack([-torch.sin(thetas), torch.zeros_like(thetas), torch.cos(thetas)],dim=-1)
        transf_rotmats = torch.stack([r1, r2, r3],dim=-2)
        return transf_rotmats


    def next_batch(self, 
                   batch_size=64, 
                   snap_to_ground=True, 
                   is_Z_up=True,
                   ):
        """generate a batch of bodies standing on the origin

        Args:
            batch_size (int): how many bodies to generate. Defaults to 64.
            snap_to_ground (bool): If False, the world coordinate is located at the pelvis. Defaults to True.
            is_Z_up (bool): the world coordinate setting. Defaults to True. False is Y-up
        """
        # generate batch of body shapes based on the shape PCA space
        ## only LISST_SHAPER_v0 is supported here.
        zs = torch.zeros(batch_size,self.shaper.num_kpts).to(self.device)
        zs[:,:15] = 15*torch.randn(batch_size, 15) # the first 15 pcs are considered
        bone_lengths = self.shaper.decode(zs) #[b, J]
        
        # generate body poses. 
        ## Due to motion generation purposes, we dont consider additional bones.
        nj_poser = 31 # the poser only learns poses of the 31 joints in the CMU mocap data.
        zp = torch.randn(batch_size, nj_poser, self.poser.z_dim).to(self.device) #[b,J,d]
        J_rotcont = self.poser.decode(zp)
        ## change 6D representation to rotation matrix
        J_rotcont_reshape = J_rotcont.reshape(-1, 6)
        J_rotmat = RotConverter.cont2rotmat(J_rotcont_reshape).reshape(batch_size, 
                                                                    nj_poser, 3,3) #[b, J, 3,3]
        ## apply random orientations about the Y-axis
        thetas = torch.rand(batch_size)*torch.pi*2.
        transf_rotmats = self.angles_to_rotmats(thetas).to(self.device)
        J_rotmat = torch.einsum('bij,bpjk->bpik', transf_rotmats, J_rotmat)
        
        # get joint locations via forward kinematics
        x_root = torch.zeros(batch_size, 1,3).to(self.device)
        J_locs = self.shaper.forward_kinematics(x_root, bone_lengths,J_rotmat)
        
        # snap to the ground
        ## Note that it is still Y-up now!
        if snap_to_ground:
            lfoot_idx = self.shaper.joint_names.index('lfoot')
            rfoot_idx = self.shaper.joint_names.index('rfoot')
            deltaT_y = torch.amin(J_locs[:,[lfoot_idx, rfoot_idx],1],dim=-1).abs()
            x_root[:,0,1] += deltaT_y

        # change the body to Z-up, via rotating about X-axis by -90 deg.
        if is_Z_up:
            x_root = torch.einsum('ij,bpj->bpi', self.rotmat_pos_x_90deg, x_root)
            J_rotmat = torch.einsum('ij,bpjk->bpik', self.rotmat_pos_x_90deg, J_rotmat)
            
        # update the joint locations
        J_locs = self.shaper.forward_kinematics(x_root, bone_lengths,J_rotmat)
        
        return x_root, J_rotmat, bone_lengths, J_locs
        
        

    def next_batch_rl(self, 
                   batch_size=64, 
                   goal_range = 10, # the target waypoint range
                   ):
        """generate a batch of bodies in the world coordinate and the goal.
        This function is used to train the motion policy

        Args:
            batch_size (int): how many bodies to generate. Defaults to 64.
            goal_range (int): the range to randomly place the waypoint on the ground (Z=0)
        
        Returns:
            Yg (torch.Tensor): the seed bone transformation, [t=1,b,J,9]
            bone_length (torch.Tensor): the seed bone lengths, [b,J]
            goal_loc (torch.Tensor): the goal location, [b,3]
        """
        
        # generate the body batch
        x_root, J_rotmat, bone_lengths, J_locs = self.next_batch(batch_size, snap_to_ground=True, is_Z_up=True)
        J_rotcont = RotConverter.rotmat2cont(J_rotmat.reshape(-1,3,3)).contiguous().view(batch_size, -1, 6)
        Yw = torch.cat([J_locs, J_rotcont],dim=-1).detach()
        
        # generate the goal
        goal_loc = goal_range*(2*torch.rand(3)-1) #ending point
        goal_loc[-1] = 0
        goal_loc = goal_loc.repeat(batch_size, 1)
        return Yw[None,...], bone_lengths, goal_loc
            
            
            
    













