import torch.utils.data as data
import os.path
import glob
from shape_inversion.utils.io import read_ply_xyz
# from shape_inversion.utils.pc_transform import swap_axis

class MarsSimDataset(data.Dataset):
    def __init__(self, args):
        self.dataset = args.dataset
        self.dataset_path = args.dataset_path

        if self.dataset in ['MarsSim']:
            pathnames_gt = sorted(glob.glob(self.dataset_path+'/gt_pcl/*'))
            pathnames_in = sorted(glob.glob(self.dataset_path+'/stereo_pcl/*'))
            basenames_gt = [os.path.basename(itm) for itm in pathnames_gt]
            basenames_in = [os.path.basename(itm) for itm in pathnames_in]
            assert len(basenames_gt) == len(basenames_in)

            # gt filenames are of form '100_gt.ply' so cut off last 7 chars.
            self.datum_ids = [int(itm[:-7]) for itm in basenames_gt]

            self.gt_ls    = [read_ply_xyz(str(itm[:-7]) + '_gt.ply')  for itm in pathnames_gt]
            self.input_ls = [read_ply_xyz(str(itm[:-8]) + '_raw.ply') for itm in pathnames_in]
 
            # swap axis as multimodal and ShapeInversion have different canonical pose
            # self.input_ls = [swap_axis(itm, swap_mode='210') for itm in input_ls]
            # self.gt_ls = [swap_axis(itm, swap_mode='210') for itm in gt_ls]
        else:
            raise NotImplementedError
    
    def __getitem__(self, index):
        datum_id = self.datum_ids[index]
        input_pcd = self.input_ls[index]
        gt_pcd = self.gt_ls[index]
        return (gt_pcd, input_pcd, datum_id)
    
    def __len__(self):
        return len(self.input_ls)
