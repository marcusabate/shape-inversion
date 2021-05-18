import ruamel.yaml as yaml

class Arguments:
    def __init__(self, filename, stage='pretrain'):
        data = None
        with open(filename, 'r') as file:
            data = yaml.safe_load(file)
        
        if data is None:
            raise FileNotFoundError

        if stage == 'eval_completion':
            self.add_eval_completion_args(data)
        else:
            self.add_common_args(data)
            if stage == 'pretrain':
                self.add_pretrain_args(data)
            elif stage == 'inversion':
                self.add_inversion_args(data)
            elif stage == 'eval_treegan':
                self.add_eval_treegan_args(data)

    def add_common_args(self, data):
        ### data related
        self.class_choice = data['class_choice']
        self.dataset = data['dataset']
        self.dataset_path = data['dataset_path']
        self.split = data['split']
        
        ### TreeGAN architecture related
        self.DEGREE = data['DEGREE']
        self.G_FEAT = data['G_FEAT']
        self.D_FEAT = data['D_FEAT']
        self.support = data['support']
        self.loop_non_linear = data['loop_non_linear']
        
        ### others
        self.FPD_path = data['FPD_path']
        self.gpu = data['gpu']
        self.ckpt_load = data['ckpt_load']
        if self.ckpt_load == '':
            self.ckpt_load = None
    
    def add_pretrain_args(self, data):
        ### general training related
        self.batch_size = data['batch_size']
        self.epochs = data['epochs']
        self.lr = data['lr'] 
        self.lambdaGP = data['lambdaGP']
        self.D_iter = data['D_iter']
        self.w_train_ls = data['w_train_ls']

        ### uniform losses related
        # PatchVariance
        self.knn_loss = data['knn_loss']
        self.knn_k = data['knn_k']
        self.knn_n_seeds = data['knn_n_seeds']
        self.knn_scalar = data['knn_scalar']
        # PU-Net's uniform loss
        self.krepul_loss = data['krepul_loss']
        self.krepul_k = data['krepul_k']
        self.krepul_n_seeds = data['krepul_n_seeds']
        self.krepul_scalar = data['krepul_scalar']
        self.krepul_h = data['krepul_h']
        # MSN's Expansion-Penalty
        self.expansion_penality = data['expansion_penality']
        self.expan_primitive_size = data['expan_primitive_size']
        self.expan_alpha = data['expan_alpha']
        self.expan_scalar = data['expan_scalar']

        ### ohters
        self.ckpt_path = data['ckpt_path']
        self.ckpt_save = data['ckpt_save']
        self.eval_every_n_epoch = data['eval_every_n_epoch']
        self.save_every_n_epoch = data['save_every_n_epoch']
     
    def add_inversion_args(self, data):
        ### loss related
        self.w_nll = data['w_nll']
        self.p2f_chamfer = data['p2f_chamfer']
        self.p2f_feature = data['p2f_feature']
        self.w_D_loss = data['w_D_loss']
        self.directed_hausdorff = data['directed_hausdorff']
        self.w_directed_hausdorff_loss = data['w_directed_hausdorff_loss']  

        ### mask related
        self.mask_type = data['mask_type']
        self.k_mask_k = data['k_mask_k']
        self.voxel_bins = data['voxel_bins']
        self.surrounding = data['surrounding']
        self.tau_mask_dist = data['tau_mask_dist']
        self.hole_radius = data['hole_radius']
        self.hole_k = data['hole_k']
        self.hole_n = data['hole_n']
        self.masking_option = data['masking_option']
        
        ### inversion mode related
        self.inversion_mode = data['inversion_mode']
        ### diversity
        self.n_z_candidates = data['n_z_candidates']
        self.n_outputs = data['n_outputs']

        ### other GAN inversion related
        self.random_G = data['random_G']
        self.select_num = data['select_num']
        self.sample_std = data['sample_std']
        self.iterations = data['iterations']
        self.G_lrs = data['G_lrs']
        self.z_lrs = data['z_lrs']
        self.warm_up = data['warm_up']
        self.update_G_stages = data['update_G_stages']
        self.progressive_finetune = data['progressive_finetune']
        self.init_by_p2f_chamfer = data['init_by_p2f_chamfer']
        self.early_stopping = data['early_stopping']
        self.stop_cd = data['stop_cd']
        self.target_downsample_method = data['target_downsample_method']
        self.target_downsample_size = data['target_downsample_size']
        
        ### others
        self.save_inversion_path = data['save_inversion_path']
        self.dist = data['dist']
        self.port = data['port']
        self.visualize = data['visualize']

    def add_eval_completion_args(self, data):
        self.eval_with_GT = data['eval_with_GT']
        self.saved_results_path = data['saved_results_path']

    def add_eval_treegan_args(self, data):
        self.eval_treegan_mode = data['eval_treegan_mode']
        self.save_sample_path = data['save_sample_path']
        self.model_pathname = data['model_pathname']
        self.batch_size = data['batch_size']
        self.n_samples = data['n_samples']
