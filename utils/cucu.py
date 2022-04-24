class space_carving_rotation_2d():
    def __init__(self, dataset_path, gt_mode=False, rotation_steps=0, total_col_positions=180):
        # bias of n steps for simulating rotation of the object
        self.rotation_steps = rotation_steps
        # number of posible positions around the circle
        self.total_positions = total_col_positions
        # get all .png file names from folder path
        self.masks_files = sorted(
            glob.glob(os.path.join(dataset_path, 'masks', '*.png')))
        self.extrinsics = self.load_extrinsics(
            os.path.join(dataset_path, 'extrinsics'))
        #self.bbox = json.load(open(os.path.join(dataset_path, 'bbox.json')))
        self.bbox = json.load(open(os.path.join(
            dataset_path, '/home/pico/uni/romi/scanner_cube/bbox_min_max.json')))
        self.camera_model = json.load(
            open(os.path.join(dataset_path, 'camera_model.json')))
        self.intrinsics = self.camera_model['params'][0:4]

        params = json.load(open(os.path.join(dataset_path, 'params.json')))

        self.gt_mode = gt_mode

        if self.gt_mode is True:
            self.gt = np.load(os.path.join(
                dataset_path, 'volumes', 'vol_180.npy'))
            self.gt_solid_mask = np.where(self.gt == 1, True, False)
            self.gt_n_solid_voxels = np.count_nonzero(self.gt_solid_mask)

        self.n_dilation = params["sc"]["n_dilation"]
        self.voxel_size = params['sc']['voxel_size']

        self.set_sc(self.bbox)

    def reset(self):
        del(self.sc)
        self.set_sc(self.bbox)

    def load_extrinsics(self, path):
        ext = []
        ext_files = glob.glob(os.path.join(path, '*.json'))
        assert len(ext_files) != 0, "json list is empty."
        for i in sorted(ext_files):
            ext.append(json.load(open(i)))
        return ext

    def load_mask(self, idx):
        img = cv2.imread(self.masks_files[idx], cv2.IMREAD_GRAYSCALE)
        return img

    def set_sc(self, bbox):
        x_min, x_max = bbox['x']
        y_min, y_max = bbox['y']
        z_min, z_max = bbox['z']

        nx = int((x_max - x_min) / self.voxel_size) + 1
        ny = int((y_max - y_min) / self.voxel_size) + 1
        nz = int((z_max - z_min) / self.voxel_size) + 1

        self.origin = np.array([x_min, y_min, z_min])
        self.sc = cl.Backprojection(
            [nx, ny, nz], [x_min, y_min, z_min], self.voxel_size)
        self.volume = self.sc.values()

    def carve(self, row, col):
        if self.rotation_steps != 0:
            col = self.calculate_col_position(col, -self.rotation_steps)

        idx = (self.total_positions * row) + col

        im = self.load_mask(idx)
        self.space_carve(im, self.extrinsics[idx])

        if self.rotation_steps == 0:
            self.volume = self.sc.values()
        else:
            self.volume = rotate(self.sc.values(),
                angle=self.rotation_steps*(360//self.total_positions), reshape=False)
            
        #rotate according with position of camera plane xy is always pointing where camera sees
        #self.volume = rotate(self.sc.values(),angle=-idx*(360//self.total_positions),reshape=False)

    def space_carve(self, mask, rt):
        # mask = im.copy() #get_mask(im)
        rot = sum(rt['R'], [])
        tvec = rt['T']
        if self.n_dilation:
            for k in range(self.n_dilation):
                mask = binary_dilation(mask)
        self.sc.process_view(self.intrinsics, rot, tvec, mask)

    def gt_compare(self):
        if self.gt_mode is False:
            return 0
        # compare current volume with ground truth (voxelwise) and return percentage
        comp = np.where(self.gt == self.sc.values(), True, False)
        eq_count = np.count_nonzero(comp)
        #perc_sim = (eq_count/np.prod(gt_vol.shape) )*100.
        # perc_sim = (eq_count/682176)*100. #682176number of voxels of the volumes used here
        perc_sim = eq_count * 0.00014658973637301812
        return perc_sim

    def gt_compare_solid(self):
        if self.gt_mode is False:
            return 0
        # compares only solid voxels (with 1;s) between ground truth and test_vol
        vol_solid_mask = np.where(self.sc.values() == 1, True, False)
        vol_n_solid_voxels = np.count_nonzero(vol_solid_mask)
        intersection = self.gt_solid_mask & vol_solid_mask
        n_intersection = np.count_nonzero(intersection)
        ratio = n_intersection / \
            (self.gt_n_solid_voxels + vol_n_solid_voxels - n_intersection)
        return ratio

    def calculate_col_position(self, init_state, steps):
        n_positions = self.total_positions
        n_pos = init_state + steps
        if n_pos > (n_positions-1):
            n_pos -= n_positions
        elif n_pos < 0:
            n_pos += n_positions
        return n_pos