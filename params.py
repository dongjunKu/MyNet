class Params():
    def __init__(self):
        self.PATH = "./workspace/Gu/EANet/EANet_0.0/"
        self.SAVE_PATH = "./workspace/Gu/EANet/EANet_0.0/saved_model/"
        self.DATA_PATH = "./workspace/Gu/Dataset/"

        self.mode = 'train'

        self.device = 'cuda'

        self.batch_size = 1
        
        self.train_disparity = 192
        self.train_size = (256, 512)
        self.train_kernel_coeff = 0

        self.test_disparity = 192
        self.test_size = (256, 512)
        self.test_kernel_coeff = 0.0001

        self.feature_num_list = [3, 64, 128, 256, 512, 1024]

        self.sceneflow_paths_train_img_left = "sceneflow_paths_train_img_left.pkl"
        self.sceneflow_paths_train_img_right = "sceneflow_paths_train_img_right.pkl"
        self.sceneflow_paths_train_disp_left = "sceneflow_paths_train_disp_left.pkl"
        self.sceneflow_paths_train_disp_right = "sceneflow_paths_train_disp_right.pkl"
        self.sceneflow_paths_test_img_left = "sceneflow_paths_test_img_left.pkl"
        self.sceneflow_paths_test_img_right = "sceneflow_paths_test_img_right.pkl"
        self.sceneflow_paths_test_disp_left = "sceneflow_paths_test_disp_left.pkl"
        self.sceneflow_paths_test_disp_right = "sceneflow_paths_test_disp_right.pkl"

        self.middlebury_paths_train_img_left = "middlebury_paths_train_img_left.pkl"
        self.middlebury_paths_train_img_right = "middlebury_paths_train_img_right.pkl"
        self.middlebury_paths_train_disp_left = "middlebury_paths_train_disp_left.pkl"
        self.middlebury_paths_train_disp_right = "middlebury_paths_train_disp_right.pkl"
        self.middlebury_paths_test_img_left = "middlebury_paths_test_img_left.pkl"
        self.middlebury_paths_test_img_right = "middlebury_paths_test_img_right.pkl"
        self.middlebury_paths_test_disp_left = "middlebury_paths_test_disp_left.pkl"
        self.middlebury_paths_test_disp_right = "middlebury_paths_test_disp_right.pkl"
