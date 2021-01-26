class Params():
    def __init__(self):
        self.PATH = "./workspace/Gu/EANet/EANet_0.0/"
        self.SAVE_PATH = self.PATH + "saved_model/"
        self.DATA_PATH = "./workspace/Gu/Dataset/"

        self.mode = 'test'

        self.device = 'cuda'

        self.batch_size = 1
        
        self.train_disparity = 192 # train, validate
        self.train_size = (256, 512) # train, validate
        self.train_kernel_coeff = 0 # train, validate

        self.test_disparity = 320 # test 320
        self.test_size = (512, 768) # test (512, 768)
        self.test_kernel_coeff = 0 # test

        # for dataloader
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

        # for viewer
        self.kernel_path = self.PATH + "result/kernel/"