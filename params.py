class Params():
    def __init__(self):
        self.PATH = "./workspace/Gu/EANet/EANet_0.0/"
        self.SAVE_PATH = "./workspace/Gu/EANet/EANet_0.0/saved_model/"
        self.DATA_PATH = "./workspace/Gu/Dataset/"
        self.paths_train_img_left = "paths_train_img_left.pkl"
        self.paths_train_img_right = "paths_train_img_right.pkl"
        self.paths_train_disp_left = "paths_train_disp_left.pkl"
        self.paths_train_disp_right = "paths_train_disp_right.pkl"
        self.paths_test_img_left = "paths_test_img_left.pkl"
        self.paths_test_img_right = "paths_test_img_right.pkl"
        self.paths_test_disp_left = "paths_test_disp_left.pkl"
        self.paths_test_disp_right = "paths_test_disp_right.pkl"

        self.mode = 'train'

        self.device = 'cuda'

        self.batch_size = 1
        
        self.train_disparity = 192
        self.train_size = (256, 512)
        self.test_disparity = 320

        self.feature_num_list = [3, 64, 128, 256, 512, 1024]
