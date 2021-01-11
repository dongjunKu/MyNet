class Params():
    def __init__(self):
        self.PATH = "./workspace/Gu/EANet/EANet_0.0/"
        self.DATA_PATH = "./workspace/Gu/Dataset/"
        self.paths_train_img_left = "paths_train_img_left.pkl"
        self.paths_train_img_right = "paths_train_img_right.pkl"
        self.paths_train_disp_left = "paths_train_disp_left.pkl"
        self.paths_train_disp_right = "paths_train_disp_right.pkl"
        self.paths_test_img_left = "paths_test_img_left.pkl"
        self.paths_test_img_right = "paths_test_img_right.pkl"
        self.paths_test_disp_left = "paths_test_disp_left.pkl"
        self.paths_test_disp_right = "paths_test_disp_right.pkl"

        self.batch_size = 1
        
        self.train_disparity = 192
        self.test_disparity = 320
