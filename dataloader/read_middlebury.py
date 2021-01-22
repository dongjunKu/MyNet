import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pickle
import params

p = params.Params()

dir_middlebury = p.DATA_PATH + 'middlebury/'

paths_train_img_left = []
paths_train_img_right = []
paths_train_disp_left = []
paths_test_img_left = []
paths_test_img_right = []

# flythings3d
for root, dirs, files in os.walk(dir_middlebury + 'MiddEval3-data-F/MiddEval3/trainingF'):
    for file in sorted(files):
        temp = file.split('.')
        if temp[-1] == 'png':
            if temp[0] == 'im0':
                paths_train_img_left.append(os.path.join(root, file))
            if temp[0] == 'im1':
                paths_train_img_right.append(os.path.join(root, file))
for root, dirs, files in os.walk(dir_middlebury + 'MiddEval3-GT0-F/MiddEval3/trainingF'):
    for file in sorted(files):
        if file.split('.')[-1] == 'pfm':
            paths_train_disp_left.append(os.path.join(root, file))
for root, dirs, files in os.walk(dir_middlebury + 'MiddEval3-data-F/MiddEval3/testF'):
    for file in sorted(files):
        temp = file.split('.')
        if temp[-1] == 'png':
            if temp[0] == 'im0':
                paths_test_img_left.append(os.path.join(root, file))
            if temp[0] == 'im1':
                paths_test_img_right.append(os.path.join(root, file))

fout_paths_train_img_left = open(p.DATA_PATH + p.middlebury_paths_train_img_left, 'wb')
fout_paths_train_img_right = open(p.DATA_PATH + p.middlebury_paths_train_img_right, 'wb')
fout_paths_train_disp_left = open(p.DATA_PATH + p.middlebury_paths_train_disp_left, 'wb')
fout_paths_test_img_left = open(p.DATA_PATH + p.middlebury_paths_test_img_left, 'wb')
fout_paths_test_img_right = open(p.DATA_PATH + p.middlebury_paths_test_img_right, 'wb')

pickle.dump(paths_train_img_left, fout_paths_train_img_left)
pickle.dump(paths_train_img_right, fout_paths_train_img_right)
pickle.dump(paths_train_disp_left, fout_paths_train_disp_left)
pickle.dump(paths_test_img_left, fout_paths_test_img_left)
pickle.dump(paths_test_img_right, fout_paths_test_img_right)

fout_paths_train_img_left.close()
fout_paths_train_img_right.close()
fout_paths_train_disp_left.close()
fout_paths_test_img_left.close()
fout_paths_test_img_right.close()

print("the number of paths in", p.middlebury_paths_train_img_left, ":", len(paths_train_img_left))
print("the number of paths in", p.middlebury_paths_train_img_right, ":", len(paths_train_img_right))
print("the number of paths in", p.middlebury_paths_train_disp_left, ":", len(paths_train_disp_left))
print("the number of paths in", p.middlebury_paths_test_img_left, ":", len(paths_test_img_left))
print("the number of paths in", p.middlebury_paths_test_img_right, ":", len(paths_test_img_right))