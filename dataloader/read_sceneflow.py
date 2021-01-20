import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pickle
import params

p = params.Params()

dir_flyings3d = p.DATA_PATH + 'flythings3d/'
dir_monkaa = p.DATA_PATH + 'monkaa/'
dir_driving = p.DATA_PATH + 'driving/'

paths_train_img = []
paths_train_disp = []
paths_test_img = []
paths_test_disp = []

# flythings3d
for root, dirs, files in os.walk(dir_flyings3d + 'frames_cleanpass/TRAIN'):
    for file in sorted(files):
        paths_train_img.append(os.path.join(root, file))
for root, dirs, files in os.walk(dir_flyings3d + 'disparity/TRAIN'):
    for file in sorted(files):
        paths_train_disp.append(os.path.join(root, file))
for root, dirs, files in os.walk(dir_flyings3d + 'frames_cleanpass/TEST'):
    for file in sorted(files):
        paths_test_img.append(os.path.join(root, file))
for root, dirs, files in os.walk(dir_flyings3d + 'disparity/TEST'):
    for file in sorted(files):
        paths_test_disp.append(os.path.join(root, file))

# monkaa
for root, dirs, files in os.walk(dir_monkaa + 'frames_cleanpass'):
    for file in sorted(files):
        paths_train_img.append(os.path.join(root, file)) # train or test?
for root, dirs, files in os.walk(dir_monkaa + 'disparity'):
    for file in sorted(files):
        paths_train_disp.append(os.path.join(root, file)) # train or test?

# driving
for root, dirs, files in os.walk(dir_driving + 'frames_cleanpass'):
    for file in sorted(files):
        paths_train_img.append(os.path.join(root, file)) # train or test?
for root, dirs, files in os.walk(dir_driving + 'disparity'):
    for file in sorted(files):
        paths_train_disp.append(os.path.join(root, file)) # train or test?

paths_train_img_left = [path for path in paths_train_img if path.find('left') >= 0]
paths_train_img_right = [path for path in paths_train_img if path.find('right') >= 0]
paths_train_disp_left = [path for path in paths_train_disp if path.find('left') >= 0]
paths_train_disp_right = [path for path in paths_train_disp if path.find('right') >= 0]
paths_test_img_left = [path for path in paths_test_img if path.find('left') >= 0]
paths_test_img_right = [path for path in paths_test_img if path.find('right') >= 0]
paths_test_disp_left = [path for path in paths_test_disp if path.find('left') >= 0]
paths_test_disp_right = [path for path in paths_test_disp if path.find('right') >= 0]

fout_paths_train_img_left = open(p.DATA_PATH + p.sceneflow_paths_train_img_left, 'wb')
fout_paths_train_img_right = open(p.DATA_PATH + p.sceneflow_paths_train_img_right, 'wb')
fout_paths_train_disp_left = open(p.DATA_PATH + p.sceneflow_paths_train_disp_left, 'wb')
fout_paths_train_disp_right = open(p.DATA_PATH + p.sceneflow_paths_train_disp_right, 'wb')
fout_paths_test_img_left = open(p.DATA_PATH + p.sceneflow_paths_test_img_left, 'wb')
fout_paths_test_img_right = open(p.DATA_PATH + p.sceneflow_paths_test_img_right, 'wb')
fout_paths_test_disp_left = open(p.DATA_PATH + p.sceneflow_paths_test_disp_left, 'wb')
fout_paths_test_disp_right = open(p.DATA_PATH + p.sceneflow_paths_test_disp_right, 'wb')

pickle.dump(paths_train_img_left, fout_paths_train_img_left)
pickle.dump(paths_train_img_right, fout_paths_train_img_right)
pickle.dump(paths_train_disp_left, fout_paths_train_disp_left)
pickle.dump(paths_train_disp_right, fout_paths_train_disp_right)
pickle.dump(paths_test_img_left, fout_paths_test_img_left)
pickle.dump(paths_test_img_right, fout_paths_test_img_right)
pickle.dump(paths_test_disp_left, fout_paths_test_disp_left)
pickle.dump(paths_test_disp_right, fout_paths_test_disp_right)

fout_paths_train_img_left.close()
fout_paths_train_img_right.close()
fout_paths_train_disp_left.close()
fout_paths_train_disp_right.close()
fout_paths_test_img_left.close()
fout_paths_test_img_right.close()
fout_paths_test_disp_left.close()
fout_paths_test_disp_right.close()

print("the number of paths in", p.sceneflow_paths_train_img_left, ":", len(paths_train_img_left))
print("the number of paths in", p.sceneflow_paths_train_img_right, ":", len(paths_train_img_right))
print("the number of paths in", p.sceneflow_paths_train_disp_left, ":", len(paths_train_disp_left))
print("the number of paths in", p.sceneflow_paths_train_disp_right, ":", len(paths_train_disp_right))
print("the number of paths in", p.sceneflow_paths_test_img_left, ":", len(paths_test_img_left))
print("the number of paths in", p.sceneflow_paths_test_img_right, ":", len(paths_test_img_right))
print("the number of paths in", p.sceneflow_paths_test_disp_left, ":", len(paths_test_disp_left))
print("the number of paths in", p.sceneflow_paths_test_disp_right, ":", len(paths_test_disp_right))