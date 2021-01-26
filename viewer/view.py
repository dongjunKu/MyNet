import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize

def print_losses(head, losses, tail=None):
    txt = head
    for i in range(len(losses)):
        txt += ", \tloss" + str(i) + ": " + str(losses[i])
    if tail is not None:
        txt += tail
    print(txt)

def plot_pred(cost_vol, max_disp):
    plt.imshow(np.squeeze(np.argmax(cost_vol, axis=1)), vmin=0, vmax=max_disp)

# cost_vol: (H, W, D), disp_true: (H, W, 1) or (H, W)
def plot_pred_error(axes, disp_true, cost_vol, size, max_disp):

    assert len(axes) == 3

    for axis in axes:
        axis.cla()

    pred = np.argmax(cost_vol, axis=-1) # (H,W,D)=>(H,W)
    disp_true = np.squeeze(disp_true) # (H,W)
    pred = resize(pred.astype(np.float), disp_true.shape) / pred.shape[1] * disp_true.shape[1] # TODO
    diff = abs(pred - disp_true)
    mask = disp_true < max_disp / pred.shape[1] * disp_true.shape[1]
    
    im_diff = diff.copy()
    im_diff[np.isinf(disp_true)] = 0    
    
    bad_h = np.sum(diff[mask] > 0.5) / (diff[mask].size + 1e-12)
    bad_1 = np.sum(diff[mask] > 1) / (diff[mask].size + 1e-12)
    bad_2 = np.sum(diff[mask] > 2) / (diff[mask].size + 1e-12)
    bad_4 = np.sum(diff[mask] > 4) / (diff[mask].size + 1e-12)
    bad_8 = np.sum(diff[mask] > 8) / (diff[mask].size + 1e-12)
    avgerr = np.sum(diff[mask]) / (diff[mask].size + 1e-12)
    rms = (np.sum(diff[mask]**2) / (diff[mask].size + 1e-12))**0.5

    info = "bad0.5: " + str(bad_h) + \
            "\nbad1.0: " + str(bad_1) + \
            "\nbad2.0: " + str(bad_2) + \
            "\nbad4.0: " + str(bad_4) + \
            "\nbad8.0: " + str(bad_8) + \
            "\n\navgerr: " + str(avgerr) + \
            "\nrms:     " + str(rms)
    
    axes[0].set_title("disp_pred"), axes[0].imshow(pred, vmin=0, vmax=max_disp/size[1]*disp_true.shape[1])
    axes[1].set_title("diff"), axes[1].imshow(im_diff, vmin=0)
    axes[2].axis('off'), axes[2].text(0.05, 0.3, info, fontsize=8)

# img: (H, W, C), ker: (H, W, kernel_H, kernel_W, C=1), pos: (W, H)
def visualize_kernel(img, ker, pos):
    x, y = pos
    x = int(x)
    y = int(y)
    kerH = ker.shape[2]
    kerW = ker.shape[3]
    assert kerH % 2 == 1 and kerW % 2 == 1
    scaleH = img.shape[0] // ker.shape[0]
    scaleW = img.shape[1] // ker.shape[1]
    y = (y // scaleH) * scaleH
    x = (x // scaleW) * scaleW
    print(y//scaleH,"/", ker.shape[0],",", x//scaleW, "/", ker.shape[1])
    im_ker = np.uint(255 * resize(ker[y//scaleH,x//scaleW], (kerH*scaleH, kerW*scaleW)))
    # print(scaleH, scaleW, y-(kerH//2)*scaleH, )
    img[y-(kerH//2)*scaleH:y+(kerH//2+1)*scaleH,x-(kerW//2)*scaleW:x+(kerW//2+1)*scaleW] = im_ker
    
    return img

def plot_kernel(axis, img, size, ker, pos):
    axis.cla()
    axis.set_title("image with kernel"), axis.imshow(visualize_kernel(cv2.resize(img, size[::-1]), ker, pos))

def plot_all(axes, imL, imR, dispL, costVol, size, max_disp, kerL=None, kerR=None, pos=None):

    assert len(axes) == 6

    for axis in axes:
        axis.cla()

    if kerL is None:
        axes[0].set_title("imageL"), axes[0].imshow(imL)
    else:
        plot_kernel(axes[0], imL, size, kerL, pos)
    if kerR is None:
        axes[1].set_title("imageR"), axes[1].imshow(imR)
    else:
        plot_kernel(axes[1], imR, size, kerR, pos)
    axes[2].set_title("disp_true"), axes[2].imshow(dispL, vmin=0, vmax=max_disp/size[1]*dispL.shape[1])
    plot_pred_error(axes[3:6], dispL, costVol, size, max_disp)