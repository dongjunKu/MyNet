import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

def print_losses(head, losses, tail=None):
    txt = head
    for i in range(len(losses)):
        txt += ", \tloss" + str(i) + ": " + str(losses[i])
    if tail is not None:
        txt += tail
    print(txt)

def plot_error(cost_vol, disp_true, max_disp):
    pred = np.squeeze(np.argmax(cost_vol, axis=1)) # (H,W)
    disp_true = np.squeeze(disp_true)
    pred = resize(pred.astype(np.float), disp_true.shape) / pred.shape[1] * disp_true.shape[1] # TODO
    diff = abs(pred - disp_true)
    mask = disp_true < max_disp / pred.shape[1] * disp_true.shape[1]
    
    plt.imshow(diff, vmin=0)
    
    bad_h = np.sum(diff[mask] > 0.5) / (diff[mask].size + 1e-12)
    bad_1 = np.sum(diff[mask] > 1) / (diff[mask].size + 1e-12)
    bad_2 = np.sum(diff[mask] > 2) / (diff[mask].size + 1e-12)
    bad_4 = np.sum(diff[mask] > 4) / (diff[mask].size + 1e-12)
    bad_8 = np.sum(diff[mask] > 8) / (diff[mask].size + 1e-12)
    avgerr = np.sum(diff[mask]) / (diff[mask].size + 1e-12)
    rms = (np.sum(diff[mask]**2) / (diff[mask].size + 1e-12))**0.5

    text = "bad0.5: " + str(bad_h) + \
            "\nbad1.0: " + str(bad_1) + \
            "\nbad2.0: " + str(bad_2) + \
            "\nbad4.0: " + str(bad_4) + \
            "\nbad8.0: " + str(bad_8) + \
            "\n\navgerr: " + str(avgerr) + \
            "\nrms:     " + str(rms)
    
    print(text)

    return text

def show_errors(cost_vols, disp_true):
    i = 1
    while i**2 < len(cost_vols):
        i += 1

    for j in len(cost_vols):
        plt.subplot(i,i,j+1)
        plot_error(cost_vols[j], disp_true)

    plt.show()