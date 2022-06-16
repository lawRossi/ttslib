"""
@Author: Rossi
Created At: 2022-04-20
"""

import importlib

from matplotlib import pyplot as plt
import numpy as np
import torch


def plot_spectrogram(spectrogram,  fig_size=(16, 10)):
    if isinstance(spectrogram, torch.Tensor):
        spectrogram_ = spectrogram.detach().cpu().numpy().squeeze().T
    else:
        spectrogram_ = spectrogram.T
    spectrogram_ = spectrogram_.astype(np.float32) if spectrogram_.dtype == np.float16 else spectrogram_
    fig = plt.figure(figsize=fig_size)
    plt.imshow(spectrogram_, aspect="auto", origin="lower")
    plt.colorbar()
    plt.tight_layout()
    plt.close()
    return fig


def find_class(class_path):
    module = class_path[:class_path.rfind(".")]
    class_name = class_path[class_path.rfind(".")+1:]
    ip_module = importlib.import_module(".", module)
    class_ = getattr(ip_module, class_name)
    return class_
