import numpy as np
import wandb
from lightningtools import reporter


def save_figure_to_numpy(fig, spectrogram=False):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if spectrogram:
        return data
    data = np.transpose(data, (2, 0, 1))
    return data


def log_image(image):
    image = ((image[0] + 0.5)).float()
    return wandb.Image(image)


reporter.register("image", log_image)
