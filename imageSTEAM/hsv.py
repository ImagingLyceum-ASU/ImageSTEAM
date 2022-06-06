from . import utils
from . import data
from . import color

import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import skimage
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2
import asyncio
import cv2
import numpy as np
import requests
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout
from IPython.display import display
from time import time
import os
import cv2
import numpy as np
import imageio as io


def pixelHSVExample(pixel):
    # pixel = #np.zeros((1,1,3), dtype='float32')
    pixel = rgb_to_hsv(pixel)

    sliderH = widgets.FloatSlider(description='Hue', value=0.5, min=0, max=1)
    sliderS = widgets.FloatSlider(description='Saturation', value=0.5, min=0, max=1)
    sliderV = widgets.FloatSlider(description='Value', value=0.5, min=0, max=1)
    sliders = VBox([sliderH, sliderS, sliderV])

    def _update_display(h, s, v):
        tmp_img = pixel.copy()
        for c in range(pixel.shape[2]):
            tmp_img[..., c] = pixel[..., c] + [h, s, v][c]

        plt.imshow(hsv_to_rgb(tmp_img), cmap='gray')
        plt.show()

    output = widgets.interactive_output(_update_display,
                                        {'h': sliderH, 's': sliderS, 'v': sliderV})
    final_widget = VBox([output, sliders])
    display(final_widget)