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

def pixelHSVExample_(pixel):

    pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
    tmp_img = pixel.copy()

    segmented_out = widgets.Output()
    sliderH = widgets.FloatSlider(description='Hue', value=0, min=0, max=100)
    sliderS = widgets.FloatSlider(description='Saturation', value=0, min=0, max=100)
    sliderV = widgets.FloatSlider(description='Value', value=0, min=0, max=100)
    sliders = VBox([sliderH, sliderS, sliderV])

    def _update_display(h, s, v, tmp_img=tmp_img):
        tmp_img = pixel.copy()
        tmp_img[:,:,0] = cv2.add(tmp_img[:,:,0], h)
        tmp_img[:,:,1] = cv2.add(tmp_img[:,:,1], s)
        tmp_img[:,:,2] = cv2.add(tmp_img[:,:,2], v)
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_HSV2RGB)
        with segmented_out:
            plt.imshow(tmp_img)
            plt.show()
            segmented_out.clear_output(wait=True)


    output = widgets.interactive_output(_update_display,
                                        {'h': sliderH, 's': sliderS, 'v': sliderV})
    final_widget = VBox([output, sliders,segmented_out])
    display(final_widget)
def singlepixelHSVExample():
  pixel = np.zeros((1,1,3), dtype='float32')

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



def HSVSegmentationrange(img):
    img = img.astype('float32') / 255.0
    img_segm = np.zeros_like(img, dtype='float32')
    img_hsv = rgb_to_hsv(img)

    segmented_out = widgets.Output()

    sliderH = widgets.FloatRangeSlider(description='Hue', value=[0, 1.0], min=0, max=1, step=0.01)
    sliderS = widgets.FloatRangeSlider(description='Saturation', value=[0, 1.0], min=0, max=1, step=0.01)
    sliderV = widgets.FloatRangeSlider(description='Value', value=[0, 1.0], min=0, max=1, step=0.01)
    sliders = VBox([sliderH, sliderS, sliderV])

    def _update_display(h, s, v, img_segm=img_segm):
        tmp_img = img_hsv.copy()
        mask = (tmp_img[..., 0] >= h[0]) & (tmp_img[..., 0] <= h[1]) & (tmp_img[..., 1] >= s[0]) & (
                    tmp_img[..., 1] <= s[1]) & (tmp_img[..., 2] >= v[0]) & (tmp_img[..., 2] <= v[1])
        for c in range(img.shape[2]):
            img_segm[..., c] = img[..., c] * mask

        with segmented_out:
            plt.imshow(img_segm, cmap='gray')
            plt.show()
            segmented_out.clear_output(wait=True)

        img_segm = img

    output = widgets.interactive_output(_update_display,
                                        {'h': sliderH, 's': sliderS, 'v': sliderV})
    with output:
        plt.imshow(img);
        plt.show()

    final_widget = VBox([output, sliders, segmented_out])
    display(final_widget)


def HSVSegmentation(img):
    img = img.astype('float32') / 255.0
    img_segm = np.zeros_like(img, dtype='float32')
    img_hsv = rgb_to_hsv(img)
    # print(img_hsv)

    segmented_out = widgets.Output()

    sliderH = widgets.FloatSlider(description='Hue', value=0, min=0, max=1, step=0.05)
    sliderS = widgets.FloatSlider(description='Saturation', value=0, min=0, max=1, step=0.05)
    sliderV = widgets.FloatSlider(description='Value', value=0, min=0, max=1, step=0.05)
    sliders = VBox([sliderH, sliderS, sliderV])

    def _update_display(h, s, v, img_segm=img_segm):
        tmp_img = img_hsv.copy()
        # for c in range(img.shape[2]):
        #   tmp_img[..., c] = img[..., c] + [h, s, v][c]

        plt.imshow(hsv_to_rgb(tmp_img), cmap='gray')
        plt.show()

        mask = (tmp_img[..., 0] > h - 0.05) & (tmp_img[..., 0] < h + 0.05) & (tmp_img[..., 1] > s - 0.05) & (
                    tmp_img[..., 1] < s + 0.05) & (tmp_img[..., 2] > v - 0.05) & (tmp_img[..., 2] < v + 0.05)
        for c in range(img.shape[2]):
            img_segm[..., c] = img[..., c] * mask

        with segmented_out:
            plt.imshow(img_segm, cmap='gray')
            plt.show()
            segmented_out.clear_output(wait=True)

        img_segm = img

    output = widgets.interactive_output(_update_display,
                                        {'h': sliderH, 's': sliderS, 'v': sliderV})
    final_widget = VBox([output, sliders, segmented_out])
    display(final_widget)

# def pixelHSVExample(pixel):
#     # pixel = #np.zeros((1,1,3), dtype='float32')
#     pixel = pixel.astype('float32') / 255.0
#     pixel = rgb_to_hsv(pixel)
#
#     segmented_out = widgets.Output()
#     sliderH = widgets.FloatSlider(description='Hue', value=0.5, min=0, max=1)
#     sliderS = widgets.FloatSlider(description='Saturation', value=0.5, min=0, max=1)
#     sliderV = widgets.FloatSlider(description='Value', value=0.5, min=0, max=1)
#     sliders = VBox([sliderH, sliderS, sliderV])
#
#     def _update_display(h, s, v):
#         tmp_img = pixel.copy()
#         for c in range(pixel.shape[2]):
#             tmp_img[..., c] = pixel[..., c] + [h, s, v][c]
#
#         with segmented_out:
#             plt.imshow(hsv_to_rgb(tmp_img), cmap='gray')
#             plt.show()
#             segmented_out.clear_output(wait=True)
#
#
#     output = widgets.interactive_output(_update_display,
#                                         {'h': sliderH, 's': sliderS, 'v': sliderV})
#     final_widget = VBox([output, sliders,segmented_out])
#     display(final_widget)
