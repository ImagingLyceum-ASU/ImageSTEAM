from . import utils

import numpy as np
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout
from IPython.display import display

def cropSegmentation(img):
    h, w = img.shape[:2]
    # Initialize empty Output
    img_segmented = np.zeros_like(img) 
    output = widgets.Output()
    with output:
        utils.display_img(img_segmented, title='Segmented Output', dpi=100)
        output.clear_output(wait=True)

    ## Controls
    # Sliders TODO: Simplify this
    sliderHmin = widgets.IntSlider(description='Hmin', value=0, min=0, max=h)
    sliderHmax = widgets.IntSlider(description='Hmax', value=h, min=0, max=h)
    sliderWmin = widgets.IntSlider(description='Wmin', value=0, min=0, max=w)
    sliderWmax = widgets.IntSlider(description='Wmax', value=w, min=0, max=w)
    sliders = VBox([sliderHmin, sliderHmax, sliderWmin, sliderWmax])
    # Buttons
    buttonConfirm = widgets.Button(description="Confirm Crop")
    buttonReset = widgets.Button(description="Reset")
    buttons = HBox([buttonConfirm, buttonReset])
    # Combined Controls
    ui = VBox([sliders, buttons], layout=Layout(justify_content='center'))

    ## Interactive Crop Preview
    def _preview_crop(h0, h1, w0, w1):
        tmp_img = (img.copy())//2
        tmp_img[h0:h1+1, w0:w1+1] *= 2
        utils.display_img(tmp_img, title='Crop Preview', dpi=100)

    preview = widgets.interactive_output(_preview_crop, {
        'h0': sliderHmin, 'h1': sliderHmax,
        'w0': sliderWmin, 'w1': sliderWmax})


    ## Segmentation Buttons
    def _on_confirm_clicked(b, img_segmented=img_segmented):
        h0, h1 = sliderHmin.value, sliderHmax.value
        w0, w1 = sliderWmin.value, sliderWmax.value
        img_segmented[h0:h1+1, w0:w1+1] = img[h0:h1+1, w0:w1+1]
        with output:
            utils.display_img(img_segmented, title='Segmented Output', dpi=100)
            output.clear_output(wait=True)
        return img_segmented

    def _on_reset_clicked(b, img_segmented=img_segmented):
        img_segmented *= 0
        with output:
            utils.display_img(img_segmented, title='Segmented Output', dpi=100)
            output.clear_output(wait=True)
        return img_segmented

    buttonConfirm.on_click(_on_confirm_clicked)
    buttonReset.on_click(_on_reset_clicked)
    
    ## Display Final Widget
    final_widget = HBox([preview, ui, output], 
                        layout=Layout(justify_content='flex-start'))
    display(final_widget)