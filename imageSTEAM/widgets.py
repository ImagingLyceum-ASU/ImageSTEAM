from . import utils
from . import data

import cv2
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
    
def pixelRGBExample():
    pixel_grid = np.zeros((1,1,3), dtype='uint8')

    sliderR = widgets.IntSlider(description='Red', value=0, min=0, max=255)
    sliderG = widgets.IntSlider(description='Green', value=0, min=0, max=255)
    sliderB = widgets.IntSlider(description='Blue', value=0, min=0, max=255)
    sliders = VBox([sliderR, sliderG, sliderB])
  
    def _update_display(r, g, b):
        tmp_img = pixel_grid.copy()
        for c in range(pixel_grid.shape[2]):
            # print([r, g, b][c].shape)
            tmp_img[..., c] = pixel_grid[..., c] + [r, g, b][c]

        utils.display_img(tmp_img, axis=False)

    output = widgets.interactive_output(_update_display, 
                                        {'r': sliderR, 'g': sliderG, 'b': sliderB})
    final_widget = VBox([output, sliders])
    display(final_widget)
  
def pixelColor():
    ## Import example grayscale image
    img_orig = data.coffee() ##TODO: Update this to RGB image
    h, w = img_orig.shape[:2]

    # Make copy which will be our working copy so we can reset
    img_edit = img_orig.copy()

    ## User Controls
    # Sliders
    sliderRow = widgets.IntSlider(description='Row', value=None, min=0, max=h-1)
    sliderCol = widgets.IntSlider(description='Column', value=None, min=0, max=w-1)
    sliderR = widgets.IntSlider(description='Red', value=128, min=0, max=255, orientation='vertical')
    sliderG = widgets.IntSlider(description='Green', value=128, min=0, max=255, orientation='vertical')
    sliderB = widgets.IntSlider(description='Blue', value=128, min=0, max=255, orientation='vertical')
    slidersRGB = HBox([sliderR, sliderG, sliderB])
    sliders = VBox([sliderRow, sliderCol, slidersRGB])
    # Buttons
    buttonConfirm = widgets.Button(description='Confirm')
    buttonReset = widgets.Button(description='Reset')
    buttons = HBox([buttonConfirm, buttonReset])
    # Combined UI
    ui = VBox([sliders, buttons])

    ## Interactive Image Display
    def _update_display(row, col, r, g, b):
        tmp_img = img_edit.copy()
        for c in range(tmp_img.shape[2]):
            tmp_img[row-3:row+4, col-3:col+4, c] = [r, g, b][c]
        utils.display_img(tmp_img, dpi=100)
        # preview.clear_output(wait=True)
  
    ## Button Functionality
    def _confirm_clicked(b, img_edit=img_edit):
        row, col = sliderRow.value, sliderCol.value
        rgb = [sliderR.value, sliderG.value, sliderB.value]
        for c in range(img_edit.shape[2]):
            img_edit[row, col, c] = rgb[c]

        with preview:
            preview.clear_output(wait=True)
        # display(final_widget)
        return img_edit

    def _reset_clicked(b, img_edit=img_edit):
        with preview:
            # utils.display_img(img_orig, title='Img_Orig', dpi=100)
            img_edit = img_orig#.copy()
            # # utils.display_img(img_edit, title='Img_Edit', dpi=100)
            # # clear_output() 
            # with preview:
            preview.clear_output(wait=True)
        # display(final_widget)
        return img_edit

    buttonConfirm.on_click(_confirm_clicked)
    buttonReset.on_click(_reset_clicked)

    preview = widgets.interactive_output(_update_display,
                                         {'row': sliderRow, 'col': sliderCol, 
                                          'r': sliderR, 'g': sliderG, 'b': sliderB})

    ## Final Widget
    final_widget = VBox([preview, ui])
    display(final_widget)
    # display.clear_output()
    
def templateMatch(image, template):
    wlarge = image #cv2.imread('Waldolarge.png')
    # wlarge = #cv2.cvtColor(wlarge,cv2.COLOR_BGR2RGB)
    wsmall = template #cv2.imread('Waldosmall.png')
    input_img = widgets.Output()
    input_template = widgets.Output()
    output_img = widgets.Output()

    with input_img:
        utils.display_img(wlarge, title="Where's Waldo?", dpi=100)
    with input_template:
        utils.display_img(wsmall, title="Template")


    # Perform Template Matching
    template = cv2.matchTemplate(wlarge, wsmall, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(template)

    r = 50
    out = wlarge // 2
    out[maxLoc[1]:maxLoc[1]+r, maxLoc[0]:maxLoc[0]+r, :] = wlarge[maxLoc[1]:maxLoc[1]+r, maxLoc[0]:maxLoc[0]+r, :]
    # out[maxLoc[1]-r:maxLoc[1]+r, maxLoc[0]-r:maxLoc[0]+r, :] = wlarge[maxLoc[1]-r:maxLoc[1]+r, maxLoc[0]-r:maxLoc[0]+r, :]
    # out[maxLoc[0]-r:maxLoc[0]+r, maxLoc[1]-r:maxLoc[1]+r, :] = wlarge[maxLoc[0]-r:maxLoc[0]+r, maxLoc[1]-r:maxLoc[1]+r, :]

  
    with output_img:
        cv2.rectangle(out, (maxLoc[0],maxLoc[1]), (maxLoc[0]+r,maxLoc[1]+r), color=[0,255,0], thickness=2)
        utils.display_img(out, title="Output", dpi=200)
  
    final_out = VBox([HBox([input_img, input_template]), output_img])
    display(final_out)