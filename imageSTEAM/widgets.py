from . import utils
from . import data
from . import color
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

class Timer:
    def __init__(self, timeout, callback):
        self._timeout = timeout
        self._callback = callback

    async def _job(self):
        await asyncio.sleep(self._timeout)
        self._callback()

    def start(self):
        self._task = asyncio.ensure_future(self._job())

    def cancel(self):
        self._task.cancel()

def debounce(wait):
    """ Decorator that will postpone a function's
        execution until after `wait` seconds
        have elapsed since the last time it was invoked. """
    def decorator(fn):
        timer = None
        def debounced(*args, **kwargs):
            nonlocal timer
            def call_it():
                fn(*args, **kwargs)
            if timer is not None:
                timer.cancel()
            timer = Timer(wait, call_it)
            timer.start()
        return debounced
    return decorator


def throttle(wait):
    """ Decorator that prevents a function from being called
        more than once every wait period. """
    def decorator(fn):
        time_of_last_call = 0
        scheduled, timer = False, None
        new_args, new_kwargs = None, None
        def throttled(*args, **kwargs):
            nonlocal new_args, new_kwargs, time_of_last_call, scheduled, timer
            def call_it():
                nonlocal new_args, new_kwargs, time_of_last_call, scheduled, timer
                time_of_last_call = time()
                fn(*new_args, **new_kwargs)
                scheduled = False
            time_since_last_call = time() - time_of_last_call
            new_args, new_kwargs = args, kwargs
            if not scheduled:
                scheduled = True
                new_wait = max(0, wait - time_since_last_call)
                timer = Timer(new_wait, call_it)
                timer.start()
        return throttled
    return decorator


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

    r = wsmall.shape[:2]
    out = wlarge // 2
    out[maxLoc[1]:maxLoc[1]+r[0], maxLoc[0]:maxLoc[0]+r[1], :] = wlarge[maxLoc[1]:maxLoc[1]+r[0], maxLoc[0]:maxLoc[0]+r[1], :]
  
    with output_img:
        cv2.rectangle(out, (maxLoc[0],maxLoc[1]), (maxLoc[0]+r[1],maxLoc[1]+r[0]), color=[0,255,0], thickness=2)
        utils.display_img(out, title="Output", dpi=200)
  
    final_out = VBox([HBox([input_img, input_template]), output_img])
    display(final_out)
    
def lightwave():

    l_min, l_max = 380, 780 # wavelength in nm
    c = 3 * 10**8           # speed of light in m/s
    Hz_min, Hz_max = [c / (l_max * 10**-9),
                        c / (l_min * 10**-9)]   # Frequency in Hz
    THz_min, THz_max = [int(Hz_min * 10**-12),
                        int(Hz_max * 10**-12)]  # Frequency in THz

    pixel = np.zeros((1,1,3), dtype='uint8')

    wave_out = widgets.Output()
    rgb_out = widgets.Output()
    
    style = {'description_width': 'initial'}
    sliderA = widgets.FloatSlider(description='Amplitude', value=1, min=0, max=1, step=0.05, style=style)
    sliderF = widgets.IntSlider(description='Frequency (THz)', value=THz_min, min=THz_min, max=THz_max, step=1, style=style)
    sliderL = widgets.FloatSlider(description='Wavelength (nm)', value=l_max, min=l_min, max=l_max, step=1, style=style)
    sliders = VBox([sliderA, sliderF, sliderL])

    # @throttle(0.2)
    def sliderF_changed(change):
        new_val = int((c/(change.new * 10**12)) * 10**9)
        sliderL.value = max(min(new_val, l_max), l_min)
  
    # @throttle(0.2)
    def sliderL_changed(change):
        sliderF.value = int((c/(change.new * 10**-9)) * 10**-12)

    sliderF.observe(sliderF_changed, names='value')
    sliderL.observe(sliderL_changed, names='value')

    x = np.linspace(0, l_max*2, 100)/(l_max)
    def _update_display(A, f, wl):
        rgb = color.wavelength_to_rgb(wl)
        pixel[0,0,:] = A * np.asarray(rgb)
        Freq = 2*np.pi * (l_max / (wl))
        wave = A * np.sin(Freq * x)
        plt.figure()
        ax = plt.axes()
        plt.ylim(-2,2)
        plt.xlim(0, l_max*2)
        plt.xlabel('Length (nm)')
        plt.ylabel('Amplitude')

        with wave_out:
            ax.set_facecolor(pixel[0,0,:]/255)
            plt.plot(x*l_max, wave, c='k', lw='6')
            plt.plot(x*l_max, wave, c='w', lw='4')
            ax.axhline(y=0, color='k', lw='1')
            plt.grid('on')
            plt.show()
            wave_out.clear_output(wait=True)

        # with rgb_out:
        #   plt.imshow(pixel)
        #   plt.axis('off')
        #   plt.show()
        #   rgb_out.clear_output(wait=True)      

    output = widgets.interactive_output(_update_display, 
                                        {'A': sliderA, 'f': sliderF, 'wl': sliderL})

    final_widget = VBox([wave_out, sliders, rgb_out])
    display(final_widget)

def yolo_(image = io.imread('./imageSTEAM/data/cat_owners.jpeg')):
    urls = {'yolov3.cfg': 'https://raw.github.com/pjreddie/darknet/master/cfg/yolov3.cfg',
            'yolov3.weights': 'https://pjreddie.com/media/files/yolov3.weights',
            'coco.names': 'https://raw.github.com/pjreddie/darknet/master/data/coco.names'}

    for key in urls.keys():
        if not os.path.isfile(key):
            print(key)
            myfile = requests.get(urls[key])
            open(key, 'wb').write(myfile.content)

    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # loading image
    img = image[:, :, ::-1].copy()  # data.choose_image()
    # img = cv2.resize(img,None,fx=2,fy=2)
    height, width, channels = img.shape

    # detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # for b in blob:
    #     for n,img_blob in enumerate(b):
    #         cv2.imshow(str(n),img_blob)

    net.setInput(blob)
    outs = net.forward(outputlayers)
    # print(outs[1])

    # Showing info on screen/ get confidence score of algorithm in detecting an object in blob
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # onject detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                # rectangle co-ordinaters
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                boxes.append([x, y, w, h])  # put all rectangle areas
                confidences.append(
                    float(confidence))  # how confidence was that object detected and show that percentage
                class_ids.append(class_id)  # name of the object tha was detected

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 6)
            cv2.putText(img, label, (x + 10, y + 10), font, 2, (255, 255, 255), 2)

    utils.display_img(img[:, :, ::-1], dpi=200)
    # cv2_imshow(img)


def greenScreen(image, background):
  # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  # defining range to excloude the green color from the image
  # the list contain -> [ value of the Red, value of the Green, value of the Blue
  lower_range = np.array([0,230,0])
  upper_range = np.array([30,255,30])
  # form [ (0 ->110) for Red, (100 -> 255) for Green, ...]
  mask = cv2.inRange(image, lower_range, upper_range)
  # set all other areas to zero except where mask area
  image[mask != 0] = [0, 0, 0]
  background = cv2.resize(background, (image.shape[1], image.shape[0]) )
  # set the mask area with black to be replaced with  Image
  background[mask == 0 ] = [0, 0, 0]
  complete_image = background + image
  plt.imshow(complete_image); plt.show()