import matplotlib.pyplot as plt

# print('Utils Import')

def test_func_utils():
    print("Test function in utils prints")
    
# Displays an image, converts BGR to RGB
def display_img(img, title = None, dpi=None, axis=True):
    vmax = 255 if img.dtype == 'uint8' else 1.0
    if dpi != None:
        plt.figure(dpi=dpi)
    plt.imshow(img, cmap='gray', vmin=0, vmax=vmax)
    if title != None:
        plt.title(title)
    if axis == False:
        plt.axis('off')
    plt.show()
    
def crop(image, display=True):
    h, w = image.shape[:2]
    print("Cropping image with original dimensions: \n{} rows x {} columns\n".format(h, w))
    h0 = int(input("Enter crop START ROW (min=0, max={}): ".format(h)))
    h1 = int(input("Enter crop END ROW (min=0, max={}): ".format(h)))
    w0 = int(input("Enter crop START COLUMN (min=0, max={}): ".format(w)))
    w1 = int(input("Enter crop END COLUMN (min=0, max={}): ".format(w)))
    
    h_min, h_max = min(h0, h1), max(h0, h1)
    w_min, w_max = min(w0, w1), max(w0, w1)
    
    img_crop = image[h_min:h_max, w_min:w_max]
  
    if display == True:
        display_img(img_crop, title='Cropped')

    return img_crop

import cv2
import numpy as np

class ImageArray(np.ndarray):

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.length = len(input_array)
        if obj.length >= 1:
          obj.width = len(input_array[0])
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.info = getattr(obj, 'info', None)

def read(name):
  img = cv2.imread(name)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return ImageArray(img)

def flip(src, direction='leftright'):
  if direction == 'leftright':
    print("true")
    fd = 1
    title = 'Left/Right'
  else:
    fd = 0
    title = 'Up/Down'
  display_img(cv2.flip(src, fd), title=title)

# Rotates the image
def rotate(rot, angle, origin, scale=1):
  pt_y = int(origin[1])
  pt_x = int(origin[0])

  rows = int(len(rot) * scale)
  cols = int(len(rot[0]) * scale)

  # rotation - get proper rc length for output?
  M = cv2.getRotationMatrix2D((pt_x,pt_y),angle,scale)
  dst = cv2.warpAffine(rot,M,(cols,rows))
  display_img(dst, title='Rotated ' + str(angle) + ' Degrees')
