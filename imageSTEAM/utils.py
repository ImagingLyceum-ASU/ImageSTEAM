import matplotlib.pyplot as plt

# print('Utils Import')

def test_func_utils():
    print("Test function in utils prints")
    
# Displays an image, converts BGR to RGB
def display_img(a, title = "Original", dpi=None):
    if dpi != None:
      plt.figure(dpi=dpi)
    plt.imshow(a, cmap='gray'), plt.title(title)
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