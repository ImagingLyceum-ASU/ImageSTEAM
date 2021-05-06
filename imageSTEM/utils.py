import matplotlib.pyplot as plt

# print('Utils Import')

def test_func_utils():
    print("Test function in utils prints")
    
# Displays an image, converts BGR to RGB
def display_img(a, title = "Original"):
    plt.imshow(a, cmap='gray'), plt.title(title)
    plt.show()
    
def imgZoom(image, h_range, w_range, display=True):
  h_min, h_max = min(h_range), max(h_range)
  w_min, w_max = min(w_range), max(w_range)
  img_zoom = image[h_min:h_max, w_min:w_max]
  
  if display == True:
    display_img(img_zoom)

  return img_zoom