import matplotlib.pyplot as plt

# print('Utils Import')

def test_func_utils():
    print("Test function in utils prints")
    
# Displays an image, converts BGR to RGB
def display_img(a, title = "Original"):
    plt.imshow(a, cmap='gray'), plt.title(title)
    plt.show()