import os.path as osp

import imageio as io

from .. import utils

data_dir = osp.abspath(osp.dirname(__file__))

__all__ = ['example',
           'example_gray',
           'animals',
           'astronaut',
           'camera',
           'coffee',
           'coins',
           'page',
           'amongUs',
           'headScan',
           'fruits',
           'waldo',
           'waldo2',
           'src',
           'bgr',
           'greenscreen_video',
           'choose_image']


def example():
    return astronaut()


def example_gray():
    return camera()


def animals():
    return io.imread(osp.join(data_dir, 'animals.png'))


def astronaut():
    return io.imread(osp.join(data_dir, 'astronaut.png'))


def camera():
    return io.imread(osp.join(data_dir, 'camera.png'))


def coffee():
    return io.imread(osp.join(data_dir, 'coffee.png'))


def coins():
    return io.imread(osp.join(data_dir, 'coins.png'))


def dog():
    return io.imread(osp.join(data_dir, 'dog.jpg'))


def page():
    return io.imread(osp.join(data_dir, 'page.png'))


def amongUs():
    return io.imread(osp.join(data_dir, 'AmongUs_gray.png'))


def headScan():
    headScan_0 = io.imread(osp.join(data_dir, 'headScan/HeadScan_0.jpg'))
    headScan_1L = io.imread(osp.join(data_dir, 'headScan/HeadScan_1L.jpg'))
    headScan_1R = io.imread(osp.join(data_dir, 'headScan/HeadScan_1R.jpg'))
    headScan_2 = io.imread(osp.join(data_dir, 'headScan/HeadScan_2.jpg'))
    return [headScan_0, headScan_1L, headScan_1R, headScan_2]


def fruits():
    return io.imread(osp.join(data_dir, 'fruits.jpg'))

def src():
    return io.imread(osp.join(data_dir, 'greenscreen.jpg'))

def bgr():
    return io.imread(osp.join(data_dir, 'greenscreen_bg.jpg'))

def greenscreen_video():
    return io.get_reader(osp.join(data_dir, 'greenscreen_video.mp4'))


def waldo():
    return io.imread(osp.join(data_dir, 'waldo.png'))


def waldo2():  # TODO: Combine this to a single Waldo
    return io.imread(osp.join(data_dir, 'waldo22.png'))[:, :, :3]


def choose_image():
    from ipywidgets import interact
    from .. import data

    def _update(name):
        global img
        img = getattr(data, name)()
        utils.display_img(img, title=name)
        # return img

    out = interact(_update, name=sorted(set(__all__) - {'example', 'example_gray', 'choose_image'}));

    # utils.display_img(out)

    return img
