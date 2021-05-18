import imageio as io
import os.path as osp

data_dir = osp.abspath(osp.dirname(__file__))

def amongUs():
    return io.imread(osp.join(data_dir, 'AmongUs_gray.png'))

def headScan():
    headScan_0 = io.imread(osp.join(data_dir, 'headScan/HeadScan_0.jpg'))
    headScan_1L = io.imread(osp.join(data_dir, 'headScan/HeadScan_1L.jpg'))
    headScan_1R = io.imread(osp.join(data_dir, 'headScan/HeadScan_1R.jpg'))
    headScan_2 = io.imread(osp.join(data_dir, 'headScan/HeadScan_2.jpg'))
    return [headScan_0, headScan_1L, headScan_1R, headScan_2]