import imageio as io
import os.path as osp

data_dir = osp.abspath(osp.dirname(__file__))

def amongUs():
  return io.imread(osp.join(data_dir, 'AmongUs_gray.png'))