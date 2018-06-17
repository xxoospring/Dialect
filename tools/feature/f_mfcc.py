import wave
from time import sleep

import librosa
import tensorflow as tf
import numpy as np
import sys
sys.path.append('../')
from file import *


name_map = {
    'minnan':   0,
    'nanchang': 1
}
data_root = 'F:/DL/data/Dialect_Challenge/'


def mix_all(root, t_d='train'): # train set or dev set
    dirs = get_dir_name(root)
    data_lst = []
    for d in dirs:
        data_dirs = get_dir_name(root + d + '/' + t_d+'/')
        for data_d in data_dirs:
            if t_d == 'dev':
                path = root + d + '/' + t_d+ '/' + data_d+'/' + 'long'+'/'
            else:
                path = root + d + '/' + t_d+ '/' + data_d+'/'
            items = file_filter(path, 'pcm')
            items = [path + it for it in items]
            data_lst.extend(items)
    return data_lst


def mfcc(data, n=32, pace=10, overlap=5):
    data_num = len(data)
    sr = 16000
    reg = []
    max_l = 0
    for d in data:
        y = np.memmap(d, dtype='h', mode='r')
        l = len(y) / sr
        print(l)
        if l > 5.12:
            max_l += 1
        reg.append(len(y))
    reg_np = np.array(reg)
    print(len(reg), np.max(reg_np)/sr, np.min(reg_np)/sr, np.average(reg_np) / sr)
    print(max_l)


if __name__ == '__main__':
    data_lst = mix_all(data_root, 'train')
    print(len(data_lst))
    mfcc(data_lst)