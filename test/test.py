import wave
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os


def pcm2wav(src, store_path):
    with open(src, 'rb') as pcmfile:
        pcmdata = pcmfile.read()
    print(src.split('.')[-2])
    with wave.open(store_path+'0.wav', 'wb') as wavfile:
        wavfile.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
        wavfile.writeframes(pcmdata)


def show(y):
    # fig = plt.figure()
    # ax = fig.add_subplot('111')
    # ax.scatter(np.arange(len(y)), y)
    plt.plot(y)
    plt.show()

if __name__ == '__main__':
    pcm2wav('./nanchang_train_speaker02_001.pcm', './')
    y ,sr = librosa.load('./0.wav', sr=16000)
    # os.remove('./0.wav')
    # mfcc = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=40)
    # print(len(mfcc))
    # from scipy.io.wavfile import read
    # sr, d = read('./nanchang_train_speaker02_001.pcm',)
    # print(sr, len(d) / sr)
    d = np.memmap('./nanchang_train_speaker02_001.pcm',dtype='h', mode='r')
    print(y[:10])
    print(d[:10])
    show(librosa.feature.mfcc(y,sr=16000,n_mfcc=128))
    show(librosa.feature.mfcc(d.astype(dtype=np.float32),sr=16000,n_mfcc=128))
    # mfcc = librosa.feature.mfcc(y=d, sr=sr, n_mfcc=40)
    # print(len(mfcc))
