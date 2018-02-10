#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python 2.7.10                                                                                                                                                
# tensorflow (1.4.1)
# Keras (2.1.3)

import numpy as np
import scipy as sp
import sys
import wave
import tempfile
from six.moves import xrange, zip
import scipy.special as spc
#from python_speech_features import fbank
import scipy.signal

from keras.layers import Dense
from keras.models import Model, Sequential
import os.path

banks = 120
input_len = 600
hidden_dim = 100
batch_size = 256
epocs = 1
_window = None
_fs = 44100

def preEmphasis(signal, p):
    """プリエンファシスフィルタ"""
    # 係数 (1.0, -p) のFIRフィルタを作成
    return scipy.signal.lfilter([1.0, -p], 1, signal)

def hz2mel(f):
    """Hzをmelに変換"""
    return 1127.01048 * np.log(f / 700.0 + 1.0)

def mel2hz(m):
    """melをhzに変換"""
    return 700.0 * (np.exp(m / 1127.01048) - 1.0)

def melFilterBank(fs, nfft, numChannels):
    """メルフィルタバンクを作成"""
    # ナイキスト周波数（Hz）
    fmax = fs / 2
    # ナイキスト周波数（mel）
    melmax = hz2mel(fmax)
    # 周波数インデックスの最大数
    nmax = nfft / 2
    # 周波数解像度（周波数インデックス1あたりのHz幅）
    df = fs / nfft
    # メル尺度における各フィルタの中心周波数を求める
    dmel = melmax / (numChannels + 1)
    melcenters = np.arange(1, numChannels + 1) * dmel
    # 各フィルタの中心周波数をHzに変換
    fcenters = mel2hz(melcenters)
    # 各フィルタの中心周波数を周波数インデックスに変換
    indexcenter = np.round(fcenters / df)
    # 各フィルタの開始位置のインデックス
    indexstart = np.hstack(([0], indexcenter[0:numChannels - 1]))
    # 各フィルタの終了位置のインデックス
    indexstop = np.hstack((indexcenter[1:numChannels], [nmax]))

    indexcenter = indexcenter.astype(np.int64)
    indexstart = indexstart.astype(np.int64)
    indexstop = indexstop.astype(np.int64)

    ii64 = np.iinfo(np.int64)
    int64_max = ii64.max

    filterbank = np.zeros((numChannels, nmax))
    for c in np.arange(0, numChannels):
        # 三角フィルタの左の直線の傾きから点を求める
        increment = int64_max
        if indexcenter[c] - indexstart[c] != 0:
            increment= 1.0 / (indexcenter[c] - indexstart[c])
        for i in np.arange(indexstart[c], indexcenter[c]):
            filterbank[c, i] = (i - indexstart[c]) * increment
        # 三角フィルタの右の直線の傾きから点を求める
        decrement = int64_max
        if indexstop[c] - indexcenter[c] != 0:
            decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in np.arange(indexcenter[c], indexstop[c]):
            filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)

    return filterbank, fcenters


def uniting_channles(leftsignal, rightsignal):
    ret = []
    for i, j in zip(leftsignal, rightsignal):
        ret.append(i)
        ret.append(j)
#    return np.array(ret, np.float32)
    return np.array(ret, np.complex)

def train(train_in, train_out, test_in, test_out):
    model = Sequential()

    model.add(Dense(banks, activation='relu', input_shape=(banks,)))
    model.add(Dense(hidden_dim, activation='relu'))
    model.add(Dense(hidden_dim, activation='relu'))
    model.add(Dense(banks,  activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy')

    x_train_in, _ = preprocess(train_in)
    x_train_out, _ = preprocess(train_out)
    y_test_in, _ = preprocess(test_in)
    y_test_out, _ = preprocess(test_out)

    print("preprocess for train finished.")

    model.fit(x_train_in, x_train_out,
                nb_epoch=epocs,
                batch_size=batch_size,
                shuffle=False,
                validation_data=(y_test_in, y_test_out))

    #model.save_weights('./denoise.weight')
    #autoencoder.load_weights('autoencoder.h5')

    return model


def write(fname, params,signal):
#    st = tempfile.TemporaryFile()
#    wf=wave.open(st,'wb')
    wf=wave.open(fname,'wb')
    wf.setparams(params)
    s=sp.int16(signal*32767.0).tostring()
    wf.writeframes(s)
    wf.close()
#    st.seek(0)
#    print st.read()

#    with open("dnn_denoised.wav", "wb") as fout:
#        fout.write(st.read())

def read(fname,winsize):
    if fname =="-":
        wf=wave.open(sys.stdin,'rb')
        n=wf.getnframes()
        str=wf.readframes(n)
        params = ((wf.getnchannels(), wf.getsampwidth(),
                   wf.getframerate(), wf.getnframes(),
                   wf.getcomptype(), wf.getcompname()))
        siglen=((int )(len(str)/2/winsize) + 1) * winsize
        signal=sp.zeros(siglen, sp.complex)
        signal[0:len(str)/2] = sp.float32(sp.fromstring(str,sp.int16))/32767.0
        return signal,params
    else:
        return read_signal(fname,winsize)

def separate_channels(signal):
    return signal[0::2], signal[1::2] 

_filterbank, _fcenters = melFilterBank(_fs, input_len * 2, banks)

def preprocess(signal):
    p = 0.97         # プリエンファシス係数
    psignal = preEmphasis(signal, p)

    q, mod = divmod(len(psignal),input_len)
    signal2 = signal[0:len(psignal) - mod]

    qq, mod = divmod(len(signal2),input_len / 2)
    for idx in xrange(0, qq-2):
        signal2[idx*(input_len/2):idx*(input_len/2)+input_len] *= _window

    signal2 = np.reshape(signal2, (q, input_len))
#    tmp_arr = np.array([])

    for idx in xrange(0, q):
        signal2[idx] = np.fft.fftpack.fft(signal2[idx])
#        signal[idx] = np.fft.fftpack.fft(signal[idx])
#        tmp_arr = np.r_[tmp_arr, fbank(signal[idx*input_len:(idx+1)*input_len],samplerate=44100,winlen=0.025,winstep=0.01,nfilt=banks,nfft=input_len,lowfreq=0,highfreq=None,preemph=0.97)[0][0]]

    s_amp = np.absolute(signal2)
    s_phase = np.angle(signal2)

    # 振幅スペクトルに対してフィルタバンクの各フィルタをかけ、
    # 振幅の和の対数をとる    
    mspec = []
    for idx in xrange(0, q):
        for c in np.arange(0, banks):
            mspec.append(np.log10(sum(s_amp[idx] * _filterbank[c])))
    s_amp = np.array(mspec)
    
    qq, mod = divmod(len(s_amp), banks)
    s_amp = s_amp[0:len(s_amp) - mod]
    s_amp = np.reshape(s_amp, (qq, banks))

    return s_amp, s_phase

def denoise(signal, model):
    signal_len = len(signal)

    amps, s_phase = preprocess(signal)

    q, mod = divmod(signal_len, input_len)
    q2, mod2 = divmod(signal_len - mod, banks)

    pred_amps = model.predict(amps, batch_size=batch_size)

    pred_amps_ = np.zeros((q, input_len))
    for idx in xrange(0, q):
        for idx2 in xrange(0, banks):
            gen_val = pow(10, pred_amps[idx][idx2])
            sum_val = np.sum(_filterbank[idx2])
            if sum_val == 0:
                continue
            for idx3 in xrange(0, input_len):
                if _filterbank[idx2][idx3] != 0:
                    pred_amps_[idx][idx3] = gen_val * (_filterbank[idx2][idx3] / sum_val)

    pred_amps = pred_amps_

    spec = pred_amps * np.exp(s_phase * 1j)
    for idx in xrange(0, q):
        spec[idx] = np.fft.fftpack.ifft(spec[idx])

    ret = spec.flatten()
    qq, mod3 = divmod(signal_len, input_len/2)
    for idx in xrange(0, qq-2):
        ret[idx*(input_len/2):idx*(input_len/2)+input_len] /= _window

    ret = np.r_[ret, signal[len(signal)-(mod+mod2):len(signal)]]

    return np.real(ret)

def read_signal(filename, winsize):
    wf = wave.open(filename, 'rb')
    n = wf.getnframes()
    st = wf.readframes(n)
    params = ((wf.getnchannels(), wf.getsampwidth(),
               wf.getframerate(), wf.getnframes(),
               wf.getcomptype(), wf.getcompname()))
    siglen = ((int)(len(st) / 2 / winsize) + 1) * winsize
#    signal = np.zeros(siglen, np.float32)
    signal = np.zeros(siglen, np.complex)
    signal[0:int(len(st) / 2)] = np.float32(np.fromstring(st, np.int16)) / 32767.0
    return [signal, params]


if __name__ == '__main__':
    train_input_signal, train_input_params = read("./asakai13_train.wav", 512)
    train_output_signal, train_output_params = read("./asakai13_train_denoised.wav", 512)
    test_input_signal, test_input_params = read("./asakai3_test.wav", 512)
    test_output_signal, test_output_params = read("./asakai3_test_denoised.wav", 512)

    l_input_train,r_input_train = separate_channels(train_input_signal)
    l_output_train,r_output_train = separate_channels(train_output_signal)
    l_input_test,r_input_test = separate_channels(test_input_signal)
    l_output_test,r_output_test = separate_channels(test_output_signal)

#     l_input_train_ = l_input_train[0:1200]
#     l_output_train_ = l_output_train[0:1200]
#     l_input_test_ = l_input_test[0:1200]
#     l_output_test_ = l_output_test[0:1200]
#     r_input_test_ = r_input_test[0:1200]
#     r_input_train_ = r_input_train[0:1200]

    l_input_train_ = l_input_train
    l_output_train_ = l_output_train[0:34437888]
    l_input_test_ = l_input_test
    l_output_test_ = l_output_test[0:9371392]
    r_input_test_ = r_input_test
    r_input_train_ = r_input_train

    _window = sp.hanning(input_len)
    
    model = train(l_input_train_, l_output_train_, l_input_test_, l_output_test_)
    

#     write(test_input_params, uniting_channles(denoise(l_input_test_, model),denoise(r_input_test_, model)))
    write("./dnn_denoised_train.wav", train_input_params, uniting_channles(denoise(l_input_train_, model),denoise(r_input_train_, model)))
    write("./dnn_denoised_test.wav", test_input_params, uniting_channles(denoise(l_input_test_, model),denoise(r_input_test_, model)))
