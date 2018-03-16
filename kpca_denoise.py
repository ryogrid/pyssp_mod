np.complex#!/usr/bin/env python
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
import scipy.signal

import os.path

from our_kpca import kPCA

_input_len = 512
_n_reconstruct = 180
_sigma = 0.5
_window = None

def uniting_channles(leftsignal, rightsignal):
    ret = []
    for i, j in zip(leftsignal, rightsignal):
        ret.append(i)
        ret.append(j)
    return np.array(ret, np.complex)

def write(fname, params,signal):
    wf=wave.open(fname,'wb')
    wf.setparams(params)
    s=sp.int16(signal*32767.0).tostring()
    wf.writeframes(s)
    wf.close()

def read(fname,winsize):
    if fname =="-":
        wf=wave.open(sys.stdin,'rb')
        n=wf.getnframes()
        str=wf.readframes(n)
        params = ((wf.getnchannels(), wf.getsampwidth(),
                   wf.getframerate(), wf.getnframes(),
                   wf.getcomptype(), wf.getcompname()))
        siglen=((int )(len(str)/2/winsize) + 1) * winsize
        signal=sp.zeros(siglen, np.complex)
        signal[0:len(str)/2] = sp.float32(sp.fromstring(str,sp.int16))/32767.0
        return signal,params
    else:
        return read_signal(fname,winsize)

def separate_channels(signal):
    return signal[0::2], signal[1::2]


def preprocess(signal):
    q, mod = divmod(len(signal),_input_len)
    signal2 = signal[0:len(signal) - mod]
    signal2 = np.reshape(signal2, (q, _input_len))

    for idx in xrange(0, q):
        signal2[idx] = np.fft.fftpack.fft(signal2[idx])

    s_amp = np.absolute(signal2)
    s_amp2 = np.zeros((_input_len, q), np.complex)
    for jj in xrange(0, _input_len):
        for ii in xrange(0, q):
            s_amp2[jj][ii] = s_amp[ii][jj]

    s_phase = np.angle(signal2)
    s_phase2 = np.zeros((_input_len, q), np.complex)
    for jj in xrange(0, _input_len):
        for ii in xrange(0, q):
            s_phase2[jj][ii] = s_phase[ii][jj]

    return s_amp2, s_phase2

def denoise(signal_train, signal_test):
    amps_train, s_phase_train = preprocess(signal_train)
    amps_test, s_phase_test = preprocess(signal_test)

    kpca = kPCA(amps_train, amps_test)
    denoised_amp = kpca.obtain_preimages(_n_reconstruct, _sigma)
    q, mod = divmod(len(signal_test),_input_len)
    denoised_amp2 = np.zeros((_input_len, q), np.complex)
    for jj in xrange(0, _input_len):
        for ii in xrange(0, q):
            denoised_amp2[ii][jj] = denoised_amp[jj][ii]

    denoised_spec = denoised_amp2 * np.exp(s_phase_test * 1j)
    for idx in xrange(0, q):
        denoised_spec[idx] = np.fft.fftpack.ifft(denoised_spec[idx])

    ret = denoised_spec.flatten()
    ret = np.r_[ret, signal_test[len(signal_test)-mod:len(signal_test)]]
    return ret

def read_signal(filename, winsize):
    wf = wave.open(filename, 'rb')
    n = wf.getnframes()
    st = wf.readframes(n)
    params = ((wf.getnchannels(), wf.getsampwidth(),
               wf.getframerate(), wf.getnframes(),
               wf.getcomptype(), wf.getcompname()))
    siglen = ((int)(len(st) / 2 / winsize) + 1) * winsize
    signal = np.zeros(siglen, np.complex)
    signal[0:int(len(st) / 2)] = np.float32(np.fromstring(st, np.int16)) / 32767.0
    return [signal, params]


if __name__ == '__main__':
    train_input_signal, train_input_params = read("./asakai13_train.wav", _input_len)
    test_input_signal, test_input_params = read("./asakai3_test.wav", _input_len)

    l_input_train,r_input_train = separate_channels(train_input_signal)
    l_input_test,r_input_test = separate_channels(test_input_signal)

    l_input_train_ = l_input_train[0:1200]
    l_input_test_ = l_input_test[0:1200]

    # l_input_train_ = l_input_train
    # l_input_test_ = l_input_test
    # r_input_train_ = r_input_train
    # r_input_test_ = r_input_test

    denoised_signal = denoise(l_input_train_, l_input_test_)
    write("./dnn_denoised_test.wav", test_input_params, uniting_channles(denoised_signal, denoised_signal))
