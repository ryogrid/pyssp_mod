#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import sys
import wave
import tempfile
from six.moves import xrange, zip
import scipy.special as spc

def write(param,signal):
    st = tempfile.TemporaryFile()
    wf=wave.open(st,'wb')
    wf.setparams(params)
    s=sp.int16(signal*32767.0).tostring()
    wf.writeframes(s)
    st.seek(0)
    print st.read()

def read(fname,winsize):
    if fname =="-":
        wf=wave.open(sys.stdin,'rb')
        n=wf.getnframes()
        str=wf.readframes(n)
        params = ((wf.getnchannels(), wf.getsampwidth(),
                   wf.getframerate(), wf.getnframes(),
                   wf.getcomptype(), wf.getcompname()))
        siglen=((int )(len(str)/2/winsize) + 1) * winsize
        signal=sp.zeros(siglen, sp.float32)
        signal[0:len(str)/2] = sp.float32(sp.fromstring(str,sp.int16))/32767.0
        return signal,params
    else:
        return read_signal(fname,winsize)

def denoise(signal):
    s_spec = np.fft.fftpack.fft(signal * _window)
    s_amp = np.absolute(s_spec)
    s_phase = np.angle(s_spec)


    spec = s_amp * np.exp(s_phase * 1j)
    return np.real(np.fft.fftpack.ifft(spec))

def read_signal(filename, winsize):
    wf = wave.open(filename, 'rb')
    n = wf.getnframes()
    st = wf.readframes(n)
    params = ((wf.getnchannels(), wf.getsampwidth(),
               wf.getframerate(), wf.getnframes(),
               wf.getcomptype(), wf.getcompname()))
    siglen = ((int)(len(st) / 2 / winsize) + 1) * winsize
    signal = np.zeros(siglen, np.float32)
    signal[0:int(len(st) / 2)] = np.float32(np.fromstring(st, np.int16)) / 32767.0
    return [signal, params]

if __name__ == '__main__':
    input_signal, input_params = read("./asakai13_train.wav", 512)
    output_signal, output_params = read("./asakai3_test.wav", 512)

    window = sp.hanning(512)
    import os.path

    __init__(512,window,ratio=1.0,constant=0.001,alpha=0.99)

    if False:
        write(params, noise_reduction(signal,512,window,None,300))
    elif True:
        l,r = separate_channels(signal)
        write(params, uniting_channles(noise_reduction(l,params,512,window,None,300),noise_reduction(r,params,512,window,None,300)))
