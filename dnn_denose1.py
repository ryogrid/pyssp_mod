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


from keras.layers import Input, Dense
from keras.models import Model, Sequential
import os.path

def uniting_channles(leftsignal, rightsignal):                                                                                                                                                               ret = []
    for i, j in zip(leftsignal, rightsignal):
        ret.append(i)
        ret.append(j)
    return np.array(ret, np.float32)

def train(train_in, train_out, test_in, test_out):
    model = Sequential()

    input_len = 120
    hidden_dim = 300

    model.add(Dence(input_len, activation='relu', input_shape=(input_len,1)))
    model.add(Dence(hidden_dim, activation='relu'))
    model.add(Dence(input_len, activation='relu'))

    model.compile(optimizer='adam', loss='binary_crossentropy')

    #autoencoder.save_weights('autoencoder.h5')
    #autoencoder.load_weights('autoencoder.h5')

    q, mod = divmode(len(train_in),input_len)
    train_in = train_in[0:len(train_in) - mod]
    x_train_in = np.reshape(train_in, (q, input_len))

    q, mod = divmode(len(train_out),input_len)
    train_out = train_out[0:len(train_out) - mod]
    x_train_out = np.reshape(train_out, (q, input_len))

    q, mod = divmode(len(test_in),input_len)
    test_in = test_in[0:len(test_inl) - mod]
    y_test_in = np.reshape(test_in, (q, input_len))

    q, mod = divmode(len(test_out),input_len)
    test_out = train_out[0:len(test_out) - mod]
    y_test_out = np.reshape(test_out, (q, input_len))


    model.fit(x_train_in, x_train_out,
                nb_epoch=1,
                batch_size=256,
                shuffle=True,
                validation_data=(y_test_in, y_test_out))

   return model


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

def separate_channels(signal):
    return signal[0::2], signal[1::2] 

def preprocess(signal):
    s_spec = np.fft.fftpack.fft(signal)
    s_amp = np.absolute(s_spec)
    return s_amp

def denoise(signal, model):
    signal_len = len(signal)

    s_spec = np.fft.fftpack.fft(signal)
    s_amp = np.absolute(s_spec)
    s_phase = np.angle(s_spec)

    input_len = 120

    q, mod = divmode(len(signal),input_len)
    input = signal[0:len(signal) - mod]
    input = np.reshape(input, (q, input_len))
    
    pred = model.predict(input, batch_size=256)
    s_phase = np.reshape(pred, (q*input_len))
    s_phase = np.r_(s_pase, signal[len(signal)-q:len(signal)])

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
    train_input_signal, train_input_params = read("./asakai13_train.wav", 512)
    train_output_signal, train_output_params = read("./asakai13_train_denoised.wav", 512)
    test_input_signal, test_input_params = read("./asakai3_test.wav", 512)
    test_output_signal, test_output_params = read("./asakai3_test_denoised.wav", 512)

    l_input_train,r_input_train = separate_channels(train_input_signal)
    l_output_train,r_output_train = separate_channels(train_output_signal)
    l_input_test,r_input_test = separate_channels(test_input_signal)
    l_output_test,r_output_test = separate_channels(test_output_signal)

    l_input_train_ = prerocess(l_input_train)
    l_output_train_ = preprocess(l_output_train)
    l_input_test_ = preprocess(l_input_test)
    l_output_test_ = preprocess(l_output_test)
    
    model = train(l_input_train_, l_output_train_, l_input_test_, l_output_test_)
    write(test_input_params, uniting_channles(denoise(l_input_test, model),denoise(r_input_test, model))
