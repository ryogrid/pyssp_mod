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


from keras.layers import Dense
from keras.models import Model, Sequential
import os.path

input_len = 120
hidden_dim = 300
batch_size = 256
epocs = 20

def uniting_channles(leftsignal, rightsignal):
    ret = []
    for i, j in zip(leftsignal, rightsignal):
        ret.append(i)
        ret.append(j)
#    return np.array(ret, np.float32)
    return np.array(ret, np.complex)

def train(train_in, train_out, test_in, test_out):
    model = Sequential()

    model.add(Dense(input_len, activation='relu', input_shape=(input_len,)))
    model.add(Dense(hidden_dim, activation='relu'))
    model.add(Dense(input_len,  activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy')

    x_train_in, _ = preprocess(train_in)
    x_train_out, _ = preprocess(train_out)
    y_test_in, _ = preprocess(test_in)
    y_test_out, _ = preprocess(test_out)

    model.fit(x_train_in, x_train_out,
                nb_epoch=epocs,
                batch_size=batch_size,
                shuffle=False,
                validation_data=(y_test_in, y_test_out))

    #model.save_weights('./denoise.weight')
    #autoencoder.load_weights('autoencoder.h5')

    return model


def write(params,signal):
#    st = tempfile.TemporaryFile()
#    wf=wave.open(st,'wb')
    wf=wave.open("./dnn_denoised.wav",'wb')
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

def preprocess(signal):
    q, mod = divmod(len(signal),input_len)
    signal = signal[0:len(signal) - mod]
    signal = np.reshape(signal, (q, input_len))

    for idx in xrange(0, q):
        signal[idx] = np.fft.fftpack.fft(signal[idx])

    s_amp = np.absolute(signal)
    s_phase = np.angle(signal)
    return s_amp, s_phase

def denoise(signal, model):
    signal_len = len(signal)

    amps, s_phase = preprocess(signal)

    q, mod = divmod(signal_len,input_len)

    pred_amps = model.predict(amps, batch_size=batch_size)

    spec = pred_amps * np.exp(s_phase * 1j)
    for idx in xrange(0, q):
        spec[idx] = np.fft.fftpack.ifft(spec[idx])

    ret = spec.flatten()
    ret = np.r_[ret, signal[len(signal)-mod:len(signal)]]

    return ret

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
#    print(len(l_input_train))
    l_output_train,r_output_train = separate_channels(train_output_signal)
#    print(len(l_output_train))
    l_input_test,r_input_test = separate_channels(test_input_signal)
#    print(len(l_input_test))
    l_output_test,r_output_test = separate_channels(test_output_signal)
#    print(len(l_output_test))
    
#     l_input_train_ = l_input_train[0:1000]
#     l_output_train_ = l_output_train[0:1000]
#     l_input_test_ = l_input_test[0:1000]
#     l_output_test_ = l_output_test[0:1000]
#     r_input_test_ = r_input_test[0:1000]

    l_input_train_ = l_input_train
    l_output_train_ = l_output_train[0:34437888]
    l_input_test_ = l_input_test
    l_output_test_ = l_output_test[0:9371392]
    r_input_test_ = r_input_test
    
    model = train(l_input_train_, l_output_train_, l_input_test_, l_output_test_)

    write(test_input_params, uniting_channles(denoise(l_input_test_, model),denoise(r_input_test_, model)))
