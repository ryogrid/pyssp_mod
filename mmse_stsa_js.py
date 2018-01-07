#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import sys
import wave
import tempfile
from six.moves import xrange, zip
import scipy.special as spc

_window = None
_G = None
_prevGamma = None
_alpha = None
_prevAmp = None
_ratio = None
_constant = None
_gamma15 = None


def noise_reduction(signal,params,winsize,window,ss,ntime):
    out=sp.zeros(len(signal),sp.float32)
    n_pow = compute_avgpowerspectrum(signal[0:winsize*int(params[2] /float(winsize)/(1000.0/ntime))],winsize,window)#maybe 300ms
    nf = len(signal)/(winsize/2) - 1
    for no in xrange(nf):
        s = get_frame(signal, winsize, no)
        add_signal(out, compute_by_noise_pow(s,n_pow), winsize, no)
    return out


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


def __init__(winsize, window, constant=0.001, ratio=1.0, alpha=0.99):
    global _window
    global _G
    global _prevGamma
    global _alpha
    global _prevAmp
    global _ratio
    global _constant
    global _gamma15

    _window = window
    _G = np.zeros(winsize, np.float64)
    _prevGamma = np.zeros(winsize, np.float64)
    _alpha = alpha
    _prevAmp = np.zeros(winsize, np.float64)
    _ratio = ratio
    _constant = constant
    _gamma15 = spc.gamma(1.5)

def compute_by_noise_pow(signal, n_pow):
    global _window
    global _G
    global _prevGamma
    global _alpha
    global _prevAmp
    global _ratio
    global _constant
    global _gamma15

    s_spec = np.fft.fftpack.fft(signal * _window)
    s_amp = np.absolute(s_spec)
    s_phase = np.angle(s_spec)
    gamma = _calc_aposteriori_snr(s_amp, n_pow)
    xi = _calc_apriori_snr(gamma)
    print("xi:" + str(type(xi)))
    _prevGamma = gamma
    nu = gamma * xi / (1.0 + xi)
    _G = (_gamma15 * np.sqrt(nu) / gamma) * np.exp(-nu / 2.0) *\
              ((1.0 + nu) * spc.i0(nu / 2.0) + nu * spc.i1(nu / 2.0))
    idx = np.less(s_amp ** 2.0, n_pow)
    _G[idx] = _constant
    idx = np.isnan(_G) + np.isinf(_G)
    _G[idx] = xi[idx] / (xi[idx] + 1.0)
    idx = np.isnan(_G) + np.isinf(_G)
    _G[idx] = _constant
    _G = np.maximum(_G, 0.0)
    amp = _G * s_amp
    amp = np.maximum(amp, 0.0)
    amp2 = _ratio * amp + (1.0 - _ratio) * s_amp
    _prevAmp = amp
    spec = amp2 * np.exp(s_phase * 1j)
    return np.real(np.fft.fftpack.ifft(spec))

def _sigmoid(gain):
    for i in xrange(len(gain)):
        gain[i] = sigmoid(gain[1], 1, 2, _gain)

def compute(signal, noise):
    n_spec = np.fft.fftpack.fft(noise * _window)
    n_pow = np.absolute(n_spec) ** 2.0
    return compute_by_noise_pow(signal, n_pow)

def _calc_aposteriori_snr(s_amp, n_pow):
    return s_amp ** 2.0 / n_pow

def _calc_apriori_snr(gamma):
    return _alpha * _G ** 2.0 * _prevGamma +\
        (1.0 - _alpha) * np.maximum(gamma - 1.0, 0.0)  # a priori s/n ratio

def _calc_apriori_snr2(gamma, n_pow):
    return _alpha * (_prevAmp ** 2.0 / n_pow) +\
        (1.0 - _alpha) * np.maximum(gamma - 1.0, 0.0)  # a priori s/n ratio


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


def get_frame(signal, winsize, no):
    shift = int(winsize / 2)
    start = int(no * shift)
    end = start + winsize
    return signal[start:end]


def add_signal(signal, frame, winsize, no):
    shift = int(winsize / 2)
    start = int(no * shift)
    end = start + winsize
    signal[start:end] = signal[start:end] + frame


def write_signal(filename, params, signal):
    wf = wave.open(filename, 'wb')
    wf.setparams(params)
    s = np.int16(signal * 32767.0).tostring()
    wf.writeframes(s)


def get_window(winsize, no):
    shift = int(winsize / 2)
    s = int(no * shift)
    return (s, s + winsize)


def separate_channels(signal):
    return signal[0::2], signal[1::2]


def uniting_channles(leftsignal, rightsignal):
    ret = []
    for i, j in zip(leftsignal, rightsignal):
        ret.append(i)
        ret.append(j)
    return np.array(ret, np.float32)


def compute_avgamplitude(signal, winsize, window):
    windownum = int(len(signal) / (winsize / 2)) - 1
    avgamp = np.zeros(winsize)
    for l in xrange(windownum):
        avgamp += np.absolute(sp.fft(get_frame(signal, winsize, l) * window))
    return avgamp / float(windownum)


def compute_avgpowerspectrum(signal, winsize, window):
    windownum = int(len(signal) / (winsize / 2)) - 1
    avgpow = np.zeros(winsize)
    for l in xrange(windownum):
        avgpow += np.absolute(sp.fft(get_frame(signal, winsize, l) * window))**2.0
    return avgpow / float(windownum)


def sigmoid(x, x0, k, a):
    y = k * 1 / (1 + np.exp(-a * (x - x0)))
    return y


def calc_kurtosis(samples):
    n = len(samples)
    avg = np.average(samples)
    moment2 = np.sum((samples - avg) ** 2) / n
    s_sd = np.sqrt(((n / (n - 1)) * moment2))
    k = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((samples - avg) / s_sd) ** 4)
    return k - 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3))


if __name__ == '__main__':
    signal, params = read("./asakai60.wav", 512)

    window = sp.hanning(512)
    import os.path

    __init__(512,window,ratio=1.0,constant=0.001,alpha=0.99)

    if False:
        write(params, noise_reduction(signal,512,window,None,300))
    elif True:
        l,r = separate_channels(signal)
        write(params, uniting_channles(noise_reduction(l,params,512,window,None,300),noise_reduction(r,params,512,window,None,300)))
