// -*- coding: utf-8 -*-
var np = require("numjs")
var fs = require("fs")
var math = require("mathjs")
//import scipy as sp
//import sys
//import wave
//import tempfile
//from six.moves import xrange, zip
//import scipy.special as spc

var frame_num = 2646000;
var bufferSize = frame_num * 4;
var all_buffersize = bufferSize + 44;

var _winsize = 512
var _window = null
var _G = null
var _prevGamma = null
var _alpha = null
var _prevAmp = null
var _ratio = null
var _constant = null
var _gamma15 = null

// result doesn't contain index *end*
function slice_nparray(arr,begin,end){
  var ret_len = end - begin
  var ret_arr = np.array(new Array(ret_len))
  for(var i=begin;i<end;i++){
    ret_arr.set(i-begin,arr.get(i))
  }
  return ret_arr
}

function exp_nparray(arr,real,imaginary){
  var arr_len = arr.size
  var ret = np.array(new Array(arr_len))
  var x = math.complex(real, imaginary)
  for(var i=0;i<arr_len;i++){
    ret[i] = math.exp(math.multiply(arr[i],x))
  }
  return ret
}

function noise_reduction(signal,params,winsize,window,ss,ntime){
    var out=np.array(new Array(frame_num))
    var n_pow = compute_avgpowerspectrum(slice_nparray(signal,0,Math.round(params[2]/(1000/ntime))),winsize,window) //maybe 300ms
    var nf = frame_num/(winsize/2) - 1
    var end = Math.round(frame_num/(winsize/2) - 1)
    //for no in xrange(nf):
    for(var no=0;no<end;no++){
        var s = get_frame(signal, winsize, no)
        add_signal(out, compute_by_noise_pow(s,n_pow), winsize, no)
    }
    return out
}

// function write(param,signal){
//     st = tempfile.TemporaryFile()
//     wf=wave.open(st,'wb')
//     wf.setparams(params)
//     s=sp.int16(signal*32767.0).tostring()
//     wf.writeframes(s)
//     st.seek(0)
//     print st.read()
// }
//
// function read(fname,winsize){
//     if fname =="-":
//         wf=wave.open(sys.stdin,'rb')
//         n=wf.getnframes()
//         str=wf.readframes(n)
//         params = ((wf.getnchannels(), wf.getsampwidth(),
//                    wf.getframerate(), wf.getnframes(),
//                    wf.getcomptype(), wf.getcompname()))
//         siglen=((int )(len(str)/2/winsize) + 1) * winsize
//         signal=sp.zeros(siglen, sp.float32)
//         signal[0:len(str)/2] = sp.float32(sp.fromstring(str,sp.int16))/32767.0
//         return signal,params
//     else:
//         return read_signal(fname,winsize)
// }

function __init__(winsize, window, constant, ratio, alpha){
    _window = window
    _G = np.array(new Array(winsize))
    _prevGamma = np.array(new Array(winsize))
    _alpha = alpha
    _prevAmp = np.array(new Array(winsize))
    _ratio = ratio
    _constant = constant
    _gamma15 = math.gamma(1.5)
}

function my_angle(ndarr){
  var arr_len = ndarr.size
  var ret_arr = np.array(new Array(arr_len))
  for(var i=0;i<arr_len;i++){
    ret_arr.set(i, Math.atan(ndarr.get(i))*2)
  }
  return ret_arr
}

function compute_by_noise_pow(signal, n_pow){
//    s_spec = np.fft.fftpack.fft(signal * _window)
    var s_spec = my_fft(signal.multiply(_window),_winsize)
    var s_amp = np.abs(s_spec)
    var s_phase = my_angle(s_spec)
    var gamma = _calc_aposteriori_snr(s_amp, n_pow)
    var xi = _calc_apriori_snr(gamma)
    _prevGamma = gamma
    var nu = gamma.multiply(xi).divide((xi.add(1.0)))
    _G = (_gamma15.multiply(np.sqrt(nu)).divide(gamma)).multiply(np.exp(nu.multiply(-1.0).divide(2.0)))
          .multiply((nu.add(1.0).multiply(spc.i0(nu.divide(2.0)))).add(nu.multiply(spc.i1(nu.divide(2.0)))))
    var idx = np.less(s_amp.multiply(s_amp), n_pow)
    _G[idx] = _constant
    idx = np.isnan(_G) + np.isinf(_G)
    _G[idx] = xi[idx] / (xi[idx] + 1.0)
    idx = np.isnan(_G) + np.isinf(_G)
    _G[idx] = _constant
    _G = np.maximum(_G, 0.0)
    var amp = _G.multiply(s_amp)
    amp = np.maximum(amp, 0.0)
    var amp2 = amp.multiply(_ratio).add(s_amp.multiply(1.0 - _ratio))
    _prevAmp = amp
    var spec = amp2.multiply(exp_nparray(s_phase,0,1))
//    return np.real(np.fft.fftpack.ifft(spec))
    return np.real(np.ifft(spec))
}

// function _sigmoid(gain){
//     for i in xrange(len(gain)):
//         gain[i] = sigmoid(gain[1], 1, 2, _gain)
// }

function compute(signal, noise){
//    n_spec = np.fft.fftpack.fft(noise * _window)
    var n_spec = my_fft(noise.multiply(_window),_winsize)
//    n_pow = np.abs(n_spec) ** 2.0
    var n_pow = np.abs(n_spec).multiply(np.abs(n_spec))
    return compute_by_noise_pow(signal, n_pow)
}

function _calc_aposteriori_snr(s_amp, n_pow){
    //return s_amp ** 2.0 / n_pow
    return s_amp.multiply(s_amp).divide(n_pow)
}

function _calc_apriori_snr(gamma){
    // return _alpha * _G ** 2.0 * _prevGamma +\
    //     (1.0 - _alpha) * np.maximum(gamma - 1.0, 0.0)  # a priori s/n ratio
    return _G.multiply(_G).multiply(_prevGamma).multiply(_alpha).add(
        np.maximum(gamma - 1.0, 0.0).add(1.0 - _alpha))  // a priori s/n ratio
}

// function _calc_apriori_snr2(gamma, n_pow){
//     return _alpha * (_prevAmp ** 2.0 / n_pow) +\
//         (1.0 - _alpha) * np.maximum(gamma - 1.0, 0.0)  # a priori s/n ratio
// }

// function read_signal(filename, winsize){
//     wf = wave.open(filename, 'rb')
//     n = wf.getnframes()
//     st = wf.readframes(n)
//     params = ((wf.getnchannels(), wf.getsampwidth(),
//                wf.getframerate(), wf.getnframes(),
//                wf.getcomptype(), wf.getcompname()))
//     siglen = ((int)(len(st) / 2 / winsize) + 1) * winsize
//     signal = np.zeros(siglen, np.float32)
//     signal[0:int(len(st) / 2)] = np.float32(np.fromstring(st, np.int16)) / 32767.0
//     return [signal, params]
// }

function get_frame(signal, winsize, no){
    var shift = Math.round(winsize / 2)
    var start = Math.round(no * shift)
    var end = start + winsize
    return slice_nparray(signal,start,end)
}

function add_signal(signal, frame, winsize, no){
    var shift = Math.round(winsize / 2)
    var start = Math.round(no * shift)
    var end = start + winsize
    var sliced_nparr = slice_nparray(signal,start,end).add(frame)
    for(var i=start;i<end;i++){
      signal[i] = sliced_nparr[i-start]
    }
}

// function write_signal(filename, params, signal){
//     wf = wave.open(filename, 'wb')
//     wf.setparams(params)
//     s = np.int16(signal * 32767.0).tostring()
//     wf.writeframes(s)
// }

function get_window(winsize, no){
    var shift = Math.round(winsize / 2)
    var s = Math.round(no * shift)
    return (s, s + winsize)
}

// function separate_channels(signal){
//     return signal[0::2], signal[1::2]
// }

function compute_avgamplitude(signal, winsize, window){
    var windownum = Math.round(frame_num / (winsize / 2)) - 1
    var avgamp = np.array(new Array(winsize))
    for(var l=0;l<windownum;l++){
        avgamp.add(np.abs(my_fft(get_frame(signal, winsize, l).multiply(window)),winsize))
    }
    return avgamp.divide(windownum * 1.0)
}

function my_fft(ndarr,input_len){
  var fft_arr = []
  for(var i=0;i<input_len;i++){
    fft_arr.push([ndarr.get(i),0])
  }
  var tmp = np.fft(np.array(fft_arr))
  return tmp.slice(0,1).flatten()
}

function compute_avgpowerspectrum(signal, winsize, window){
    var windownum = Math.round(frame_num / (winsize / 2)) - 1
    var avgpow = np.array(new Array(winsize))
    for(var l=1;l<windownum;l++){
//        var tmp = np.abs(np.fft(get_frame(signal, winsize, l).multiply(window)))
        var real_arr = get_frame(signal, winsize, l).multiply(window)
        var tmp = np.abs(my_fft(real_arr,winsize))
        avgpow.add(tmp.multiply(tmp))
    }
    return avgpow.divide(windownum * 1.0)
}

// function sigmoid(x, x0, k, a){
//     y = k * 1 / (1 + np.exp(-a * (x - x0)))
//     return y
// }

// function calc_kurtosis(samples){
//     n = len(samples)
//     avg = np.average(samples)
//     moment2 = np.sum((samples - avg) ** 2) / n
//     s_sd = np.sqrt(((n / (n - 1)) * moment2))
//     k = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((samples - avg) / s_sd) ** 4)
//     return k - 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
// }

function generate_hann_arr(M){
  var ret_arr = new Array(M)
  for(i=0;i<M;i++){
    ret_arr[i] = 0.5 - 0.5*Math.cos(2*Math.PI*i/(M-1))
  }

  return np.array(ret_arr)
}

var base_arr = new Array(frame_num)
var signal = np.array(base_arr)
for(var i=0;i<frame_num;i++){
  signal.set(i,0)
}

// var sample_str = fs.readFileSync('./sample60.txt',"ascii");
// var splited = sample_str.split(",");
// for(var i=0;i<frame_num;i++){
//   signal[i] = Number(splited[i]);
// }

    //signal, params = read("./tools/asakai60.wav", 512)
    //     params = ((wf.getnchannels(), wf.getsampwidth(),
    //                wf.getframerate(), wf.getnframes(),
    //                wf.getcomptype(), wf.getcompname()))
var params = [2,-1,44100]

//    var window = sp.hanning(512)
var window = generate_hann_arr(_winsize)

__init__(_winsize,window,1.0,0.001,0.99)

var output = noise_reduction(signal,params,_winsize,window,null,300)

//    if False:
//        write(params, noise_reduction(signal,512,window,null,300))
//    elif True:
//        l,r = separate_channels(signal)
//        write(params, uniting_channles(noise_reduction(l,params,512,window,null,300),noise_reduction(r,params,512,window,null,300)))

var result_str = "";
for(var i=0;i<output.length;i++){
  if(i==0){
    result_str = String(output[i]) + "," + String(output[i]);
  }else{
    result_str = result_str + "," + String(output[i]) + "," + String(output[i]);
  }
}

var wstream = fs.createWriteStream('./samples60_denoised.txt');
wstream.write(result_str, (err) => {
  if (err) throw err;
    console.log('The file has been saved!');
});
