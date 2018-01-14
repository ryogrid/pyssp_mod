// -*- coding: utf-8 -*-
var np = require("numjs")
var fs = require("fs")
var math = require("mathjs")
var BESSEL = require('bessel')
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

function op2_nparray(op1,op2,fnk){
    var arr_len = op1.size
    var is_op2_arr = false
    var ret_arr = np.array(new Array(arr_len))
    if(op2 instanceof np.NdArray){
      is_op2_arr = true
    }
    for(var i=0;i<arr_len;i++){
      var op2_val = op2
      if(is_op2_arr){
        op2_val = op2.get(i)
      }
      ret_arr.set(i,fnk(op1.get(i),op2_val))
      //console.log(fnk(op1.get(i),op2_val))
    }
    return ret_arr
}

function op1_nparray(op1,fnk){
    var arr_len = op1.size
    var is_op1_arr = false
    var ret_arr = np.array(new Array(arr_len))
    if(op1 instanceof np.NdArray){
      is_op1_arr = true
    }
    for(var i=0;i<arr_len;i++){
      var op1_val = op1
      if(is_op1_arr){
        op1_val = op1.get(i)
      }
      ret_arr.set(i,fnk(op1.get(i)))
    }
    return ret_arr
}

function my_abs(op1){
    return op1_nparray(op1,math.abs)
}

function my_exp(op1){
    return op1_nparray(op1,math.exp)
}

function my_sqrt(op1){
    return op1_nparray(op1,math.sqrt)
}

function my_multiply(op1,op2){
    return op2_nparray(op1,op2,math.multiply)
}

function my_add(op1,op2){
    return op2_nparray(op1,op2,math.add)
}

function my_division(op1,op2){
    return op2_nparray(op1,op2,math.divide)
}

function my_real(op1){
    return op1_nparray(op1,function(val){return val.re})
}

// result doesn't contain index *end*
function slice_nparray(arr,begin,end){
  var ret_len = end - begin
  var ret_arr = np.array(new Array(ret_len))
  for(var i=begin;i<end;i++){
    //console.log(arr.get(i))
    ret_arr.set(i-begin,arr.get(i))
  }
  // console.log(end)
  // console.log(arr.get(end-1))
  // console.log(arr.get(end-2))
  // console.log(arr.get(end-3))
  // console.log(arr.get(end-4))
  // console.log(arr.get(end-5))
  return ret_arr
}

function mul_exp_nparray(arr,real,imaginary){
  var arr_len = arr.size
  var ret = np.array(new Array(arr_len))
  var x = math.complex(real, imaginary)
  for(var i=0;i<arr_len;i++){
    var tmp = math.exp(math.multiply(arr.get(i),x))
    //var tmp = math.exp(arr.get(i))
    //console.log(arr.get(i))
    //console.log(math.multiply(arr.get(i),x))
    //console.log(tmp)
    ret.set(i,tmp)
    //console.log(math.exp(math.multiply(arr.get(i),x)))
  }
  return ret
}

function maximum_nparray(arr,val){
  var arr_len = arr.size
  var ret = np.array(new Array(arr_len))
  for(var i=0;i<arr_len;i++){
    ret.set(i,math.max(arr.get(i),val))
  }
  return ret
}

function i0_nparray(arr){
  var arr_len = arr.size
  var ret = np.array(new Array(arr_len))
  for(var i=0;i<arr_len;i++){
    ret.set(i,BESSEL.besseli(arr.get(i),0))
  }
  //console.log(ret)
  return ret
}

function i1_nparray(arr){
  var arr_len = arr.size
  var ret = np.array(new Array(arr_len))
  for(var i=0;i<arr_len;i++){
    ret.set(i,BESSEL.besseli(arr.get(i),1))
  }
  //console.log(ret)
  return ret
}

function less_nparray(arr1,arr2){
  var arr_len = arr1.size
  var ret_arr = np.array(new Array(arr_len))

  for(var i=0;i<arr_len;i++){
    if(arr1.get(i) < arr2.get(i)){
      ret_arr.set(i,true)
    }else {
      ret_arr.set(i,false)
    }
  }

  return ret_arr
}

function set_with_bool_nparray(arr,bool_arr,val){
  var arr_len = arr.size
  var tmp_arr = []

  for(var i=0;i<arr_len;i++){
    if(bool_arr.get(i)){
      arr.set(i,val)
    }
  }
}

function copy_with_bool_nparray(arr1,arr2,bool_arr){
  var arr_len = arr1.size
  var tmp_arr = []

  for(var i=0;i<arr_len;i++){
    if(bool_arr.get(i)){
      arr1.set(i,arr2.get(i))
    }
  }
}

function fill_nparray(arr,val){
  var arr_len = arr.size
  var ret = np.array(new Array(arr_len))
  for(var i=0;i<arr_len;i++){
    ret.set(i,val)
  }
  return ret
}

function isnan_nparray(arr){
  var arr_len = arr.size
  var ret = np.array(new Array(arr_len))
  for(var i=0;i<arr_len;i++){
    if(Number.isNaN(arr.get(i))){
      ret.set(i,true)
    }else{
      ret.set(i,false)
    }
  }
  return ret
}

function isinf_nparray(arr){
  var arr_len = arr.size
  var ret = np.array(new Array(arr_len))
  for(var i=0;i<arr_len;i++){
    if(arr.get(i) == Infinity){
      ret.set(i,true)
    }else{
      ret.set(i,false)
    }
  }
  return ret
}

function noise_reduction(signal,params,winsize,window,ss,ntime){
    var out=np.array(new Array(frame_num))
    for(var i=0;i<frame_num;i++){
      out.set(i,0.0)
    }
    // console.log(signal.size) // signal.size is ok
    var n_pow = compute_avgpowerspectrum(slice_nparray(signal,0,winsize*Math.round(params[2]/winsize/(1000.0/ntime))),winsize,window) //maybe 300ms
    //console.log(n_pow) // bad values array
    var nf = frame_num/(winsize/2) - 1
    var end = Math.round(frame_num/(winsize/2) - 1)
    //for no in xrange(nf):
    var shift = Math.round(winsize / 2)
    for(var no=0;no<end;no++){
        //console.log("noise_reduction:" + String(no))
        var slice_start = Math.round(no * shift)
        var slice_end = slice_start + winsize
        if(slice_end <= frame_num){
          var s = get_frame(signal, winsize, no)
          add_signal(out, compute_by_noise_pow(s,n_pow), winsize, no)
        }
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
    //ret_arr.set(i, math.atan(ndarr.get(i))*2)
    var val = ndarr.get(i)
    var atan_ret = null
    if(val.im != 0){
      atan_ret = Math.atan(val.re/val.im)
    }else {
      atan_ret = 0.0
    }

    ret_arr.set(i,atan_ret)
    //console.log(ret_arr.get(i))
  }
  return ret_arr
}

// function compute_by_noise_pow(signal, n_pow){
//     //console.log(signal)
//     var s_spec = my_fft(signal,signal.size)
//     var ret = my_ifft(s_spec,s_spec.size)
//     return my_real(ret)
// }

function compute_by_noise_pow(signal, n_pow){
    //console.log(signal.multiply(_window))
    var s_spec = my_fft(signal.multiply(_window),signal.size)
    //console.log(s_spec) //error
    var s_amp = my_abs(s_spec)
    var s_phase = my_angle(s_spec)
    //console.log(s_phase)  // zero only arr
    var gamma = _calc_aposteriori_snr(s_amp, n_pow)
    var xi = _calc_apriori_snr(gamma)
    //console.log(s_amp)
    //console.log(xi)
    _prevGamma = gamma
    var nu = gamma.multiply(xi).divide((xi.add(1.0)))
    //console.log(nu)
    _G = (np.sqrt(nu).multiply(_gamma15).divide(gamma)).multiply(np.exp(nu.multiply(-1.0).divide(2.0)))
          .multiply((nu.add(1.0).multiply(i0_nparray(nu.divide(2.0)))).add(nu.multiply(i1_nparray(nu.divide(2.0)))))
    //console.log(_G)
    var idx = less_nparray(s_amp.multiply(s_amp), n_pow)

    // _G[idx] = _constant
    // idx = np.isnan(_G) + np.isinf(_G)
    // _G[idx] = xi[idx] / (xi[idx] + 1.0)
    // idx = np.isnan(_G) + np.isinf(_G)
    // _G[idx] = _constant

    set_with_bool_nparray(_G,idx,_constant)
    idx = isnan_nparray(_G).add(isinf_nparray(_G))
//    var xi_len = xi.size
    var xi_len = xi.size
    for(var i=0;i<xi_len;i++){
      if(idx.get(i)){
          xi.set(i,xi.get(i)/(xi.get(i)+1.0))
      }
    }
    copy_with_bool_nparray(_G,xi,idx)
    idx = isnan_nparray(_G).add(isinf_nparray(_G))
    set_with_bool_nparray(_G,idx,_constant)

    _G = maximum_nparray(_G, 0.0)
    //console.log(_G)
    var amp = _G.multiply(s_amp)
    amp = maximum_nparray(amp, 0.0)
    var amp2 = amp.multiply(_ratio).add(s_amp.multiply(1.0 - _ratio))
    //console.log(amp2) //OK
    _prevAmp = amp
    //console.log(s_phase)
    //var spec = amp2.multiply(mul_exp_nparray(s_phase,0,1))
    var mul_exp = mul_exp_nparray(s_phase,0,1)
    var spec = my_multiply(amp2,mul_exp)
    // for(var i=0;i<_winsize;i++){
    //   var amp2_val = amp2.get(i)
    //   var mul_exp_val = mul_exp.get(i)
    //   spec.push([amp2_val*mul_exp_val[0],amp2_val*mul_exp_val[1]])
    // }
    //console.log(spec) //error
    //spec = np.array(spec)

//    return np.real(np.fft.fftpack.ifft(spec))
//    return np.real(spec_ifft)
    var ret = my_ifft(spec, spec.size)
    //console.log(ret)
    return my_real(ret)
}

// function _sigmoid(gain){
//     for i in xrange(len(gain)):
//         gain[i] = sigmoid(gain[1], 1, 2, _gain)
// }

// function compute(signal, noise){
// //    n_spec = np.fft.fftpack.fft(noise * _window)
//     var n_spec = my_fft(noise.multiply(_window),_winsize)
// //    n_pow = np.abs(n_spec) ** 2.0
//     var n_pow = np.abs(n_spec).multiply(np.abs(n_spec))
//     return compute_by_noise_pow(signal, n_pow)
// }

function _calc_aposteriori_snr(s_amp, n_pow){
    //return s_amp ** 2.0 / n_pow
    return s_amp.multiply(s_amp).divide(n_pow)
}

function _calc_apriori_snr(gamma){
    // return _alpha * _G ** 2.0 * _prevGamma +\
    //     (1.0 - _alpha) * np.maximum(gamma - 1.0, 0.0)  # a priori s/n ratio
    return (_prevGamma.multiply(_G.multiply(_G)).multiply(_alpha)).add(
        maximum_nparray(gamma.add(-1.0), 0.0).multiply(1.0 - _alpha))  // a priori s/n ratio
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
    //console.log("get_frame:" + String(signal.size))
    //console.log(start)
    //console.log(end)
    var ret = slice_nparray(signal,start,end)
    //console.log(ret.size)
    //console.log(ret)
    return ret
}

function add_signal(signal, frame, winsize, no){
    var shift = Math.round(winsize / 2)
    var start = Math.round(no * shift)
    var end = start + winsize
    //console.log("add_signal")
    //console.log("output size:" + String(signal.size))
    var sliced_nparr = slice_nparray(signal,start,end).add(frame)
    for(var i=start;i<end;i++){
      signal.set(i,sliced_nparr.get(i-start))
    }
}

// function write_signal(filename, params, signal){
//     wf = wave.open(filename, 'wb')
//     wf.setparams(params)
//     s = np.int16(signal * 32767.0).tostring()
//     wf.writeframes(s)
// }

// function get_window(winsize, no){
//     var shift = Math.round(winsize / 2)
//     var s = Math.round(no * shift)
//     return (s, s + winsize)
// }

// function separate_channels(signal){
//     return signal[0::2], signal[1::2]
// }

// function compute_avgamplitude(signal, winsize, window){
//     var windownum = Math.round(frame_num / (winsize / 2)) - 1
//     var avgamp = np.array(new Array(winsize))
//     for(var l=0;l<windownum;l++){
//         avgamp.add(np.abs(my_fft(get_frame(signal, winsize, l).multiply(window)),winsize))
//     }
//     return avgamp.divide(windownum * 1.0)
// }

function my_fft(ndarr,input_len){
  var fft_arr = []
  if(math.typeof(ndarr.get(0)) == "Complex"){
    for(var i=0;i<input_len;i++){
      //console.log(ndarr.get(i))
      fft_arr.push([ndarr.get(i).re,ndarr.get(i).im])
    }
  }else{
    for(var i=0;i<input_len;i++){
      //console.log(ndarr.get(i))
      fft_arr.push([ndarr.get(i),0])
    }
  }
  var tmp = np.fft(np.array(fft_arr))
  //console.log(tmp) //OK
  var ret_arr = np.array(new Array(input_len))
  for(var i=0;i<input_len;i++){
    ret_arr.set(i,math.complex(tmp.get(i,0),tmp.get(i,1)))
    // if(i==0){
    //   ret_arr.set(i,math.complex(tmp.get(i,0),tmp.get(i,1)))
    // }else{
    //   ret_arr.set(i,math.complex(tmp.get(i,1),tmp.get(i,0)))
    // }
  }
  //console.log(tmp)
  return ret_arr
}

function my_ifft(ndarr,input_len){
  var fft_arr = []
  if(math.typeof(ndarr.get(0)) == "Complex"){
    for(var i=0;i<input_len;i++){
      //console.log(ndarr.get(i))
      //fft_arr.push([ndarr.get(i).im,ndarr.get(i).re])
      fft_arr.push([ndarr.get(i).re,ndarr.get(i).im])
    }
  }else{
    for(var i=0;i<input_len;i++){
      //console.log(ndarr.get(i))
      fft_arr.push([ndarr.get(i),0])
      //fft_arr.push([0,ndarr.get(i)])
    }
  }
  var tmp = np.ifft(np.array(fft_arr))
  var ret_arr = np.array(new Array(input_len))
  for(var i=0;i<input_len;i++){
    ret_arr.set(i,math.complex(tmp.get(i,0),tmp.get(i,1)))
    // if(i==0){
    //   ret_arr.set(i,math.complex(tmp.get(i,0),tmp.get(i,1)))
    // }else{
    //   ret_arr.set(i,math.complex(tmp.get(i,1),tmp.get(i,0)))
    // }
  }
  //console.log(tmp)
  return ret_arr
}

function compute_avgpowerspectrum(signal, winsize, window){
    var windownum = Math.round(signal.size / (winsize / 2)) - 1
    var avgpow = np.array(new Array(winsize))
    for(var i=0;i<winsize;i++){
      avgpow.set(i,0.0)
    }
    for(var l=0;l<windownum;l++){
        //console.log("compute_avgpowerspectrum signal.size" + String(signal.size))
//        var tmp = np.abs(np.fft(get_frame(signal, winsize, l).multiply(window)))
        var real_arr = get_frame(signal, winsize, l).multiply(window)
        //console.log(real_arr)
        var tmp = my_abs(my_fft(real_arr,real_arr.size))
        //console.log(tmp)
        //console.log(tmp.multiply(tmp))
        avgpow = avgpow.add(tmp.multiply(tmp))
        //console.log(avgpow)
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
  var ret_arr = np.array(new Array(M))
  for(i=0;i<M;i++){
    ret_arr.set(i,0.5 - 0.5*Math.cos(2*Math.PI*i/(M-1)))
  }

  return ret_arr
}

var base_arr = new Array(frame_num)
var signal = np.array(base_arr)
// for(var i=0;i<frame_num;i++){
//   signal.set(i,0)
// }

var start_ms = new Date().getTime()

var sample_str = fs.readFileSync('./sample60.txt',"ascii")
var splited = sample_str.split(",")
for(var i=0;i<frame_num;i++){
  signal.set(i,Number(splited[i]))
  // console.log(i)
  // console.log(Number(splited[i]))
}

var elapsed_ms = new Date().getTime() - start_ms
console.log('elapsed time at read data：' + String(elapsed_ms) + "ms")

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

var result_str = ""
for(var i=0;i<output.size;i++){
  if(i==0){
    result_str = String(output.get(i)) + "," + String(output.get(i))
  }else{
    result_str = result_str + "," + String(output.get(i)) + "," + String(output.get(i))
  }
}

var wstream = fs.createWriteStream('./samples60_pyssp_mmsestsa.txt')
wstream.write(result_str, (err) => {
  if (err) throw err
    console.log('The file has been saved!')
})
