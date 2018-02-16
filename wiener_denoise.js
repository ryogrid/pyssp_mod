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
var all_buffersize = bufferSize + 44
var dtype_str = "array"

var _winsize = 512
var _lambd_est = 0.001  // estimated noise lev

function op2_nparray(op1,op2,fnk){
    var arr_len = op1.size
    var is_op2_arr = false
    var ret_arr = np.array(new Array(arr_len),dtype=dtype_str)
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
    var ret_arr = np.array(new Array(arr_len),dtype=dtype_str)
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

function my_conj(op1){
    return op1_nparray(op1,math.conj)
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

function fill_nparray(arr,val){
  var arr_len = arr.size
  var ret = np.array(new Array(arr_len),dtype=dtype_str)
  for(var i=0;i<arr_len;i++){
    ret.set(i,val)
  }
  return ret
}

function my_concatenate(arr1, arr2){
  var all_len = arr1.size + arr2.size
  var ret_arr = np.array(new Array(all_len), dtype=dtype_str)
  for(var i=0;i<arr1.size;i++){
    ret_arr[i] = arr1.get(i)
  }
  for(var i=0;i<arr2.size;i++){
    ret_arr[arr1.size + i] = arr2.get(i)
  }
  return ret_arr
}

// result doesn't contain index *end*
function slice_nparray(arr,begin,end){
  var ret_len = end - begin
  var ret_arr = np.array(new Array(ret_len),dtype=dtype_str)
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

// function noise_reduction(signal,noise,params,winsize){
//     var out=np.array(new Array(frame_num),dtype=dtype_str)
//     for(var i=0;i<frame_num;i++){
//       out.set(i,0.0)
//     }
//     //console.log(n_pow) // bad values array
//     var end = Math.round(frame_num/winsize)
//     //for no in xrange(nf):
//     var shift = winsize
//     var noise = slice_nparray(noise, 0, winsize)
//     for(var no=0;no<end;no++){
//         //console.log("noise_reduction:" + String(no))
//         var slice_start = no * shift
//         var slice_end = slice_start + winsize
//         if(slice_end <= frame_num){
//           var s = get_frame(signal, winsize, no)
//           add_signal(out, decompose(s, noise), winsize, no)
//         }
//     }
//     return out
// }


function decompose(signal, noise_signal, lambd){
    // var kernel = my_concatenate(noise_signal, fill_nparray(np.array(new Array(signal.size - noise_signal.size)), 0)) // zero pad the kernel to same length
    // for(var i=0;i<kernel.size;i++){
    //   console.log(kernel.get(i))
    // }

    var H = my_fft(noise_signal, noise_signal.size)
    var a = my_multiply(my_fft(signal, signal.size), my_conj(H))
    var b = my_add(my_multiply(H, my_conj(H)),lambd*lambd)
    var deconvolved = my_ifft(my_division(a, b), signal.size)
    return my_real(deconvolved)
}

function get_frame(signal, winsize, no){
    var shift = winsize
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
    var shift = Math.round(winsize)
    var start = Math.round(no * shift)
    var end = start + winsize
    //console.log("add_signal")
    //console.log("output size:" + String(signal.size))
    for(var i=start;i<end;i++){
      signal.set(i,frame.get(i-start))
    }
}


function my_fft(ndarr,input_len){
  var fft_arr = []
  if(math.typeof(ndarr.get(0)) == "Complex"){
    for(var i=0;i<input_len;i++){
      //console.log(ndarr.get(i))
      if(i==0){
        fft_arr.push([ndarr.get(i).re,ndarr.get(i).im])
      }else{
        fft_arr.push([ndarr.get(i).im,ndarr.get(i).re])
      }
    }
  }else{
    for(var i=0;i<input_len;i++){
      //console.log(ndarr.get(i))
      //fft_arr.push([ndarr.get(i),0])

      if(i==0){
        fft_arr.push([0,ndarr.get(i)])
      }else{
        //console.log(ndarr.get(i))
        fft_arr.push([ndarr.get(i),0])
      }

    }
  }
  var tmp = np.fft(np.array(fft_arr,dtype=dtype_str))
  //console.log(tmp) //OK
  var ret_arr = np.array(new Array(input_len),dtype=dtype_str)
  for(var i=0;i<input_len;i++){
    // console.log(tmp.get(i,0))
    // console.log(tmp.get(i,1))
    //ret_arr.set(i,math.complex(tmp.get(i,0),tmp.get(i,1)))

    // if(i==0){
    //   ret_arr.set(i,math.complex(tmp.get(i,0),tmp.get(i,1)))
    // }else{
    //   ret_arr.set(i,math.complex(tmp.get(i,1),tmp.get(i,0)))
    // }
    ret_arr.set(i,math.complex(tmp.get(i,0),tmp.get(i,1)))
  }
  //console.log(tmp)
  return ret_arr
}

function my_ifft(ndarr,input_len){
  var fft_arr = []
  if(math.typeof(ndarr.get(0)) == "Complex"){
    for(var i=0;i<input_len;i++){
      //console.log(ndarr.get(i))

      // if(i==0){
      //   fft_arr.push([ndarr.get(i).im,ndarr.get(i).re])
      // }else{
      //   fft_arr.push([ndarr.get(i).re,ndarr.get(i).im])
      // }

      fft_arr.push([ndarr.get(i).re,ndarr.get(i).im])
    }
  }else{
    for(var i=0;i<input_len;i++){
      //console.log(ndarr.get(i))
      //fft_arr.push([ndarr.get(i),0])

      if(i==0){
        fft_arr.push([ndarr.get(i),0])
      }else{
        fft_arr.push([0,ndarr.get(i)])
      }

    }
  }
  var tmp = np.ifft(np.array(fft_arr,dtype=dtype_str))
  var ret_arr = np.array(new Array(input_len),dtype=dtype_str)
  for(var i=0;i<input_len;i++){
    //console.log(tmp.get(i,0))
    //console.log(tmp.get(i,1))
    //ret_arr.set(i,math.complex(tmp.get(i,0),tmp.get(i,1)))

    // if(i==0){
    //   ret_arr.set(i,math.complex(tmp.get(i,0),tmp.get(i,1)))
    // }else{
    //   ret_arr.set(i,math.complex(tmp.get(i,1),tmp.get(i,0)))
    // }
    ret_arr.set(i,math.complex(tmp.get(i,0),tmp.get(i,1)))
  }
  //console.log(tmp)
  return ret_arr
}



// for(var i=0;i<frame_num;i++){
//   signal.set(i,0)
// }

var start_ms = new Date().getTime()

var sample_str = fs.readFileSync('./sample60.txt',"ascii")
var splited = sample_str.split(",")
var signal_len = splited.length
var base_arr = new Array(splited.length)
var signal = np.array(base_arr,dtype=dtype_str)
for(var i=0;i<splited.length;i++){
  signal.set(i,Number(splited[i]))
  //console.log(signal.get(i))
  //console.log(Number(splited[i]))
}

//my_fft(signal, splited.length)
// for(var i=0;i<splited.length;i++){
//   console.log(signal.get(i))
// }


var noise_str = fs.readFileSync('./noise_sample.txt',"ascii")
splited = noise_str.split(",")
base_arr = new Array(signal_len)
var noise = np.array(base_arr,dtype=dtype_str)
for(var i=0;i<signal_len;i++){
  if(i < splited.length){
    noise.set(i,Number(splited[i]))
  }else{
    noise.set(i,0)
  }
  //console.log(noise.get(i))
  // console.log(Number(splited[i]))
}


var elapsed_ms = new Date().getTime() - start_ms
console.log('elapsed time at read data：' + String(elapsed_ms) + "ms")

var params = [2,-1,44100]

// var output = noise_reduction(signal, noise, params, _winsize)
var output = decompose(signal, noise, _lambd_est)

var result_str = ""
for(var i=0;i<output.size;i++){
  if(i==0){
    result_str = String(output.get(i)) + "," + String(output.get(i))
  }else{
    result_str = result_str + "," + String(output.get(i)) + "," + String(output.get(i))
  }
}

var wstream = fs.createWriteStream('./samples60_wiener.txt')
wstream.write(result_str, (err) => {
  if (err) throw err
    console.log('The file has been saved!')
})
