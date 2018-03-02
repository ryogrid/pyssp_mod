// -*- coding: utf-8 -*-
var np = require("numjs")
var fs = require("fs")
var math = require("mathjs")
var jsonfile = require('jsonfile')

var _BIGNUM = 3000000.0
var frame_num = 2646000;
var bufferSize = frame_num * 4;
var all_buffersize = bufferSize + 44
var dtype_str = "array"

var _winsize = 512
var _noise_spectol = null
var _window = null

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

// result doesn't contain index *end*
function slice_nparray(arr,begin,end){
  var ret_len = end - begin
  var ret_arr = np.array(new Array(ret_len),dtype=dtype_str)
  for(var i=begin;i<end;i++){
    //console.log(arr.get(i))
    ret_arr.set(i-begin,arr.get(i))
  }
  return ret_arr
}

function noise_reduction(signal,noise_spectol,winsize){
    var out=np.array(new Array(signal.size),dtype=dtype_str)
    for(var i=0;i<signal.size;i++){
      out.set(i,0.0)
    }
    //console.log(n_pow) // bad values array
    var end = Math.round(signal.size/winsize)
    //for no in xrange(nf):
    var shift = winsize
    for(var no=0;no<end;no++){
        //console.log("noise_reduction:" + String(no))
        var slice_start = no * shift
        var slice_end = slice_start + winsize
        if(slice_end <= frame_num){
          var s = get_frame(signal, winsize, no)
          add_signal(out, denoise(s, noise_spectol), winsize, no)
        }
    }
    return out
}

function mul_exp_nparray(arr,real,imaginary){
  var arr_len = arr.size
  var ret = np.array(new Array(arr_len),dtype=dtype_str)
  var x = math.complex(real, imaginary)
  for(var i=0;i<arr_len;i++){
    var tmp = math.exp(math.multiply(arr.get(i),x))
    ret.set(i,tmp)
  }
  return ret
}

function denoise(signal, noise_spectol){
    var signal_spectol = my_fft(signal, signal.size)
    var s_amp = my_abs(signal_spectol)
    var s_phase = my_angle(signal_spectol)

    //var denoised_amp = s_amp.multiply(0.001).add(n_amp.multiply(1.0 - 0.001))
    var denoised_amp = s_amp.add(noise_spectol.multiply(-1))
    var mul_exp = mul_exp_nparray(s_phase,0,1)
    var tmp = my_multiply(denoised_amp,mul_exp)
    return my_real(my_ifft(tmp,signal.size))
}

function get_frame_half_shift(signal, winsize, no){
    var shift = Math.round(winsize/2)
    var start = Math.round(no * shift)
    var end = start + winsize
    var ret = slice_nparray(signal,start,end)
    return ret
}

function get_frame(signal, winsize, no){
    var shift = winsize
    var start = Math.round(no * shift)
    var end = start + winsize
    var ret = slice_nparray(signal,start,end)
    return ret
}

function add_signal(signal, frame, winsize, no){
    var shift = Math.round(winsize)
    var start = Math.round(no * shift)
    var end = start + winsize
    for(var i=start;i<end;i++){
      signal.set(i,frame.get(i-start))
    }
}

function my_angle(ndarr){
  var arr_len = ndarr.size
  var ret_arr = np.array(new Array(arr_len),dtype=dtype_str)
  for(var i=0;i<arr_len;i++){
    //ret_arr.set(i, math.atan(ndarr.get(i))*2)
    var val = ndarr.get(i)
    ret_arr.set(i,math.atan2(val.im, val.re))
    //console.log(ret_arr.get(i))
  }
  return ret_arr
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

      // if(i==0){
      //   fft_arr.push([0,ndarr.get(i)])
      // }else{
      //   //console.log(ndarr.get(i))
      //   fft_arr.push([ndarr.get(i),0])
      // }
      fft_arr.push([ndarr.get(i),0])
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

function gen_noise_spectol(signal, winsize){
    var windownum = Math.round(signal.size / (winsize / 2)) - 1 -1
    var avgpow = np.array(new Array(winsize),dtype=dtype_str)
    for(var i=0;i<winsize;i++){
      avgpow.set(i, 0.0)
    }
    for(var l=0;l<windownum;l++){
        //console.log("compute_avgpowerspectrum signal.size" + String(signal.size))
//        var tmp = np.abs(np.fft(get_frame(signal, winsize, l).multiply(window)))
        var real_arr = get_frame_half_shift(signal, winsize, l).multiply(_window)
        //console.log(real_arr)
        var tmp = my_abs(my_fft(real_arr,real_arr.size))
        //console.log(tmp.multiply(tmp))
        // for(var i=0;i<winsize;i++){
        //   console.log(avgpow.get(i))
        // }
        avgpow = avgpow.add(tmp)
        // for(var i=0;i<winsize;i++){
        //   console.log(avgpow.get(i))
        // }
    }
    return avgpow.divide(windownum * 1.0)
}

function generate_hann_arr(M){
  var ret_arr = np.array(new Array(M),dtype=dtype_str)
  for(i=0;i<M;i++){
    ret_arr.set(i,0.5 - 0.5*Math.cos(2*Math.PI*i/(M-1)))
  }

  return ret_arr
}

// function gen_noise_spectol(noise_signal, winsize){
//   var all_spectrum = my_abs(my_fft(noise_signal, noise_signal.size))
//   var out_spectrum = np.array(new Array(winsize), dtype=dtype_str)
//
//   out_spectrum.set(0, all_spectrum.get(0))
//   var mo = (all_spectrum.size - 1) % (winsize - 1)
//   var slide = Math.round((all_spectrum.size - 1 - mo) / (winsize - 1))
//   for(var i=0;i<(winsize-1);i++){
//     var sumval = 0
//     var avgval = 0
//     for(var j=1+slide*i;j<1+slide*(i+1);j++){
//       sumval += all_spectrum.get(j)
//     }
//     if(i==(winsize-2)){
//       for(k=1+slide*(i+1);k<all_spectrum.size;k++){
//         sumval += all_spectrum.get(k)
//       }
//       avgval = sumval / (slide + mo)
//     }else{
//       avgval = sumval / slide
//     }
//     out_spectrum.set(i+1, avgval)
//   }
//
//   return out_spectrum
// }

var start_ms = new Date().getTime()

var sample_str = fs.readFileSync('./sample60.txt',"ascii")
var splited = sample_str.split(",")
var signal_len = splited.length
var base_arr = new Array(splited.length)
var signal = np.array(base_arr,dtype=dtype_str)
for(var i=0;i<splited.length;i++){
  signal.set(i,Number(splited[i]))
}

var noise_str = fs.readFileSync('./noise_sample.txt',"ascii")
splited = noise_str.split(",")
base_arr = new Array(splited.length)
var noise = np.array(base_arr,dtype=dtype_str)
for(var i=0;i<splited.length;i++){
  noise.set(i,Number(splited[i]))
  //console.log(noise.get(i))
  // console.log(Number(splited[i]))
}

_window = generate_hann_arr(_winsize)
_noise_spectol = gen_noise_spectol(noise, _winsize)
jsonfile.writeFile('./noise_spectol.json', _noise_spectol, {
    encoding: 'utf-8',
    replacer: null,
    spaces: null
}, function (err) {
});

var elapsed_ms = new Date().getTime() - start_ms
console.log('elapsed time at read data：' + String(elapsed_ms) + "ms")

var output = noise_reduction(signal, _noise_spectol, _winsize)
//var output = denoise(signal, noise)
//output = signal.add(output)

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
