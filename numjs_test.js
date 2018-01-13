var nj = require('numjs');
var math = require("mathjs")

var a = nj.array([[2,3],[4,5],[6,7]]);
//var a_ = nj.array([[2,0],[4,0],[6,0]]);
//console.log(a.slice(null,[0,1]))
//console.log(a)
console.log(a.get(0,0))

var b = nj.array([1,2,3,4,5,6])
b.add(nj.array([1,1,1,1,1,1]))
console.log(b)
console.log(b.constructor)

console.log(Math.atan(1.5))
console.log(true + false)

console.log(nj.fft(a))
console.log(nj.ifft(a))
//console.log(nj.fft(a_))
console.log(Math.atan(1/1))
