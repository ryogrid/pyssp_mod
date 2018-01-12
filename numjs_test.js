var nj = require('numjs');
var a = nj.array([[2,3],[4,5],[6,7]]);
//console.log(a.slice(null,[0,1]))
//console.log(a)
console.log(a.get(0,0))

var b = nj.array([1,2,3,4,5,6])
b.add(nj.array([1,1,1,1,1,1]))
console.log(b)
console.log(b.constructor)

console.log(Math.atan(1.5))
console.log(true + false)
