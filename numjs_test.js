var nj = require('numjs');
var a = nj.array([[2,3],[4,5],[6,7]]);
console.log(a.slice(null,[0,1]))
console.log(a)

var b = nj.array([1,2,3,4,5,6])
b.add(nj.array([1,1,1,1,1,1]))
console.log(b)
