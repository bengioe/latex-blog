const katex = require('katex');
const fs = require('fs');
var data = JSON.parse(fs.readFileSync("/dev/stdin", 'utf-8'));
var results = [];
for (var i=0; i<data.length;i++){
    results.push(katex.renderToString(data[i][0], {
        throwOnError: false,
        displayMode: data[i][1]
    }));
}

console.log(JSON.stringify(results));
