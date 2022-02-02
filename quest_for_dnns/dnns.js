
/* -~-~-~-~-~-====-~-~-~-~-~-
 Drawing utilities
*/

var _globalAnimOverride = false;
var _ctx;

var draw = {
    poly: function(p, color, scale, offset){
        offset[0] += this._offset[0];
        offset[1] += this._offset[1];
        _ctx.beginPath()
        _ctx.moveTo(0.5 + p[0][0]*scale + offset[0], 0.5 + -p[0][1]*scale + offset[1]);
        for (let i=0; i<p.length; i++){
            _ctx.lineTo(0.5 + p[i][0]*scale + offset[0], 0.5 + -p[i][1]*scale + offset[1]);
        }
        _ctx.lineTo(0.5 + p[0][0]*scale + offset[0], 0.5 + -p[0][1]*scale + offset[1]);
        _ctx.strokeStyle = "#000";
        _ctx.stroke();
        _ctx.fillStyle = color;
        _ctx.fill();
    },
    line: function(ox, oy, ex, ey, hl=8, color="#000"){
        _ctx.beginPath()
        _ctx.moveTo(ox + this._offset[0], oy + this._offset[1])
        _ctx.lineTo(ex + this._offset[0], ey + this._offset[1])
        _ctx.strokeStyle = color;
        _ctx.stroke();
    },
    arrow: function(ox, oy, ex, ey, hl=8, color="#000"){
        _ctx.beginPath()
        ox += this._offset[0];
        oy += this._offset[1];
        ex += this._offset[0];
        ey += this._offset[1];
        var dx = ex - ox;
        var dy = ey - oy;
        var angle = Math.atan2(dy, dx);
        
        _ctx.moveTo(ox, oy)
        _ctx.lineTo(ex, ey)
        _ctx.lineTo(ex - hl * Math.cos(angle - Math.PI / 8), ey - hl * Math.sin(angle - Math.PI / 8));
        _ctx.moveTo(ex, ey);
        _ctx.lineTo(ex - hl * Math.cos(angle + Math.PI / 8), ey - hl * Math.sin(angle + Math.PI / 8));
        _ctx.strokeStyle = color;
        _ctx.stroke();
    },
    _offsets: [],
    _offset: [0,0],
    pushOffset: function(offset){
        this._offsets.push(offset);
        this._offset[0] += offset[0];
        this._offset[1] += offset[1];
    },
    popOffset: function(){
        let offset = this._offsets.pop();
        this._offset[0] -= offset[0];
        this._offset[1] -= offset[1];
        return offset;
    },
};

function texBox(span){
    var texElems = {};
    function tex(name, string, notex){
        if (texElems.hasOwnProperty(name))
            return texElems[name]
        let elem = document.createElement("span");
        elem.innerHTML = notex === undefined? katex.renderToString(string): string;
        elem.style.position = 'absolute';
        elem.style.transform = 'scale(0.9, 0.9)';
        span.appendChild(elem);
        texElems[name] = {
            'moveTo': function(x,y){elem.style.top = y+'px';elem.style.left = x+'px'; return this;},
            'setText': function(s){elem.innerHTML = s; return this;},
            'hide': function(){elem.style.visibility = 'hidden';},
            'show': function(){elem.style.visibility = 'visible';},
            'elem': elem,
        }
        return texElems[name]
    }
    return tex;
}

function randomColor(){
    return '#'+Math.floor((0.1+Math.random()*0.9)*16777215).toString(16);
}

function drawMLP(offset, layers, color="#000",
                 radius=6, neuronsep=2, layersep=40){
    let totsep = radius*2 + neuronsep;
    _ctx.strokeStyle = color;
    let maxL = Math.max.apply(undefined, layers);
    for (var L=0;L<layers.length;L++){
        let layerOffset = (maxL - layers[L]) * totsep / 2;
        for (var i=0;i<layers[L];i++){
            _ctx.beginPath();
            let npos = [offset[0] + layersep * L, layerOffset + offset[1] + totsep * i];
            _ctx.arc(npos[0], npos[1], radius, 0, 6.29);
            _ctx.stroke();
            if (L<layers.length-1){
                _ctx.beginPath();
                for (var j=0; j<layers[L+1]; j++){
                    let nextLayerOffset = (maxL - layers[L+1]) * totsep / 2;
                    _ctx.moveTo(npos[0] + radius, npos[1])
                    _ctx.lineTo(offset[0] + layersep * (L+1) - radius,
                                nextLayerOffset + offset[1] + totsep * j)
                }
                _ctx.stroke();
            }
        }
    }
}

function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

function unpackCanvas(elem){
    let cns = document.getElementById(elem);
    let ctx = cns.getContext("2d");
    let div = document.getElementById(elem+"_div");
    div.style.position = 'relative';
    div.style.display = 'inline-block';
    let tex = texBox(div);
    return {cns, ctx, div, tex};
}




/* -~-~-~-~-~-====-~-~-~-~-~-
 Math utilities
*/

function PseudoRandom(seed) {
  this._seed = seed % 2147483647;
  if (this._seed <= 0) this._seed += 2147483646;
}

PseudoRandom.prototype.next = function () {
  return this._seed = this._seed * 16807 % 2147483647;
};

PseudoRandom.prototype.nextFloat = function (opt_minOrMax, opt_max) {
  // We know that result of next() will be 1 to 2147483646 (inclusive).
  return (this.next() - 1) / 2147483646;
};



/* -~-~-~-~-~-====-~-~-~-~-~-
 Plots
 Each canvas corresponds to one function below
*/


function connectionistDNN(elem){
    let {cns, ctx, div, tex} = unpackCanvas(elem);
    
    _ctx = ctx; // make this context current
    drawMLP([50, 50], [5,6,8,4,4,2]);
    draw.arrow(50, 35, 240, 35);
    tex('xlab', 'x').moveTo(25, 85);
    tex('fxlab', 'f(x)').moveTo(260, 85);
}


function plane_1neuron(elem){
    let {cns, ctx, div, tex} = unpackCanvas(elem);
    _ctx = ctx;
    draw.poly([[0.2, 0], [0.5, 1], [1, 1], [1, 0]],
             "#fcc", 70, [20, 90]);
    draw.poly([[0.2, 0], [0.5, 1], [0, 1], [0, 0]],
             "#acf", 70, [20, 90]);
    draw.arrow(20, 90, 125, 90);
    draw.arrow(20, 90, 20, 10);
    
    draw.poly([[0.2, 0], [0.3, 0.5], [0.8, 0]],
             "#fcc", 70, [160, 90]);
    draw.poly([[0.2, 0], [0.3, 0.5], [0, 0.8], [0, 0]],
             "#acf", 70, [160, 90]);
    draw.poly([[0.3, 0.5], [0, 0.8], [0, 1], [0.4, 1]],
             "#afa", 70, [160, 90]);
    draw.poly([[0.9, 0], [1, 0], [1, 1], [0.8, 1]],
             "#ffa", 70, [160, 90]);
    draw.poly([[0.9, 0], [0.8, 1], [0.4, 1], [0.3, 0.5], [0.8, 0]],
             "#8ce", 70, [160, 90]);
    draw.arrow(160, 90, 275, 90);
    draw.arrow(160, 90, 160, 10);
    tex('xlab', 'x_1').moveTo(120, 85);
    tex('ylab', 'x_2').moveTo(0, 5);
}

function plane_distributed_repr(elem){
    let {cns, ctx, div, tex} = unpackCanvas(elem);
    _ctx = ctx;
    draw.arrow(20, 190, 20 + 180, 190);
    draw.arrow(20, 190, 20, 10);
    draw.arrow(20, 190, 90, 130);
    tex('xlab', 'h_1').moveTo(200, 185);
    tex('ylab', 'h_2').moveTo(0, 5);
    tex('zlab', 'h_3').moveTo(60, 120);
    
    draw.arrow(100, 90, 180, 100);
    draw.arrow(100, 90, 140, 30);
    draw.arrow(100, 90, 70, 50);
    tex('a', 'fur fluffiness', true).moveTo(180, 100);
    tex('b', 'length', true).moveTo(140, 30);
    tex('c', 'eye color', true).moveTo(40, 30);
}


function plane_move_top_anim(elem){
    let {cns, ctx, div, tex} = unpackCanvas(elem);
    let _data = [];
    let _data2 = [];
    
    function recomputeColors(dat){
        let c2h = (c => {var hex = Math.floor(c*255).toString(16);
                         return hex.length == 1 ? "0" + hex : hex});
        let a2h = (a => "#" + c2h(a[0]) + c2h(a[1]) + c2h(a[2]));
        for (var i=0; i<dat.length; i++){
            for (var j=0; j<dat[i][0].length; j++){
                dat[i][0][j] = a2h(dat[i][0][j]);
            }
        }
    }

    function f(){
        if (_globalAnimOverride) return;
        if (!isInViewport(cns)){
            requestAnimationFrame(f);
            return;
        }
        requestAnimationFrame(f);
        _ctx = ctx;
        ctx.clearRect(0, 0, cns.width, cns.height);
        let x = Math.floor((0.5 + Math.sin(new Date().getTime() / 1000) / 2) * _data.length);
        frame = _data[x];
        if (frame === undefined)
            console.log(x)
        for (var i=0; i<frame[0].length; i++){
            draw.poly(frame[1][i], frame[0][i],
                     35, [20+35, 90-35]);
        }
        draw.arrow(20, 90, 125, 90);
        draw.arrow(20, 90, 20, 10);
        if (_data2.length == 0)
            return;

        draw.pushOffset([150, 0]);
        let x2 = Math.floor((0.5 + Math.sin(new Date().getTime() / 1000) / 2) * _data2.length);
        frame = _data2[x2];
        for (var i=0; i<frame[0].length; i++){
            draw.poly(frame[1][i], frame[0][i],
                     35, [20+35, 90-35]);
        }
        draw.arrow(20, 90, 125, 90);
        draw.arrow(20, 90, 20, 10);
        draw.popOffset();
    }
    
    fetch('anim_last_layer_bias.json')
        .then(response => response.json())
        .then(data => {_data=data; recomputeColors(_data); requestAnimationFrame(f);});
    fetch('anim_last_layer_weight.json')
        .then(response => response.json())
        .then(data => {_data2=data; recomputeColors(_data2);});
}

function overparam_smoothing(elem){
    let {cns, ctx, div, tex} = unpackCanvas(elem);
    let rng = new PseudoRandom(42);
    let nhids = [10, 100, 200];
    let scales = [1, 0.6, 0.3];
    let params = [];
    tex('x', 'x').moveTo(110, 55);
    tex('fx', 'f(x)').moveTo(-20, 25);
    for (var i=0;i<nhids.length;i++){
        tex('lab'+i, 'n_h='+nhids[i]).moveTo(40 + 120 * i, 60);
        let w1 = new Array(nhids[i]).fill().map((x)=>-(rng.nextFloat()*2 - 1));// /Math.sqrt(nhids[i])*Math.sqrt(20));
        let w2 = new Array(nhids[i]).fill().map((x)=>-(rng.nextFloat()*2 - 1)/2);
        let b = new Array(nhids[i]).fill().map((x)=>(rng.nextFloat() * 2 - 1) * 0.25);
        params.push([w1,w2,b])
    }
    
    function f(){
        if (_globalAnimOverride) return;
        if (!isInViewport(cns)){
            requestAnimationFrame(f);
            return;
        }
        _ctx = ctx;
        ctx.clearRect(0, 0, cns.width, cns.height);
        let uv = Math.sin(new Date().getTime() / 1000);
        let uv2 = Math.sin(new Date().getTime() / 1200);
        let xs = new Array(100).fill().map((_, i)=> i/100);
        for (var ip=0;ip<nhids.length;ip++){
            let w1 = params[ip][0]; w2 = params[ip][1]; b = params[ip][2];
            draw.pushOffset([20 + 120 * ip, 0]);
            draw.arrow(0, 60, 100, 60);
            draw.arrow(0, 60, 0, 10);
            draw.pushOffset([0, 60]);
            let s = 100; // scale
            let ys = new Array(100).fill();
            for (var i=0; i<100; i++){
                let h = new Array(nhids[ip]).fill().map(
                    (_, j) => Math.max(0, (xs[i] - 0.5) * w1[j] + b[j]*uv));
                ys[i] = h.reduce((a,hi,j) => a + w2[j]*hi*uv2) * scales[ip]+0.25;
            }
            for (var i=0; i<99; i++){
                draw.line(xs[i]*s, -ys[i]*s, xs[i+1]*s, -ys[i+1]*s, 8, "#1F77B4");
            }
            draw.popOffset();
            draw.popOffset();
        }
        requestAnimationFrame(f);
    }
    requestAnimationFrame(f);
}
