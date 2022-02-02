
let polygon = PolygonTools.polygon;
let unit_square = [[0,0],[0,1],[1,1],[1,0]];


function line2polys(a, b, c){
    // ax + by + c = 0
    // everything in [0,1]^2
    if (b != 0){
        // y = -(ax + c)/b = ux+v
        let u = -a/b;
        let v = -c/b;
        let polydown = [[0, v], [1, u+v],
                        [1, Math.min(u+v-1, 0)],
                        [0, Math.min(v-1, 0)]];
        let polyup = [[0, v], [1, u+v],
                      [1, Math.max(u+v+1, 1)],
                      [0, Math.max(v+1, 1)]];
        return [polygon.intersection(unit_square, polydown)[0],
                polygon.intersection(unit_square, polyup)[0]];
    }
}

function drawPoly(ctx, p, color, scale, offset){
    ctx.beginPath()
    ctx.moveTo(0.5 + p[0][0]*scale + offset[0], 0.5 + -p[0][1]*scale + offset[1]);
    for (let i=0; i<p.length; i++){
        ctx.lineTo(0.5 + p[i][0]*scale + offset[0], 0.5 + -p[i][1]*scale + offset[1]);
    }
    ctx.lineTo(0.5 + p[0][0]*scale + offset[0], 0.5 + -p[0][1]*scale + offset[1]);
    ctx.strokeStyle = "#000";
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.fill();
}

function randomColor(){
    return '#'+Math.floor((0.1+Math.random()*0.9)*16777215).toString(16);
}

function cload(elem){
    let cns = document.getElementById(elem);
    let ctx = cns.getContext("2d");

    let w0 = [[0.5, -0.25]];//, [0.7, 0.5]];
    let b0 = [-0.1, -0.5];

    let polys = [unit_square];

    for (let i=0; i<w0.length; i++){
        let ps = line2polys(w0[i][0], w0[i][1], b0[i]);
        var new_polys = [];
        for (let j=0; j<polys.length; j++){
            if (ps[0] !== undefined)
                new_polys = new_polys.concat(polygon.intersection(polys[j], ps[0]));
            if (ps[1] !== undefined)
                new_polys = new_polys.concat(polygon.intersection(polys[j], ps[1]));
        }
        polys = new_polys;
    }
    for (let i=0;i<polys.length;i++){
        drawPoly(ctx, polys[i], randomColor(), 50, [20, 70]);
    }

    /*
    z = 0.5x -0.25y + 0.1
    // then for z=0
    y = (0.5x + 0.1)/(-0.25)
    // this creates two linear regions in [0,1]^2
    p0 = [[0, 0], [1.5, 0], []];

    0 = ax + by + c
    0 = dx + ey + f
    */
        
    

    
}

// some random numbers
let rn = [0.50775732, 0.52683232, 0.8249355 , 0.00601473, 0.77073607,
          0.27948781, 0.10785489, 0.41280629, 0.97314865, 0.48842995,
          0.88001758, 0.87143943, 0.1401847 , 0.93957719, 0.17065431,
          0.0208293 , 0.57063754, 0.67174254, 0.7453332 , 0.29671285]
let rn2 = [0.57773516, 0.80025286, 0.68611556, 0.85399476, 0.58163026,
           0.81799314, 0.34680883, 0.24925959, 0.30118507, 0.40371694,
           0.50809106, 0.83688797, 0.92814497, 0.85441302, 0.93782646,
           0.03707604, 0.39134895, 0.67581206, 0.71210626, 0.36955049]
let rn3 = [8.03361871e-01, 8.31183462e-01, 4.32146219e-01, 2.81248255e-01,
       6.28290561e-01, 7.41027422e-01, 8.92876242e-01, 3.84329345e-01,
       4.14151852e-01, 3.68291512e-01, 7.10803627e-01, 8.00605054e-02,
       5.62639159e-01, 7.19591909e-01, 4.64529431e-01, 9.34001663e-01,
       3.80177459e-01, 7.68667533e-01, 4.58209714e-01, 5.51768673e-01,
       5.74860074e-01, 7.82275870e-01, 4.05316127e-01, 2.99773606e-01,
       8.45556286e-01, 1.89892352e-01, 5.50118158e-02, 1.11637916e-01,
       8.91440583e-01, 5.38785355e-01, 3.27113962e-01, 3.62416895e-01,
       8.34166524e-01, 2.11314658e-01, 1.05517220e-01, 1.66038913e-01,
       3.62874329e-04, 3.24462244e-01, 4.83909818e-01, 8.85893347e-01,
       1.78887043e-01, 1.30380586e-01, 5.41352736e-01, 4.05961192e-01,
       6.50627773e-01, 1.57388954e-01, 1.57657194e-01, 3.35750619e-01,
       4.93335635e-01, 4.29100880e-01, 8.23864864e-01, 7.05228418e-01,
       7.80640599e-01, 1.15568916e-01, 7.24925745e-01, 6.19946951e-01,
       5.52864543e-01, 4.27354702e-01, 8.66925136e-01, 1.44742908e-01]
var _ctx;
function drawArrow(ox, oy, ex, ey, hl, color){
    _ctx.beginPath()
    var hl = hl === undefined? 8: hl; // length of head in pixels
    var dx = ex - ox;
    var dy = ey - oy;
    var angle = Math.atan2(dy, dx);

    _ctx.moveTo(ox, oy)
    _ctx.lineTo(ex, ey)
    _ctx.lineTo(ex - hl * Math.cos(angle - Math.PI / 8), ey - hl * Math.sin(angle - Math.PI / 8));
    _ctx.moveTo(ex, ey);
    _ctx.lineTo(ex - hl * Math.cos(angle + Math.PI / 8), ey - hl * Math.sin(angle + Math.PI / 8));
    _ctx.strokeStyle = color === undefined? "#000": color;
    _ctx.stroke();
}

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


function loadArrowsAnim(elem){
    let cns = document.getElementById(elem);
    let ctx = cns.getContext("2d");
    let span = document.getElementById(elem+"_div");
    span.style.position = 'relative';
    let tex = texBox(span);
    var t = 0;
    let w = cns.width, h = cns.height;

    let N = 60 * 4;
    function frame(){
        ctx.clearRect(0, 0, w, h); 
        t = (t + 1) % N;
        _ctx = ctx;
        drawArrow(25.5, 120, 25.5, 20, undefined, '#444');
        drawArrow(25, 120.5, 200, 120.5, undefined, '#444');
        tex('xaxislabel', '\\theta').moveTo(5, 20);
        tex('yaxislabel', '\\theta').moveTo(200-10, 120);
        tex('SGD label', 'SGD', true).moveTo(200-10, 40);
        var base = [30, 100];
        let end = Math.floor(t / (N / (20+5))) + 1;
        for (i in rn){
            if (i == end){
                break;
            }
            let to = [base[0] + (rn[i] + 0.1) * 30, base[1] + (0.55 - rn2[i]) * 40];
            drawArrow(base[0], base[1], to[0], to[1]);
            base = to;
        }
        tex('sgd g2', 'g_i').moveTo(base[0]+2, base[1]-10);

        tex('mom label', 'Momentum', true).moveTo(200-20, 160);
        drawArrow(25.5, 120+120, 25.5, 20+120, undefined, '#444');
        drawArrow(25, 120.5+120, 200, 120.5+120, undefined, '#444');
        tex('xaxislabel_m', '\\theta').moveTo(5, 20+120);
        tex('yaxislabel_m', '\\theta').moveTo(200-10, 120+120);
        tex('SGD label', 'SGD', true).moveTo(200-10, 40);
        var base = [30, 220];
        var mu = [0, 0];
        var beta = 0.6;
        for (i in rn){
            if (i == end) break
            let g = [(rn[i] + 0.1) * 30, (0.55 - rn2[i]) * 50]
            let to = [base[0] + g[0], base[1] + g[1]];
            drawArrow(base[0], base[1], to[0], to[1]);
            mu = [mu[0] * beta + g[0] * (1-beta), mu[1] * beta + g[1] * (1-beta)];
            drawArrow(base[0], base[1], base[0] + mu[0], base[1] + mu[1], undefined, '#060');
            base = [base[0] + mu[0], base[1] + mu[1]];
        }
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);

}

function momentumSum(elem){
    let cns = document.getElementById(elem);
    let ctx = cns.getContext("2d");
    let span = document.getElementById(elem+"_div");
    span.style.position = 'relative';

    let tex = texBox(span);
    var t = 0;
    let w = cns.width, h = cns.height;

    let N = 60 * 4;
    function frame(){
        ctx.clearRect(0, 0, w, h); 
        t = (t + 1) % N;
        _ctx = ctx;
        var base = [30, 20];
        tex('sum eq', '\\mu_t = \\sum \\beta^{t-i}g_i').moveTo(base[0], base[1])
        base[1] += 50;
        drawArrow(base[0], base[1], base[0] + 20, base[1] - 20, undefined, '#060');
        tex('sub5', '{}_5').moveTo(base[0] + 10, base[1] - 17)
        tex('eq sign', '=').moveTo(base[0] + 30, base[1] - 20)
        base[0] += 70;
        for (i in rn){
            if (i == 5){
                break;
            }
            let g = [(rn[i] + 0.1) * 30, (0.55 - rn2[i]) * 50]
            let to = [base[0] + g[0], base[1] + g[1]];
            drawArrow(base[0], base[1], to[0], to[1]);
            tex('beta'+i, '\\beta^{'+(5-i)+'}').moveTo(base[0] - 20, base[1] - 20)
            if (i < 4)
                tex('plus sign'+i, i == 4? '+...':'+').moveTo(base[0] + g[0] + 5, base[1] - 20)
            base = [base[0] + g[0] + 40, base[1]];
        }
        //requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}


function momentumStarSum(elem){
    let cns = document.getElementById(elem);
    let ctx = cns.getContext("2d");
    let span = document.getElementById(elem+"_div");
    span.style.position = 'relative';

    let tex = texBox(span);
    var t = 0;
    let w = cns.width, h = cns.height;

    let N = 60 * 4;
    function frame(){
        ctx.clearStyle = '#222';
        ctx.clearRect(0, 0, w, h); 
        t = (t + 1) % N;
        _ctx = ctx;
        var base = [30, 20];
        tex('sum eq', '\\mu^*_t = \\sum \\beta^{t-i}g^t_i').moveTo(base[0], base[1])
        base[1] += 50;
        drawArrow(base[0], base[1], base[0] + 20, base[1] - 15, undefined, '#900');
        tex('sub5', '{}_5').moveTo(base[0] + 10, base[1] - 17)
        tex('eq sign', '=').moveTo(base[0] + 30, base[1] - 20)
        base[0] += 70;
        for (i in rn){
            if (i == 5){
                break;
            }
            i = i*1
            let g = [(rn[i] + 0.1 + 0.3*rn[i+1]) * 30, (0.55 - rn2[i]) * 50]
            let to = [base[0] + g[0], base[1] + g[1]];
            drawArrow(base[0], base[1], to[0], to[1],
                      undefined, '#a0a');
            tex('beta'+i, '\\beta^{'+(5-i)+'}').moveTo(base[0] - 20, base[1] - 20)
            if (i < 4)
                tex('plus sign'+i, i == 4? '+...':'+').moveTo(base[0] + g[0] + 5, base[1] - 20)
            base = [base[0] + g[0] + 40, base[1]];
        }
        //requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

function momentumStarSumAnim(elem){
    let cns = document.getElementById(elem);
    let ctx = cns.getContext("2d");
    let span = document.getElementById(elem+"_div");
    span.style.position = 'relative';

    let tex = texBox(span);
    var t = 0;
    let w = cns.width, h = cns.height;

    let N = 60 * 4;
    function frame(){
        ctx.clearRect(0, 0, w, h); 
        t = (t + 1) % N;
        _ctx = ctx;
        drawArrow(25.5, 120, 25.5, 20, undefined, '#444');
        drawArrow(25, 120.5, 200, 120.5, undefined, '#444');
        tex('xaxislabel', '\\theta').moveTo(5, 20);
        tex('yaxislabel', '\\theta').moveTo(200-10, 120);
        var base = [30, 100];
        let end = Math.min(Math.floor(t / (N / (20+5))) + 1, 20);
        var mu = [0, 0];
        var beta = 0.7;
        for (i in rn){
            if (i == end) break
            i = i * 1;
            let g = [(rn[i] + 0.15) * 40, (0.55 - rn2[i]) * 50]
            let to = [base[0] + g[0] + 10 * (rn3[i+end]), base[1] + g[1]];
            drawArrow(base[0], base[1], to[0], to[1], undefined, '#a0a');
            mu = [mu[0] * beta + g[0] * (1-beta), mu[1] * beta + g[1] * (1-beta)];
            drawArrow(base[0], base[1], base[0] + mu[0], base[1] + mu[1], undefined, '#900');
            base = [base[0] + mu[0], base[1] + mu[1]];
        }
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

function lerp(a, b, t){
    return [a[0] * t + b[0] * (1-t), a[1]*t + b[1]*(1-t)]
}

function momentumStarSumAnimFollow(elem){
    let cns = document.getElementById(elem);
    let ctx = cns.getContext("2d");
    let span = document.getElementById(elem+"_div");
    span.style.position = 'relative';

    let tex = texBox(span);
    var t = 0;
    let w = cns.width, h = cns.height;

    let N = 60 * 8;
    function frame(){
        ctx.clearRect(0, 0, w, h); 
        t = (t + 1) % N;
        _ctx = ctx;
        drawArrow(25.5, 120, 25.5, 20, undefined, '#444');
        drawArrow(25, 120.5, 200, 120.5, undefined, '#444');
        tex('xaxislabel', '\\theta').moveTo(5, 20);
        tex('yaxislabel', '\\theta').moveTo(200-10, 120);
        var base = [30, 100];
        let ut = t / (N / (20+5)) + 1;
        let end = Math.min(Math.floor(ut), 20);
        let s = Math.min(ut - end, 1);
        var mu = [0, 0];
        var beta = 0.7;
        var last_base = base;
        for (i in rn){
            if (i == end) break
            i = i * 1;
            let g = [(rn[i] + 0.15) * 40, (0.55 - rn2[i]) * 50]
            let to = [base[0] + g[0] + 10 * (rn3[i+end]), base[1] + g[1]];
            mu = [mu[0] * beta + g[0] * (1-beta), mu[1] * beta + g[1] * (1-beta)];
            drawArrow(base[0], base[1], base[0] + mu[0], base[1] + mu[1], undefined, '#900');
            last_base = base;
            base = [base[0] + mu[0], base[1] + mu[1]];
            
        }
        for (i in rn){
            if (i == end) break
            i = i * 1;
            //tex('test', '\\theta').moveTo(30, 30).setText(s.toFixed(3));
            let g = [(rn[i] + 0.15) * 40 * 2, (0.55 - rn2[i]) * 50 * 3];
            let beta_scale = Math.pow(0.9, end-i+s)
            g = [g[0] * beta_scale, g[1] * beta_scale]
            let lerp_base = lerp(base, last_base, s);
            let to = [lerp_base[0] + g[0] + 10 * (rn3[i+end+1]), lerp_base[1] + g[1]];
            let to2 = [lerp_base[0] + g[0] + 10 * (rn3[i+end]), lerp_base[1] + g[1]];
            let lerp_to = lerp(to, to2, s);
            drawArrow(lerp_base[0], lerp_base[1], lerp_to[0], lerp_to[1], 4, '#a0a');
        }
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}
