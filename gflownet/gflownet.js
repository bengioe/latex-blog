
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
    ctx.moveTo(0.5 + p[0][0]*scale + offset[0], 0.5 + p[0][1]*scale + offset[1]);
    for (let i=0; i<p.length; i++){
        ctx.lineTo(0.5 + p[i][0]*scale + offset[0], 0.5 + p[i][1]*scale + offset[1]);
    }
    ctx.lineTo(0.5 + p[0][0]*scale + offset[0], 0.5 + p[0][1]*scale + offset[1]);
    ctx.strokeStyle = "#000";
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.fill();
}

function randomColor(){
    return '#'+Math.floor((0.1+Math.random()*0.9)*16777215).toString(16);
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

function drawCircle(ox, oy, r, color, fill){
    _ctx.beginPath()
    _ctx.arc(ox, oy, r, 0, Math.PI * 2);
    _ctx.strokeStyle = color === undefined? "#000": color;
    if (fill !== undefined){
        _ctx.fillStyle = fill;
        _ctx.fill();
    }
    _ctx.stroke();
}

function texBox(span){
    var texElems = {};
    function tex(name, string, notex, color){
        if (texElems.hasOwnProperty(name))
            return texElems[name]
        let elem = document.createElement("span");
        elem.innerHTML = notex === undefined? katex.renderToString(string): string;
        elem.style.position = 'absolute';
        elem.style.transform = 'translate(-50%, -55%) scale(0.9, 0.9)';
        elem.style.color = color;
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


function lerp(a, b, t){
    return [a[0] * t + b[0] * (1-t), a[1]*t + b[1]*(1-t)]
}

function isElementInViewport (el) {
    var rect = el.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) && 
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}


function flownetwork(elem){
    let cns = document.getElementById(elem);
    let ctx = cns.getContext("2d");
    let span = document.getElementById(elem+"_div");
    span.style.position = 'relative';

    let tex = texBox(span);
    var t = 0;
    let w = cns.width, h = cns.height;

    let N = 60 * 8;
    let nodes = [[0,0],
                 [4,0],
                 [4,3],
                 [8,0],
                 [8,5],
                 [12,2],
                 [6, 8],
                 [8.5, 8],
                 [11, 8],
                 [16, 0],
                 [17, 3],
                 [18, 7],
                 [22, 1.5]];
    
    let edges = [[0,1],[0,2],[1,3],[2,3],[2,4],[3,5],
                 [4, 6], [4, 7],[4, 8], [5,9],[9,12], [5,10],[10,12],[5,11]];
    let edgeflows = [1,2,1,1,1,2,0.33,0.33,0.33,1,1,0.05,0.05,0.5];
    let edgeparticles = new Array(edges.length);
    let outdegrees = new Array(nodes.length).fill(0);
    for (i in edges){
        outdegrees[edges[i][0]] += 1;
        edgeparticles[i] = new Array(Math.floor(20 * edgeflows[i]));
        for (j=0;j<edgeparticles[i].length;j++){
            edgeparticles[i][j] = [Math.random(), Math.random()-0.5];
        }
    }
    
    let ppi = 15;
    function node(p){
        drawCircle(p[0]*ppi, p[1]*ppi, ppi, '#000', '#fff');
    }
    function source(p){
        drawPoly(ctx, [[(p[0]-2/3)*ppi, (p[1]-1)*ppi],
                       [(p[0]-2/3)*ppi, (p[1]+1)*ppi],
                       [(p[0]+1)*ppi, p[1]*ppi]], '#fff', 1, [0,0]);
    }
    function sink(p){
        _ctx.fillStyle = '#fff';
        let q = 3/4;
        ctx.fillRect((p[0]-q)*ppi, (p[1]-q)*ppi, ppi * 2*q, ppi*2*q);
        ctx.strokeRect((p[0]-q)*ppi, (p[1]-q)*ppi, ppi * 2*q, ppi*2*q);
    }
    function edge(i, o, d){
        let lw = 4;
        _ctx.lineWidth = lw * 2;
        let u = [o[0] * ppi, o[1] * ppi]
        let v = [d[0] * ppi, d[1] * ppi]
        drawArrow(u[0], u[1], v[0], v[1], 0, '#aaa')
        _ctx.lineWidth = 1;
        let pts = edgeparticles[i];
        _ctx.fillStyle = '#00f';
        let da = [v[0]-u[0], v[1]-u[1]]
        let norm = Math.sqrt(da[0]*da[0] + da[1]*da[1]);
        let db = [da[1]/norm, da[0]/norm]; //rotated 90d
        for (p in pts){
            ctx.fillRect(pts[p][0] * da[0]+u[0]+pts[p][1] * db[0] * lw,
                         pts[p][0] * da[1]+u[1]+pts[p][1] * db[1] * lw, 2, 2);
            pts[p][0] = (pts[p][0] + 0.01) - Math.floor(pts[p][0] + 0.0);
        }
    }
    let orig = [15, 25];
    function frame(){
        if (!isElementInViewport(cns)){
            requestAnimationFrame(frame); return;}
        ctx.clearRect(0, 0, w, h); 
        t = (t + 1) % N;
        _ctx = ctx;
        ctx.save()
        ctx.translate(orig[0], orig[1]);
        for (e in edges){
            edge(e, nodes[edges[e][0]], nodes[edges[e][1]])
        }
        for (n in nodes){
            if (n == 0) source(nodes[n])
            else if (outdegrees[n] > 0) node(nodes[n])
            else sink(nodes[n])
            
            tex('s'+n, 's_{'+n+'}').moveTo(nodes[n][0]*ppi+orig[0], nodes[n][1]*ppi+orig[1]);
        }
        ctx.restore()
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}


function flownetworkEq(elem){
    let cns = document.getElementById(elem);
    let ctx = cns.getContext("2d");
    let span = document.getElementById(elem+"_div");
    span.style.position = 'relative';

    let tex = texBox(span);
    let w = cns.width, h = cns.height;

    let nodes = [[0,0],
                 [0,3],
                 [0,6],
                 [4,3],
                 [8,0],
                 [8,3],
                 [8,6],];
    
    let edges = [[0,3],[1,3],[2,3],[3,4],[3,5],[3,6]];
    let acts = [1,7,3,4,2,8];
    
    let ppi = 15;
    function node(p){
        drawCircle(p[0]*ppi, p[1]*ppi, ppi, '#000');
    }
    let orig = [15, 25];
    function frame(){
        ctx.clearRect(0, 0, w, h); 
        _ctx = ctx;
        ctx.save()
        ctx.translate(orig[0], orig[1]);
        for (e in edges){
            o = nodes[edges[e][0]];
            d = nodes[edges[e][1]];
            let u = [o[0] * ppi, o[1] * ppi]
            let v = [d[0] * ppi, d[1] * ppi]
            tex('a'+e, 'a_'+acts[e]).moveTo(u[0] + (v[0]-u[0]) / 2+orig[0],
                                            u[1] + (v[1]-u[1]) / 2+orig[1]-10);
            let r = ppi / Math.sqrt(Math.pow(v[0]-u[0],2)+Math.pow(v[1]-u[1],2));
            v[0] = v[0] - r * (v[0]-u[0]);
            v[1] = v[1] - r * (v[1]-u[1]);
            u[0] = u[0] + r * (v[0]-u[0])*1.15;
            u[1] = u[1] + r * (v[1]-u[1])*1.15;
            drawArrow(u[0], u[1], v[0], v[1], 8, '#000')
        }
        for (n in nodes){
            node(nodes[n])            
            tex('s'+n, 's_{'+n+'}').moveTo(nodes[n][0]*ppi+orig[0], nodes[n][1]*ppi+orig[1]);
        }
        ctx.restore()
        //requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

// Here's what it would look like for graphs where the alphabet is $a,b,c$:

// \centered{\canvas{molAlpha}{200}{135}}

function molAlpha(elem){
    let cns = document.getElementById(elem);
    let ctx = cns.getContext("2d");
    let span = document.getElementById(elem+"_div");
    span.style.position = 'relative';

    let tex = texBox(span);
    let w = cns.width, h = cns.height;

    let nodes = [[0,0],
                 [0,3],
                 [0,6],
                 [4,3],
                 [8,0],
                 [8,3],
                 [8,6],];
    
    let edges = [[0,3],[1,3],[2,3],[3,4],[3,5],[3,6]];
    let acts = [1,7,3,4,2,8];
    
    let ppi = 15;
    function node(p){
        drawCircle(p[0]*ppi, p[1]*ppi, ppi, '#000');
    }
    let orig = [15, 25];
    function frame(){
        ctx.clearRect(0, 0, w, h); 
        _ctx = ctx;
        ctx.save()
        ctx.translate(orig[0], orig[1]);
        for (e in edges){
            o = nodes[edges[e][0]];
            d = nodes[edges[e][1]];
            let u = [o[0] * ppi, o[1] * ppi]
            let v = [d[0] * ppi, d[1] * ppi]
            tex('a'+e, 'a_'+acts[e]).moveTo(u[0] + (v[0]-u[0]) / 2+orig[0],
                                            u[1] + (v[1]-u[1]) / 2+orig[1]-10);
            let r = ppi / Math.sqrt(Math.pow(v[0]-u[0],2)+Math.pow(v[1]-u[1],2));
            v[0] = v[0] - r * (v[0]-u[0]);
            v[1] = v[1] - r * (v[1]-u[1]);
            u[0] = u[0] + r * (v[0]-u[0])*1.15;
            u[1] = u[1] + r * (v[1]-u[1])*1.15;
            drawArrow(u[0], u[1], v[0], v[1], 8, '#000')
        }
        for (n in nodes){
            node(nodes[n])            
            tex('s'+n, 's_{'+n+'}').moveTo(nodes[n][0]*ppi+orig[0], nodes[n][1]*ppi+orig[1]);
        }
        ctx.restore()
        //requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

function flownetworkBigger(elem){
    let cns = document.getElementById(elem);
    let ctx = cns.getContext("2d");
    let span = document.getElementById(elem+"_div");
    span.style.position = 'relative';

    let tex = texBox(span);
    var t = 0;
    let w = cns.width, h = cns.height;

    let N = 60 * 8;
    let nodes = [[0,0], // s0
                 [4,0], // s1 s2
                 [4,4],
                 [8,0],  // s3
                 [10,3], // s3 T
                 [8,5],  // s5
                 [10,8], // s5 T
                 [12, 0], // s7
                 [12, 5], // s8
                 [14, 8], // s8 T
                 [16, 0], // 10
                 [16, 3], // 11
                 [18, 6], // 11 T
                 [20, 1.5], // 13
                 [22, 4.5], // 13 T
                 [16, 6.5], // 15
                 [20, 9], // 16
                 [22, 12], // 16 T
                ];

    let terminatedNodes = [4,6,9,12,14,17];
    //let naturalEnds = [13, 15];
    let edges = [[0,1],[0,2],[1,3],[2,3],[3,4],
                 [2,5],[5,6],[3,7],[5,8],[8,9],
                 [7,10],[7,11],[8,11],[11,12],[10,13],
                 [11,13],[13,14],[8,15],[15,16],[16,17],
                 //[4, 6], [4, 7],[4, 8], [5,9],[9,12], [5,10],[10,12],[5,11]
                ];
    let edgeflows = [2,3,2,1,1.5,
                     2,0.5,1.5,1.5,1,
                     1,0.5,0.2,0.3,1,
                     0.4,1.4,0.3,0.3,0.3,
                    ];
    let edgeparticles = new Array(edges.length);
    let outdegrees = new Array(nodes.length).fill(0);
    for (i in edges){
        outdegrees[edges[i][0]] += 1;
        edgeparticles[i] = new Array(Math.floor(50 * edgeflows[i]));
        for (j=0;j<edgeparticles[i].length;j++){
            edgeparticles[i][j] = [Math.random(), Math.random()-0.5];
        }
    }
    
    let ppi = 15;
    function node(p){
        drawCircle(p[0]*ppi, p[1]*ppi, ppi, '#000', '#fff');
    }
    function source(p){
        drawPoly(ctx, [[(p[0]-2/3)*ppi, (p[1]-1)*ppi],
                       [(p[0]-2/3)*ppi, (p[1]+1)*ppi],
                       [(p[0]+1)*ppi, p[1]*ppi]], '#fff', 1, [0,0]);
    }
    function sink(p){
        _ctx.fillStyle = '#fff';
        let q = 3/4;
        ctx.fillRect((p[0]-q)*ppi, (p[1]-q)*ppi, ppi * 2*q, ppi*2*q);
        ctx.strokeRect((p[0]-q)*ppi, (p[1]-q)*ppi, ppi * 2*q, ppi*2*q);
    }
    function edge(i, o, d){
        let lw = 4;
        _ctx.lineWidth = lw * 2;
        let u = [o[0] * ppi, o[1] * ppi]
        let v = [d[0] * ppi, d[1] * ppi]
        drawArrow(u[0], u[1], v[0], v[1], 0,
                  terminatedNodes.indexOf(edges[i][1]) == -1 ? '#aaa' : '#c88');
        _ctx.lineWidth = 1;
        let pts = edgeparticles[i];
        _ctx.fillStyle = '#00f5';
        let da = [v[0]-u[0], v[1]-u[1]]
        let norm = Math.sqrt(da[0]*da[0] + da[1]*da[1]);
        let db = [-da[1]/norm, da[0]/norm]; //rotated 90d
        for (p=0;p<pts.length;p++){
            ctx.fillRect(pts[p][0] * da[0]+u[0]+pts[p][1] * db[0] * lw*1.5-1,
                         pts[p][0] * da[1]+u[1]+pts[p][1] * db[1] * lw*1.5-1, 2, 2);
            pts[p][0] = (pts[p][0] + 0.01) - Math.floor(pts[p][0] + 0.0);
        }
    }
    let orig = [15, 25];
    for (n=0;n<nodes.length;n++){
        if (terminatedNodes.indexOf(n) < 0)
            tex('s'+n, 's_{'+n+'}').moveTo(nodes[n][0]*ppi+orig[0], nodes[n][1]*ppi+orig[1]);
        else{
            tex('x'+n, 'x_{'+(n-1)+'}').moveTo(nodes[n][0]*ppi+orig[0], nodes[n][1]*ppi+orig[1]);
            tex('top'+n, '\\top').moveTo(nodes[n][0]*ppi+orig[0]-5, nodes[n][1]*ppi+orig[1]-25);
        }
        /*if (naturalEnds.indexOf(n) >= 0){
            tex('ne'+n, '\\equiv x_{'+n+'}').moveTo(nodes[n][0]*ppi+orig[0]+35, nodes[n][1]*ppi+orig[1]);
        }*/
    }
    /*
    for (i=0;i<edges.length;i++){
        let e = edges[i];
        tex('f'+e, edgeflows[i]+'', undefined, '#080'
           ).moveTo((nodes[e[0]][0]+nodes[e[1]][0])/2*ppi+orig[0],
                    (nodes[e[0]][1]+nodes[e[1]][1])/2*ppi+orig[1])
        
    }*/
    function frame(){
        if (!isElementInViewport(cns)){
            requestAnimationFrame(frame); return;}
        ctx.clearRect(0, 0, w, h); 
        t = (t + 1) % N;
        _ctx = ctx;
        ctx.save()
        ctx.translate(orig[0], orig[1]);
        for (e in edges){
            edge(e, nodes[edges[e][0]], nodes[edges[e][1]])
        }
        for (n in nodes){
            if (n == 0) source(nodes[n])
            else if (outdegrees[n] > 0) node(nodes[n])
            else sink(nodes[n])
        }
        ctx.restore()
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}
