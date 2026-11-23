/* HARNet in the browser. Same layout as src/layers.py / src/model.py. */

(function (global) {
  function tanh(x) {
    if (x > 20) return 1;
    if (x < -20) return -1;
    var e = Math.exp(2 * x);
    return (e - 1) / (e + 1);
  }
  function sigmoid(x) {
    if (x >= 0) {
      var z = Math.exp(-x);
      return 1 / (1 + z);
    }
    var z2 = Math.exp(x);
    return z2 / (1 + z2);
  }

  function zeros(n) {
    var a = new Array(n);
    for (var i = 0; i < n; i++) a[i] = 0;
    return a;
  }

  function gemv(W, x, b) {
    var rows = W.length;
    var cols = x.length;
    var out = new Array(rows);
    for (var i = 0; i < rows; i++) {
      var s = b ? b[i] : 0;
      var row = W[i];
      for (var j = 0; j < cols; j++) s += row[j] * x[j];
      out[i] = s;
    }
    return out;
  }

  function lstmCell(cell, x, h, c) {
    var H = h.length;
    var acc = gemv(cell.W, x, cell.b);
    var rec = gemv(cell.U, h, null);
    var i, f, g, o, cNew, hNew;
    i = new Array(H);
    f = new Array(H);
    g = new Array(H);
    o = new Array(H);
    cNew = new Array(H);
    hNew = new Array(H);
    for (var k = 0; k < H; k++) {
      var z = acc[k] + rec[k];
      i[k] = sigmoid(z);
      z = acc[H + k] + rec[H + k];
      f[k] = sigmoid(z);
      z = acc[2 * H + k] + rec[2 * H + k];
      g[k] = tanh(z);
      z = acc[3 * H + k] + rec[3 * H + k];
      o[k] = sigmoid(z);
      cNew[k] = f[k] * c[k] + i[k] * g[k];
      hNew[k] = o[k] * tanh(cNew[k]);
    }
    return { h: hNew, c: cNew };
  }

  function runLSTM(cell, seq) {
    var T = seq.length;
    var H = cell.U.length / 4;
    var h = zeros(H);
    var c = zeros(H);
    var outs = new Array(T);
    for (var t = 0; t < T; t++) {
      var step = lstmCell(cell, seq[t], h, c);
      h = step.h;
      c = step.c;
      outs[t] = h;
    }
    return outs;
  }

  function reverseRows(seq) {
    var out = seq.slice();
    out.reverse();
    return out;
  }

  function concat(a, b) {
    var o = new Array(a.length + b.length);
    for (var i = 0; i < a.length; i++) o[i] = a[i];
    for (var j = 0; j < b.length; j++) o[a.length + j] = b[j];
    return o;
  }

  function softmax(e) {
    var m = e[0];
    for (var i = 1; i < e.length; i++) if (e[i] > m) m = e[i];
    var s = 0;
    var out = new Array(e.length);
    for (var k = 0; k < e.length; k++) {
      out[k] = Math.exp(e[k] - m);
      s += out[k];
    }
    for (var n = 0; n < e.length; n++) out[n] /= s;
    return out;
  }

  function attention(hSeq, Wa, ba, v) {
    var T = hSeq.length;
    var e = new Array(T);
    var attnH = v.length;
    for (var t = 0; t < T; t++) {
      var h = hSeq[t];
      var s = 0;
      for (var i = 0; i < attnH; i++) {
        var u = ba[i];
        var row = Wa[i];
        for (var j = 0; j < h.length; j++) u += row[j] * h[j];
        s += v[i] * tanh(u);
      }
      e[t] = s;
    }
    var alpha = softmax(e);
    var dim = hSeq[0].length;
    var ctx = zeros(dim);
    for (var t2 = 0; t2 < T; t2++) {
      var a = alpha[t2];
      var ht = hSeq[t2];
      for (var d = 0; d < dim; d++) ctx[d] += a * ht[d];
    }
    return { context: ctx, alpha: alpha };
  }

  function predict(X) {
    var net = global.HARNET;
    var w = net.weights;
    var fwd = {
      W: w["bilstm.fwd.cell.W"],
      U: w["bilstm.fwd.cell.U"],
      b: w["bilstm.fwd.cell.b"],
    };
    var bwd = {
      W: w["bilstm.bwd.cell.W"],
      U: w["bilstm.bwd.cell.U"],
      b: w["bilstm.bwd.cell.b"],
    };
    var hFwd = runLSTM(fwd, X);
    var hBwdRev = runLSTM(bwd, reverseRows(X));
    var hBwd = reverseRows(hBwdRev);
    var h = new Array(X.length);
    for (var t = 0; t < X.length; t++) h[t] = concat(hFwd[t], hBwd[t]);
    var att = attention(h, w["attn.Wa"], w["attn.ba"], w["attn.v"]);
    var logits = gemv(w["fc.weight"], att.context, w["fc.bias"]);
    var probs = softmax(logits);
    var pred = 0;
    for (var i = 1; i < probs.length; i++) if (probs[i] > probs[pred]) pred = i;
    return { logits: logits, probs: probs, pred: pred, alpha: att.alpha };
  }

  function scaleWindow(raw) {
    var mean = global.HARNET.mean;
    var std = global.HARNET.std;
    var out = new Array(raw.length);
    for (var t = 0; t < raw.length; t++) {
      var row = raw[t];
      var s = new Array(row.length);
      for (var c = 0; c < row.length; c++) s[c] = (row[c] - mean[c]) / std[c];
      out[t] = s;
    }
    return out;
  }

  global.harInfer = { predict: predict, scaleWindow: scaleWindow, softmax: softmax };
})(window);
