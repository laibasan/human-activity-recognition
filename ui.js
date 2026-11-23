(function () {
  var names = (window.HARNET && window.HARNET.activity_names) || [];
  var pretty = names.map(function (n) {
    return n.replace(/_/g, " ").replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  });
  var select = document.getElementById("sample");
  var bars = document.getElementById("bars");
  var samples = (window.HAR_SAMPLES && window.HAR_SAMPLES.samples) || [];

  samples.forEach(function (s, i) {
    var opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = pretty[s.y] + "  (test window " + s.i + ")";
    select.appendChild(opt);
  });

  pretty.forEach(function (label, i) {
    var row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML =
      "<span>" + label + "</span><div class='track'><i data-i='" + i + "'></i></div><span data-p='" + i + "'>0%</span>";
    bars.appendChild(row);
  });

  function drawSignals(raw) {
    var cv = document.getElementById("signals");
    var ctx = cv.getContext("2d");
    var w = cv.width;
    var h = cv.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "rgba(8,16,40,0.2)";
    ctx.fillRect(0, 0, w, h);
    var colors = ["#8fd0e8", "#f2d56b", "#e7a08a"];
    var min = Infinity, max = -Infinity;
    for (var t = 0; t < raw.length; t++) {
      for (var c = 0; c < 3; c++) {
        var v = raw[t][c];
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    if (max === min) max = min + 1;
    function xAt(i) { return 16 + (i / (raw.length - 1)) * (w - 32); }
    function yAt(v) { return 16 + (1 - (v - min) / (max - min)) * (h - 32); }
    for (var ch = 0; ch < 3; ch++) {
      ctx.beginPath();
      ctx.strokeStyle = colors[ch];
      ctx.lineWidth = 1.6;
      for (var i = 0; i < raw.length; i++) {
        var x = xAt(i);
        var y = yAt(raw[i][ch]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.fillStyle = "#b7c6dc";
    ctx.font = "12px Georgia";
    ctx.fillText("body acc x / y / z", 16, 14);
  }

  function drawAttn(alpha) {
    var cv = document.getElementById("attn");
    var ctx = cv.getContext("2d");
    var w = cv.width;
    var h = cv.height;
    ctx.clearRect(0, 0, w, h);
    var peak = 0;
    for (var i = 0; i < alpha.length; i++) if (alpha[i] > peak) peak = alpha[i];
    var top = Math.max(peak * 1.15, 1 / 64);
    ctx.beginPath();
    ctx.moveTo(16, h - 16);
    for (var t = 0; t < alpha.length; t++) {
      var x = 16 + (t / (alpha.length - 1)) * (w - 32);
      var y = 12 + (1 - alpha[t] / top) * (h - 28);
      ctx.lineTo(x, y);
    }
    ctx.lineTo(w - 16, h - 16);
    ctx.closePath();
    var g = ctx.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, "rgba(242,213,107,0.55)");
    g.addColorStop(1, "rgba(74,144,195,0.12)");
    ctx.fillStyle = g;
    ctx.fill();
    ctx.strokeStyle = "#f2d56b";
    ctx.stroke();
    ctx.fillStyle = "#b7c6dc";
    ctx.font = "12px Georgia";
    ctx.fillText("alpha over 2.56 s", 16, 14);
  }

  function show(result, raw) {
    drawSignals(raw);
    drawAttn(result.alpha);
    var p = result.pred;
    document.getElementById("guess").textContent = pretty[p];
    document.getElementById("conf").textContent = (result.probs[p] * 100).toFixed(1) + "%";
    for (var i = 0; i < result.probs.length; i++) {
      var width = (result.probs[i] * 100).toFixed(1);
      bars.querySelector("i[data-i='" + i + "']").style.width = width + "%";
      bars.querySelector("span[data-p='" + i + "']").textContent = width + "%";
    }
    var mx = 0;
    for (var a = 0; a < result.alpha.length; a++) if (result.alpha[a] > mx) mx = result.alpha[a];
    document.getElementById("peak").textContent =
      "Largest alpha in this window: " + mx.toFixed(3) +
      ". A completely flat look would be 1/128 ≈ 0.0078.";
    document.getElementById("status").textContent = "Done.";

    var icons = document.querySelectorAll(".act-icon");
    icons.forEach(function (ic, idx) {
      if (idx === p) {
        ic.style.transform = "scale(1.35)";
        ic.style.opacity = "1";
        ic.style.filter = "drop-shadow(0 0 10px rgba(242, 199, 68, 0.8))";
      } else {
        ic.style.transform = "scale(1)";
        ic.style.opacity = "0.4";
        ic.style.filter = "none";
      }
    });
  }

  function rawFromScaled(X) {
    var mean = window.HARNET.mean;
    var std = window.HARNET.std;
    return X.map(function (row) {
      return row.map(function (v, c) { return v * std[c] + mean[c]; });
    });
  }

  function runSample() {
    if (!samples.length) {
      document.getElementById("status").textContent = "No samples.js yet. Train, then export_web.py.";
      return;
    }
    document.getElementById("status").textContent = "Walking the sequence…";
    var s = samples[Number(select.value)];
    setTimeout(function () {
      var out = window.harInfer.predict(s.X);
      show(out, s.X_raw || rawFromScaled(s.X));
    }, 20);
  }

  document.getElementById("run").addEventListener("click", runSample);
  select.addEventListener("change", runSample);

  document.getElementById("csv").addEventListener("change", function (ev) {
    var file = ev.target.files[0];
    if (!file) return;
    var reader = new FileReader();
    reader.onload = function () {
      var lines = String(reader.result).trim().split(/\r?\n/);
      var raw = [];
      for (var i = 0; i < lines.length; i++) {
        var line = lines[i].trim();
        if (!line || line[0] === "#" || /[a-zA-Z]/.test(line)) continue;
        var parts = line.split(/[,;\s]+/).map(Number).filter(function (x) { return !Number.isNaN(x); });
        if (parts.length >= 9) raw.push(parts.slice(0, 9));
      }
      if (raw.length !== 128) {
        document.getElementById("status").textContent =
          "Need 128 rows of 9 numbers, got " + raw.length + ".";
        return;
      }
      document.getElementById("status").textContent = "Walking the uploaded window…";
      setTimeout(function () {
        var X = window.harInfer.scaleWindow(raw);
        show(window.harInfer.predict(X), raw);
      }, 20);
    };
    reader.readAsText(file);
  });

  if (samples.length) runSample();
})();
