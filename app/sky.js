/* ============================================================
   STARRY NIGHT — Van Gogh animated canvas background
   Swirling nebulae, twinkling stars, crescent moon, cypress,
   rolling hills and village silhouette.
   ============================================================ */

(function () {
  var canvas = document.getElementById("sky");
  if (!canvas) return;
  var ctx = canvas.getContext("2d");
  var stars = [];
  var bigStars = [];

  /* ---- resize & populate ---- */
  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    stars = [];
    bigStars = [];
    var n = Math.floor((canvas.width * canvas.height) / 12000);
    for (var i = 0; i < n; i++) {
      stars.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height * 0.75,
        r: Math.random() * 1.5 + 0.3,
        phase: Math.random() * Math.PI * 2,
        speed: 0.3 + Math.random() * 1.6,
        hue: Math.random() < 0.3 ? 1 : 0, // 1 = golden, 0 = white-blue
      });
    }
    // A handful of large bright stars with halos
    for (var j = 0; j < 8; j++) {
      bigStars.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height * 0.55,
        r: 2.5 + Math.random() * 3,
        phase: Math.random() * Math.PI * 2,
        speed: 0.2 + Math.random() * 0.8,
      });
    }
  }

  /* ---- Van Gogh swirl ---- */
  function swirl(cx, cy, rx, ry, rotation, phase, color, width, opacity) {
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(rotation);
    ctx.globalAlpha = opacity || 0.18;

    // Multiple layered spirals for the impasto brush-stroke feel
    for (var layer = 0; layer < 3; layer++) {
      ctx.beginPath();
      var layerPhase = phase + layer * 0.4;
      var layerScale = 1 - layer * 0.12;
      for (var a = 0; a <= Math.PI * 2.6; a += 0.06) {
        var k = a / (Math.PI * 2);
        var wobble = Math.sin(a * 3.5 + layerPhase * 2) * 4;
        var x = Math.cos(a + layerPhase) * rx * (0.3 + k) * layerScale + wobble;
        var y = Math.sin(a * 1.1 + layerPhase * 0.6) * ry * (0.4 + 0.45 * Math.sin(a + layerPhase)) * layerScale;
        if (a === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = width - layer * 2;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.stroke();
    }
    ctx.restore();
  }

  /* ---- Crescent Moon with halo ---- */
  function moon(time) {
    var x = canvas.width * 0.84;
    var y = canvas.height * 0.13;
    var pulse = Math.sin(time * 0.0003) * 6;

    // Outer halo
    var g3 = ctx.createRadialGradient(x, y, 10, x, y, 120 + pulse);
    g3.addColorStop(0, "rgba(255, 244, 196, 0.3)");
    g3.addColorStop(0.4, "rgba(242, 199, 68, 0.08)");
    g3.addColorStop(1, "rgba(242, 199, 68, 0)");
    ctx.globalAlpha = 1;
    ctx.fillStyle = g3;
    ctx.beginPath();
    ctx.arc(x, y, 120 + pulse, 0, Math.PI * 2);
    ctx.fill();

    // Inner bright halo
    var g2 = ctx.createRadialGradient(x, y, 8, x, y, 55);
    g2.addColorStop(0, "rgba(255, 248, 220, 0.9)");
    g2.addColorStop(0.5, "rgba(242, 199, 68, 0.4)");
    g2.addColorStop(1, "rgba(242, 199, 68, 0)");
    ctx.fillStyle = g2;
    ctx.beginPath();
    ctx.arc(x, y, 55, 0, Math.PI * 2);
    ctx.fill();

    // Moon crescent
    ctx.fillStyle = "#fff4cc";
    ctx.beginPath();
    ctx.arc(x, y, 24, 0, Math.PI * 2);
    ctx.fill();

    // Crescent shadow
    ctx.fillStyle = "#1a3a6a";
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    ctx.arc(x + 10, y - 4, 20, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  /* ---- Draw large star with Van Gogh halo ---- */
  function drawBigStar(s, time) {
    var twinkle = 0.5 + 0.5 * Math.sin(time * 0.001 * s.speed + s.phase);
    var r = s.r * (0.8 + twinkle * 0.4);

    // Concentric rings (Van Gogh signature style)
    for (var ring = 3; ring >= 0; ring--) {
      var ringR = r + ring * 8;
      var alpha = (0.06 - ring * 0.012) * twinkle;
      ctx.globalAlpha = Math.max(0, alpha);
      ctx.strokeStyle = ring % 2 === 0 ? "#ffe58a" : "#8fd0e8";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(s.x, s.y, ringR, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Core glow
    var g = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, r * 4);
    g.addColorStop(0, "rgba(255, 244, 196, " + (0.6 * twinkle) + ")");
    g.addColorStop(0.5, "rgba(242, 199, 68, " + (0.15 * twinkle) + ")");
    g.addColorStop(1, "rgba(242, 199, 68, 0)");
    ctx.globalAlpha = 1;
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(s.x, s.y, r * 4, 0, Math.PI * 2);
    ctx.fill();

    // Bright core
    ctx.fillStyle = "#fff8e0";
    ctx.globalAlpha = 0.7 + twinkle * 0.3;
    ctx.beginPath();
    ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  /* ---- Cypress tree silhouette (signature Van Gogh element) ---- */
  function cypress(xPos, heightFrac, widthScale) {
    var h = canvas.height;
    var w = widthScale || 1;
    ctx.save();
    ctx.translate(xPos, h);
    ctx.globalAlpha = 0.95;

    // Dark cypress shape with wavy edges
    ctx.beginPath();
    ctx.moveTo(0, 0);
    var segments = 20;
    for (var i = 0; i <= segments; i++) {
      var t = i / segments;
      var yy = -t * h * heightFrac;
      var spread = Math.sin(t * Math.PI) * 22 * w;
      var wave = Math.sin(t * 12 + performance.now() * 0.0008) * 3;
      ctx.lineTo(-spread + wave, yy);
    }
    for (var j = segments; j >= 0; j--) {
      var t2 = j / segments;
      var yy2 = -t2 * h * heightFrac;
      var spread2 = Math.sin(t2 * Math.PI) * 22 * w;
      var wave2 = Math.sin(t2 * 12 + performance.now() * 0.0008 + 1) * 3;
      ctx.lineTo(spread2 + wave2, yy2);
    }
    ctx.closePath();

    var cg = ctx.createLinearGradient(0, 0, 0, -h * heightFrac);
    cg.addColorStop(0, "#0a1a0c");
    cg.addColorStop(0.5, "#0c2214");
    cg.addColorStop(1, "#071008");
    ctx.fillStyle = cg;
    ctx.fill();
    ctx.restore();
  }

  /* ---- Rolling hills ---- */
  function hills() {
    var w = canvas.width;
    var h = canvas.height;

    // Far hills
    ctx.globalAlpha = 0.7;
    ctx.beginPath();
    ctx.moveTo(0, h * 0.82);
    for (var x = 0; x <= w; x += 4) {
      var y = h * 0.82 - Math.sin(x * 0.004) * 20 - Math.sin(x * 0.011) * 8;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    var hg = ctx.createLinearGradient(0, h * 0.7, 0, h);
    hg.addColorStop(0, "#1a3a2a");
    hg.addColorStop(0.4, "#142a1e");
    hg.addColorStop(1, "#0a1810");
    ctx.fillStyle = hg;
    ctx.fill();

    // Near hills
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    ctx.moveTo(0, h * 0.88);
    for (var x2 = 0; x2 <= w; x2 += 3) {
      var y2 = h * 0.88 - Math.sin(x2 * 0.006 + 1) * 14 - Math.sin(x2 * 0.015) * 6;
      ctx.lineTo(x2, y2);
    }
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    var hg2 = ctx.createLinearGradient(0, h * 0.82, 0, h);
    hg2.addColorStop(0, "#122618");
    hg2.addColorStop(1, "#0a180e");
    ctx.fillStyle = hg2;
    ctx.fill();
  }

  /* ---- Village silhouette ---- */
  function village() {
    var h = canvas.height;
    var w = canvas.width;

    // Ground
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = "#0c180e";
    ctx.fillRect(0, h * 0.9, w, h * 0.1);

    // Buildings
    ctx.fillStyle = "#1a140c";
    ctx.globalAlpha = 0.8;
    var buildings = [
      { x: 0.30, bw: 20, bh: 30 },
      { x: 0.34, bw: 14, bh: 22 },
      { x: 0.38, bw: 18, bh: 35 },
      { x: 0.42, bw: 22, bh: 28 },
      { x: 0.47, bw: 16, bh: 24 },
      { x: 0.51, bw: 24, bh: 32 },
      { x: 0.56, bw: 14, bh: 20 },
      { x: 0.60, bw: 20, bh: 38 },
      { x: 0.64, bw: 18, bh: 26 },
      { x: 0.68, bw: 16, bh: 22 },
    ];
    for (var i = 0; i < buildings.length; i++) {
      var b = buildings[i];
      ctx.fillRect(w * b.x, h * 0.9 - b.bh, b.bw, b.bh);
    }

    // Church steeple
    ctx.beginPath();
    ctx.moveTo(w * 0.55, h * 0.9 - 55);
    ctx.lineTo(w * 0.555, h * 0.9 - 32);
    ctx.lineTo(w * 0.545, h * 0.9 - 32);
    ctx.closePath();
    ctx.fill();

    // Warm window lights
    ctx.fillStyle = "#e6c15a";
    ctx.globalAlpha = 0.5;
    for (var j = 0; j < 8; j++) {
      ctx.beginPath();
      ctx.arc(w * (0.32 + j * 0.05), h * 0.885, 1.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  /* ---- Main render loop ---- */
  function frame(now) {
    var w = canvas.width;
    var h = canvas.height;

    // Gradient sky
    var sky = ctx.createLinearGradient(0, 0, 0, h * 0.85);
    sky.addColorStop(0, "#0a1628");
    sky.addColorStop(0.2, "#122a52");
    sky.addColorStop(0.4, "#1a3a6a");
    sky.addColorStop(0.6, "#2a5a9a");
    sky.addColorStop(0.8, "#1a4878");
    sky.addColorStop(1, "#0c2818");
    ctx.globalAlpha = 1;
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, w, h);

    // Animated swirling nebulae — the signature Van Gogh effect
    var ph = now * 0.00015;
    swirl(w * 0.30, h * 0.18, w * 0.24, h * 0.09, -0.3, ph,         "#4a8ac4", 18, 0.15);
    swirl(w * 0.52, h * 0.14, w * 0.30, h * 0.08,  0.15, ph + 1.0,  "#6ab4e8", 14, 0.12);
    swirl(w * 0.65, h * 0.32, w * 0.32, h * 0.12,  0.4, ph + 2.0,   "#2a5a9a", 20, 0.14);
    swirl(w * 0.22, h * 0.38, w * 0.20, h * 0.08, -0.6, ph + 0.5,   "#8fd0e8", 12, 0.10);
    swirl(w * 0.80, h * 0.24, w * 0.18, h * 0.07,  0.1, ph + 1.5,   "#ffe58a", 10, 0.09);
    swirl(w * 0.44, h * 0.26, w * 0.26, h * 0.09, -0.2, ph + 2.8,   "#f2c744", 12, 0.08);
    // Additional smaller swirls for richness
    swirl(w * 0.15, h * 0.12, w * 0.10, h * 0.04, 0.8, ph + 3.5,    "#6ab4e8", 8, 0.07);
    swirl(w * 0.75, h * 0.42, w * 0.14, h * 0.05, -0.5, ph + 4.2,   "#4a8ac4", 10, 0.08);

    // Moon
    moon(now);

    // Big stars with halos
    for (var b = 0; b < bigStars.length; b++) {
      drawBigStar(bigStars[b], now);
    }

    // Small twinkling stars
    for (var i = 0; i < stars.length; i++) {
      var s = stars[i];
      var tw = 0.25 + 0.75 * Math.abs(Math.sin(now * 0.001 * s.speed + s.phase));
      ctx.globalAlpha = tw;
      ctx.fillStyle = s.hue ? "#ffe58a" : "#e8f0ff";
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r * (0.7 + tw * 0.3), 0, Math.PI * 2);
      ctx.fill();
    }

    // Landscape
    hills();
    village();

    // Cypress trees
    cypress(w * 0.07, 0.65, 1);
    cypress(w * 0.12, 0.42, 0.7);

    requestAnimationFrame(frame);
  }

  window.addEventListener("resize", resize);
  resize();
  requestAnimationFrame(frame);
})();
