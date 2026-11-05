// Motore di animazione 2D del cartone: chiavi interpolate con le eases di GSAP, pupazzi SVG ridisegnati a ogni
// fotogramma, lip-sync dal testo + tempi della timeline, battiti di ciglia, "idle" di recitazione, camera.
// Tutto e' funzione del tempo t (deterministico): HyperFrames puo' cercare qualsiasi fotogramma.
(function (root) {
  const { Chars } = root;
  const E = {};
  const ease = (n) => (n ? (E[n] ??= root.gsap ? root.gsap.parseEase(n) : (x) => x) : (x) => x * x * (3 - 2 * x));

  // ---------------------------------------------------------------- chiavi
  // keys: [[t, {param: valore, ...}, "ease"], ...]  — i numeri si interpolano con l'ease della chiave d'arrivo,
  // stringhe/booleani cambiano a scatto. Un parametro non presente in una chiave mantiene il valore precedente.
  function at(keys, t, base = {}) {
    const ks = keys.slice().sort((a, b) => a[0] - b[0]);
    let prev = { ...base }, prevT = -1e9;
    for (const [kt, kv, ke] of ks) {
      const next = { ...prev, ...kv };
      if (t < kt) {
        if (prevT === -1e9) return next;          // prima della prima chiave: la prima posa
        const u = Math.min(1, Math.max(0, (t - prevT) / Math.max(1e-6, kt - prevT)));
        const f = ease(ke)(u), out = { ...prev };
        for (const k in kv) if (typeof kv[k] === "number" && typeof prev[k] === "number") out[k] = prev[k] + (kv[k] - prev[k]) * f;
        return out;
      }
      prev = next; prevT = kt;
    }
    return prev;
  }
  const lerp = (a, b, u) => a + (b - a) * u;
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const osc = (t, f, ph = 0) => Math.sin((t * f + ph) * Math.PI * 2);

  // ---------------------------------------------------------------- lip-sync (italiano: lingua fonetica)
  const V = { a: "A", à: "A", e: "E", è: "E", é: "E", i: "I", ì: "I", o: "O", ò: "O", u: "U", ù: "U" };
  function visemeTrack(line) {
    // tempi delle parole: faster-whisper (line.words) se ci sono, altrimenti per proporzione dentro i segmenti VAD
    const words = line.caption.split(/\s+/).filter(Boolean);
    let wt;
    if (line.words && line.words.length === words.length) wt = line.words.map((w) => [w.start, w.end]);
    else {
      const segs = line.pauses && line.pauses.length ? line.pauses : [[0, line.speech]];
      const total = segs.reduce((s, [a, b]) => s + b - a, 0), chars = words.map((w) => w.replace(/[^\wàèéìòù]/g, "").length + 1);
      const sum = chars.reduce((a, b) => a + b, 0);
      const map = (x) => { let acc = 0; for (const [a, b] of segs) { if (acc + (b - a) >= x) return line.start + a + (x - acc); acc += b - a; } return line.end; };
      let c = 0; wt = chars.map((n) => { const a = map((c / sum) * total); c += n; return [a, map((c / sum) * total) - 0.03]; });
    }
    const track = [];
    words.forEach((w, i) => {
      const [a, b] = wt[i]; const L = w.toLowerCase().replace(/[^a-zàèéìòù]/g, "");
      const shapes = [];
      for (let k = 0; k < L.length; k++) {
        const ch = L[k];
        if (V[ch]) { if (shapes[shapes.length - 1] !== V[ch]) shapes.push(V[ch]); }
        else if ("mbp".includes(ch)) shapes.push("MBP");
        else if ("fv".includes(ch)) shapes.push("FV");
      }
      if (!shapes.length) shapes.push("E");
      const d = (b - a) / shapes.length;
      shapes.forEach((s, k) => track.push([a + k * d, s]));
      track.push([b, "rest"]);
    });
    return track;
  }
  // bocca al tempo t: a "passo due" (15 fps) come nell'animazione tradizionale; fuori dalle battute -> idle
  function mouthAt(lines, t, idle = "rest") {
    const tq = Math.floor(t * 15) / 15;
    for (const l of lines) {
      if (tq < l.start - 0.02 || tq > l.end + 0.05) continue;
      l._vt ??= visemeTrack(l);
      let m = "rest";
      for (const [vt, v] of l._vt) { if (vt <= tq) m = v; else break; }
      return m === "rest" ? (idle === "rest" ? "MBP" : idle) : m;
    }
    return idle;
  }
  // battiti di ciglia: lista di istanti -> fattore d'apertura (chiusura in 2 fotogrammi, riapertura in 3)
  function blinkAt(times, t) {
    for (const b of times) { const d = t - b; if (d >= 0 && d < 0.2) return d < 0.067 ? 0.15 : d < 0.13 ? 0.05 : 0.6; }
    return 1;
  }
  const autoBlinks = (t0, t1, seed = 1, every = 3.1) => { const out = []; let t = t0 + 0.6 + (seed % 5) * 0.37; while (t < t1) { out.push(t); t += every + ((seed * 7 + out.length * 13) % 10) / 10; } return out; };

  // ---------------------------------------------------------------- adattatori pupazzo -> opzioni dei personaggi
  // stato piatto (facile da animare) -> opzioni annidate di Chars.*
  function contratto(s, t) {
    const idle = s.idle ?? 1, breathe = 1 + 0.012 * osc(t, 0.45) * idle;
    return Chars.contratto({
      turn: s.turn ?? 0, view: s.view, squash: (s.sq ?? 1) * breathe, sweat: s.sweat, lf: s.lf, rf: s.rf,
      brow: { tilt: s.bt ?? 0, curve: s.bc ?? 6, y: s.by ?? -296 },
      eyes: { open: (s.open ?? 1) * (s.blink ?? 1), px: s.px ?? 0, py: s.py ?? 0, lid: s.lid ?? 0, happy: s.happy },
      mouth: s.mouth ?? "rest",
      armL: [s.lx ?? -150, (s.ly ?? -120) + 3 * osc(t, 0.45, 0.2) * idle, s.lb ?? 0.3, s.lh ?? "open"],
      armR: [s.rx ?? 150, (s.ry ?? -120) + 3 * osc(t, 0.45, 0.7) * idle, s.rb ?? -0.3, s.rh ?? "open"],
    });
  }
  function mouse(s, t) {
    return Chars.mouse({ eyes: { open: (s.open ?? 1.1) * (s.blink ?? 1), px: s.px ?? 0.3, py: s.py ?? -0.1, lid: s.lid ?? 0 }, mouth: s.mouth ?? "grin", browUp: s.bu ?? 0,
      armL: [s.lx ?? -130, s.ly ?? -150, s.lb ?? 0.35, s.lh ?? "open"], armR: [s.rx ?? 118, s.ry ?? -96, s.rb ?? -0.2, s.rh ?? "open"],
      cup: s.cup ?? true, cupTilt: s.ct ?? 8, sweat: s.sweat ?? true, stride: s.stride, cap: s.cap });
  }
  function chatgratisOld(s, t) {
    return Chars.chatgratis({ eyes: { lid: s.lid ?? 0.42, px: s.px ?? 0.6, py: s.py ?? 0, open: (s.open ?? 1) * (s.blink ?? 1), lidColor: Chars.P.gratis }, mouth: s.mouth ?? "grin",
      armL: [s.lx ?? -210, s.ly ?? -250, s.lb ?? 0.25, s.lh ?? "open"], armR: [s.rx ?? 205, s.ry ?? -255, s.rb ?? -0.25, s.rh ?? "thumb"] });
  }
  function assistenteOld(s, t) {
    return Chars.assistente({ eyes: { open: (s.open ?? 0.9) * (s.blink ?? 1), px: s.px ?? 0, py: s.py ?? 0.1, lid: s.lid ?? 0 }, mouth: s.mouth ?? "smile",
      armL: [s.lx ?? -178, s.ly ?? -128, s.lb ?? 0.3, s.lh ?? "open"], armR: [s.rx ?? 184, s.ry ?? -238, s.rb ?? -0.2, s.rh ?? "thumb"] });
  }
  // geni digitali (design scelto): stesse chiavi di prima (occhi, bocca, braccia)
  function chatgratis(s, t) {
    return root.Chatbots.genieGratis({ eyes: { lid: s.lid ?? 0.4, px: s.px ?? 0.7, py: s.py ?? 0, open: (s.open ?? 1) * (s.blink ?? 1), lidColor: Chars.P.gratis }, mouth: s.mouth ?? "grin",
      armL: [s.lx ?? -206, s.ly ?? -350, s.lb ?? 0.3, s.lh ?? "open"], armR: [s.rx ?? 210, s.ry ?? -360, s.rb ?? -0.3, s.rh ?? "thumb"] });
  }
  function assistente(s, t) {
    return root.Chatbots.genieAssist({ eyes: { open: (s.open ?? 0.9) * (s.blink ?? 1), px: s.px ?? 0, py: s.py ?? 0.1, lid: s.lid ?? 0, lidColor: Chars.P.teal }, mouth: s.mouth ?? "smile",
      armL: [s.lx ?? -150, s.ly ?? -184, s.lb ?? 0.45, s.lh ?? "open"], armR: [s.rx ?? 192, s.ry ?? -344, s.rb ?? -0.25, s.rh ?? "thumb"] });
  }
  // posiziona un pupazzo: x,y (piedi), scala, rotazione, opacita', specchio
  // filtri condivisi (alone azzurro della "copia", bagliore caldo)
  const defs = `<defs><filter id="fxCopy" x="-30%" y="-30%" width="160%" height="160%"><feDropShadow dx="0" dy="0" stdDeviation="7" flood-color="#8fc2ff" flood-opacity=".95"/></filter>
    <filter id="fxBlur"><feGaussianBlur stdDeviation="7"/></filter></defs>`;
  const place = (svg, s) => `<g transform="translate(${s.x ?? 0},${s.y ?? 0}) rotate(${s.r ?? 0}) scale(${(s.s ?? 1) * (s.flip ? -1 : 1)},${s.s ?? 1})" opacity="${s.o ?? 1}">${svg}</g>`;
  const shadow = (s, w = 120, c = "#6b4a2a", op = 0.32) => {
    const lift = Math.max(0, (s.ground ?? s.y ?? 0) - (s.y ?? 0)); const k = 1 / (1 + lift / 120);
    return `<ellipse cx="${s.x ?? 0}" cy="${(s.ground ?? s.y ?? 0) + 4}" rx="${w * (s.s ?? 1) * k}" ry="${w * 0.15 * (s.s ?? 1) * k}" fill="${c}" opacity="${op * k * (s.o ?? 1)}"/>`;
  };
  // camera: zoom/pan attorno a un punto (x,y) del mondo
  // la camera non esce mai dal fondale (1920x1080): il centro e' vincolato ai bordi per lo zoom corrente
  const cam = (c) => { const z = Math.max(1, c.z ?? 1), r = c.r ?? 0;
    const x = c.free ? (c.x ?? 960) : clamp(c.x ?? 960, 960 / z, 1920 - 960 / z), y = c.free ? (c.y ?? 540) : clamp(c.y ?? 540, 540 / z, 1080 - 540 / z); return `translate(960,540) rotate(${r}) scale(${z}) translate(${-x},${-y})`; };
  // tremolio di camera smorzato (impatti)
  const shake = (t, t0, amp = 10, dur = 0.35) => { const d = t - t0; if (d < 0 || d > dur) return 0; return amp * (1 - d / dur) * Math.sin(d * 70); };

  // palcoscenico: sfondo statico (disegnato una volta), attori + camera ridisegnati a ogni fotogramma, grana sopra
  // bg puo' essere una stringa o un oggetto {nome: svg}: in quel caso frame(t) sceglie lo sfondo con r.bg
  function stage(svg, bg, fx, frame) {
    const bgs = typeof bg === "string" ? { main: bg } : bg;
    const sid = (svg.closest("[data-composition-id]")?.getAttribute("data-composition-id") ?? "x");
    svg.innerHTML = `${defs}<defs>${clipDefs(sid)}</defs><g class="cam">${Object.entries(bgs).map(([k, v]) => `<g class="bg" data-bg="${k}">${v}</g>`).join("")}<g class="actors"></g></g><g class="fx"></g><g class="over"></g>`;
    const camG = svg.querySelector(".cam"), act = svg.querySelector(".actors"), fxG = svg.querySelector(".fx"), over = svg.querySelector(".over");
    const layers = [...svg.querySelectorAll(".bg")]; const fxs = typeof fx === "string" ? { main: fx } : fx; let curFx = null;
    return (t) => {
      root.Anim._sid = sid;
      const r = frame(t); camG.setAttribute("transform", cam(r.cam ?? {})); act.innerHTML = r.actors ?? ""; over.innerHTML = r.over ?? "";
      const b = r.bg ?? Object.keys(bgs)[0]; layers.forEach((l) => l.setAttribute("display", l.dataset.bg === b ? "inline" : "none"));
      const f = r.fx ?? Object.keys(fxs)[0]; if (f !== curFx) { fxG.innerHTML = fxs[f] ?? ""; curFx = f; }
    };
  }
  // collega il palcoscenico alla timeline GSAP della scena (un orologio da 0 a dur, senza ease)
  function drive(tl, dur, update) { const c = { t: 0 }; tl.fromTo(c, { t: 0 }, { t: dur, duration: dur, ease: "none", onUpdate: () => update(c.t) }, 0); update(0); }

  // istante (assoluto, s) di una parola della battuta: tempi faster-whisper se presenti, altrimenti proporzione nei segmenti VAD
  function wordTime(line, word, edge = "start") {
    if (line.words && line.words.length) { const w = line.words.find((x) => x.w.toLowerCase().includes(word.toLowerCase())); if (w) return edge === "end" ? w.end : w.start; }
    const text = line.caption, i = text.toLowerCase().indexOf(word.toLowerCase());
    if (i < 0) return line.start;
    const pos = (i + (edge === "end" ? word.length : 0)) / text.length;
    const segs = line.pauses && line.pauses.length ? line.pauses : [[0, line.speech]];
    const total = segs.reduce((a, [x, y]) => a + y - x, 0); let target = pos * total, acc = 0;
    for (const [x, y] of segs) { if (acc + (y - x) >= target) return line.start + x + (target - acc); acc += y - x; }
    return line.end;
  }

  // ---------------------------------------------------------------- un chatbot esce dallo schermo del portatile
  // 1) anticipazione: lo schermo si gonfia e si illumina (raggi o particelle); 2) il personaggio sale da DENTRO il
  // display (visibile solo nello schermo e sopra la cornice: passa "dietro" il bordo); 3) salto ad arco e si appollaia
  // sopra il portatile, fluttuando. reverse: true -> lo stesso percorso al contrario (ci rientra).
  // o: { t0, T (durata), render(t) -> svg del personaggio, style: "burst"|"gentle", color, perch: {x,y,s}, reverse }
  const SCREEN = { x: 264, y: 488, w: 402, h: 228, top: 472 };
  function emerge(t, o) {
    const T = o.T ?? (o.style === "gentle" ? 1.5 : 1.0), sc = o.screen ?? SCREEN, cx = sc.x + sc.w / 2, cy = sc.y + sc.h / 2;
    let d = t - o.t0; if (o.reverse) d = T - d;      // al contrario: d parte da T (appollaiato) e scende a 0 (dentro lo schermo)
    if (d < 0) return "";
    if (d >= T) return perched(t, o);
    const fa = o.style === "gentle" ? 0.28 : 0.3, fr = o.style === "gentle" ? 0.45 : 0.33;
    const u = d / T, ps = o.perch.s, col = o.color ?? "#ffd23f";
    let out = "", under = "";
    // 1) anticipazione sullo schermo
    const glow = u < fa ? u / fa : Math.max(0, 1 - (u - fa) / (1 - fa) * 1.6);
    if (glow > 0) {
      const bul = 1 + 0.05 * Math.sin(Math.min(1, u / fa) * Math.PI) ;
      under += `<g transform="translate(${cx},${cy}) scale(${bul})"><rect x="${-sc.w / 2}" y="${-sc.h / 2}" width="${sc.w}" height="${sc.h}" rx="8" fill="${col}" opacity="${0.55 * glow}"/>
        <ellipse rx="${sc.w * 0.42}" ry="${sc.h * 0.36}" fill="#fff" opacity="${0.35 * glow}"/></g>`;
      if (o.style === "gentle") for (let i = 0; i < 14; i++) { const ph = ((d * 0.9 + i / 14) % 1), x = sc.x + 20 + ((i * 97) % (sc.w - 40)), y = sc.y + sc.h - ph * (sc.h + 160);
          out += `<rect x="${x}" y="${y}" width="12" height="12" rx="2" fill="${col}" opacity="${glow * Math.sin(ph * Math.PI)}" transform="rotate(${ph * 90} ${x + 6} ${y + 6})"/>`; }
      else for (let i = 0; i < 12; i++) { const a = (i / 12) * Math.PI * 2 + 0.2, r0 = 140, r1 = 140 + 220 * glow;
          out += `<line x1="${cx + Math.cos(a) * r0}" y1="${cy + Math.sin(a) * r0 * 0.7}" x2="${cx + Math.cos(a) * r1}" y2="${cy + Math.sin(a) * r1 * 0.7}" stroke="${col}" stroke-width="${10 * glow}" stroke-linecap="round" opacity="${0.8 * glow}"/>`; }
    }
    // 2) salita da dentro lo schermo -> 3) salto sul portatile
    let pose;
    if (u >= fa && u < fa + fr) {
      const k = (u - fa) / fr, e = ease(o.style === "gentle" ? "power2.out" : "power3.out")(k);
      pose = { x: cx, y: sc.y + sc.h + 120 * ps - e * (sc.h + 120 * ps + 40), s: ps * (0.55 + 0.45 * e), sq: o.style === "gentle" ? 1.0 : 1.18 - 0.1 * e, r: 0 };
    } else if (u >= fa + fr) {
      const k = (u - fa - fr) / (1 - fa - fr), e = ease("power1.inOut")(k), lift = Math.sin(k * Math.PI) * (o.style === "gentle" ? 30 : 90);
      pose = { x: lerp(cx, o.perch.x, e), y: lerp(sc.y - 40, o.perch.y, e) - lift, s: ps, r: (o.style === "gentle" ? -4 : 10) * Math.sin(k * Math.PI),
               sq: k > 0.82 ? 1 - 0.16 * Math.sin((k - 0.82) / 0.18 * Math.PI) : 1.05 };
    }
    if (pose) {
      const g = `<g transform="translate(${pose.x},${pose.y}) rotate(${pose.r}) scale(${pose.s / Math.sqrt(pose.sq)},${pose.s * pose.sq})">${o.render(t)}</g>`;
      const inside = pose.y > sc.top - 10;     // durante la salita: ritaglio dentro lo schermo + sopra la cornice
      const sid = root.Anim._sid ?? "x";
      out += inside ? `<g clip-path="url(#clipScreen-${sid})">${g}</g><g clip-path="url(#clipAbove-${sid})">${g}</g>` : g;
    }
    return under + out;
  }
  // il genio esce dallo schermo: anticipazione (schermo che si gonfia e si illumina), salita da dentro il display con
  // slancio (back.out) e poi oscillazione ancorata alla coda. La coda resta SEMPRE dentro il monitor.
  // o: { t0, T, render(t), style: "burst"|"gentle", color, perch: {x, y, s}, reverse, phase }
  function genieOut(t, o) {
    const T = o.T ?? (o.style === "gentle" ? 1.5 : 1.0), sc = SCREEN, cx = sc.x + sc.w / 2, cy = sc.y + sc.h / 2, col = o.color ?? "#ffd23f";
    let d = t - o.t0; if (o.reverse) d = T - d;
    if (d < 0) return "";
    const fa = o.style === "gentle" ? 0.32 : 0.3, u = Math.min(1, d / T), p = o.perch;
    let out = "";
    const glow = u < fa ? u / fa : Math.max(0, 1 - (u - fa) / (1 - fa) * 1.5);
    if (glow > 0 && u < 1) {
      out += `<g transform="translate(${cx},${cy}) scale(${1 + 0.05 * Math.sin(Math.min(1, u / fa) * Math.PI)})"><rect x="${-sc.w / 2}" y="${-sc.h / 2}" width="${sc.w}" height="${sc.h}" rx="8" fill="${col}" opacity="${0.55 * glow}"/><ellipse rx="${sc.w * 0.42}" ry="${sc.h * 0.36}" fill="#fff" opacity="${0.35 * glow}"/></g>`;
      if (o.style === "gentle") for (let i = 0; i < 14; i++) { const ph = ((d * 0.9 + i / 14) % 1), x = sc.x + 20 + ((i * 97) % (sc.w - 40)), y = sc.y + sc.h - ph * (sc.h + 200);
          out += `<rect x="${x}" y="${y}" width="12" height="12" rx="2" fill="${col}" opacity="${glow * Math.sin(ph * Math.PI)}" transform="rotate(${ph * 90} ${x + 6} ${y + 6})"/>`; }
      else for (let i = 0; i < 12; i++) { const a = (i / 12) * Math.PI * 2 + 0.2, r0 = 140, r1 = 140 + 220 * glow;
          out += `<line x1="${cx + Math.cos(a) * r0}" y1="${cy + Math.sin(a) * r0 * 0.7}" x2="${cx + Math.cos(a) * r1}" y2="${cy + Math.sin(a) * r1 * 0.7}" stroke="${col}" stroke-width="${10 * glow}" stroke-linecap="round" opacity="${0.8 * glow}"/>`; }
    }
    let y, s, sq, r;
    if (u < fa) return out;                                                     // solo anticipazione
    if (u < 1) {
      const k = (u - fa) / (1 - fa), e = ease(o.style === "gentle" ? "back.out(1.2)" : "back.out(1.8)")(k);
      y = lerp(sc.y + sc.h + 470 * p.s, p.y, e); s = p.s * (0.7 + 0.3 * Math.min(1, e)); sq = o.style === "gentle" ? 1 : 1.12 - 0.12 * Math.min(1, k * 1.4); r = (o.style === "gentle" ? 2 : 6) * Math.sin(k * Math.PI);
    } else { y = p.y + 5 * Math.sin(t * 2.2 + (o.phase ?? 0)); s = p.s; sq = 1 + 0.015 * Math.sin(t * 2.2 + 1); r = 3 * Math.sin(t * 1.6 + (o.phase ?? 0)); }
    const g = `<g transform="translate(${p.x},${y}) rotate(${r}) scale(${s / Math.sqrt(sq)},${s * sq})">${o.render(t)}</g>`;
    const sid = root.Anim._sid ?? "x";
    return out + `<g clip-path="url(#clipScreen-${sid})">${g}</g><g clip-path="url(#clipAbove-${sid})">${g}</g>`;
  }
  // appollaiato sopra il portatile, fluttuando dolcemente
  function perched(t, o) {
    const p = o.perch, bob = 6 * Math.sin(t * 2.4 + (o.phase ?? 0));
    return `<ellipse cx="${p.x}" cy="${SCREEN.top - 2}" rx="${110 * p.s}" ry="${12 * p.s}" fill="#000" opacity=".14"/>
      <g transform="translate(${p.x},${p.y + bob}) rotate(${1.5 * Math.sin(t * 1.7)}) scale(${p.s})">${o.render(t)}</g>`;
  }
  const clipDefs = (sid) => `<clipPath id="clipScreen-${sid}"><rect x="${SCREEN.x}" y="${SCREEN.y}" width="${SCREEN.w}" height="${SCREEN.h}" rx="6"/></clipPath>
    <clipPath id="clipAbove-${sid}"><rect x="-2000" y="-3000" width="6000" height="${3000 + SCREEN.top}"/></clipPath>`;

  root.Anim = { genieOut, emerge, perched, SCREEN, defs, stage, drive, wordTime, at, lerp, clamp, osc, ease, visemeTrack, mouthAt, blinkAt, autoBlinks, contratto, mouse, chatgratis, assistente, place, shadow, cam, shake };
})(typeof window !== "undefined" ? window : globalThis);
