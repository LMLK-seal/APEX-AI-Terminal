// STILE A — "Carta ritagliata 2.5D". Personaggi e set costruiti come cartoncino tagliato a strati.
// Ogni pezzo = bordo di taglio chiaro in alto a sinistra + spessore scuro in basso a destra + velatura di luce;
// ogni STRATO (g con filtro) proietta un'ombra morbida sullo strato sotto e ha grana di carta + contorno "tagliato a mano".
// Origine dei personaggi: ai piedi (0,0). Luce da in alto a sinistra. Funziona in browser e Node (stringhe SVG).
// Parti raggruppate con class/id stabili e data-depth (0 = lontano, 1 = vicino) per la camera 2.5D (vedi applyCamera).
// Regola: un gruppo con data-depth non porta MAI un transform proprio (lo scrive la camera).
(function (root) {
  "use strict";
  // ------------------------------------------------------------------ tavolozza (stessi significati dell'episodio 1)
  const C = {
    ink: "#2b2233", mouth: "#5a2432", tongue: "#e8767a", white: "#fffaf0",
    cream: "#f6eedc", creamLight: "#fffaf0", creamShade: "#e4d6b8",
    copper: "#c4622d", copperLight: "#e8935d", copperDark: "#8c3d1a",
    kraft: "#cfa574", kraftLight: "#e3c79b", kraftDark: "#a37a4a", wall: "#ead9ba",
    navy: "#1d2a46", navyMid: "#31466b", navyDark: "#111a2e", server: "#1d2a44",
    brass: "#e0ae50", led: "#ff5a45",
    teal: "#2a9487", tealLight: "#73c7b9", tealDark: "#1b6a61",
    gratis: "#ffd23f", gratisLight: "#ffe985", gratisDark: "#e3a800", pink: "#ec4d86", pinkDark: "#b42b60",
    mouse: "#93a4bd", mouseLight: "#c4d1e4", mouseDark: "#6a7b97",
  };
  let _n = 0; const uid = (p) => `${p}${++_n}`;
  const f = (x) => Math.round(x * 10) / 10;
  const h2 = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const mix = (a, b, t) => { const A = h2(a), B = h2(b); return "#" + A.map((v, i) => Math.round(v + (B[i] - v) * t).toString(16).padStart(2, "0")).join(""); };
  const dk = (c, t = 0.3) => mix(c, "#2a1208", t), lt = (c, t = 0.38) => mix(c, "#ffffff", t);

  // ------------------------------------------------------------------ filtri: strato di carta
  // wobble (taglio a mano) + grana + fibre + doppia ombra (contatto + ambiente). e = "quota" dello strato.
  function layerFilter(idn, dx, dy, sd, op, wob) {
    return `<filter id="${idn}" x="-30%" y="-30%" width="165%" height="175%" color-interpolation-filters="sRGB">
      <feTurbulence type="fractalNoise" baseFrequency=".015" numOctaves="2" seed="4" result="w"/>
      <feDisplacementMap in="SourceGraphic" in2="w" scale="${wob}" xChannelSelector="R" yChannelSelector="G" result="src"/>
      <feTurbulence type="fractalNoise" baseFrequency=".9" numOctaves="2" seed="5" result="n1"/>
      <feColorMatrix in="n1" type="matrix" values="0 0 0 0 .25  0 0 0 0 .15  0 0 0 0 .07  .6 0 0 0 -.25" result="g1"/>
      <feComposite in="g1" in2="src" operator="in" result="grain"/>
      <feTurbulence type="fractalNoise" baseFrequency=".05 .6" numOctaves="3" seed="11" result="n2"/>
      <feColorMatrix in="n2" type="matrix" values="0 0 0 0 1  0 0 0 0 .96  0 0 0 0 .88  1.2 0 0 0 -.68" result="g2"/>
      <feComposite in="g2" in2="src" operator="in" result="fibre"/>
      <feFlood flood-color="#3a1c0a" flood-opacity="${op}"/><feComposite in2="src" operator="in" result="fl"/>
      <feGaussianBlur in="fl" stdDeviation="${sd}" result="bl"/><feOffset in="bl" dx="${dx}" dy="${dy}" result="amb"/>
      <feFlood flood-color="#2a1208" flood-opacity="${f(op * 0.9)}"/><feComposite in2="src" operator="in" result="fl2"/>
      <feGaussianBlur in="fl2" stdDeviation=".7" result="bl2"/><feOffset in="bl2" dx="${f(dx * 0.35)}" dy="${f(dy * 0.4)}" result="con"/>
      <feMerge><feMergeNode in="amb"/><feMergeNode in="con"/><feMergeNode in="src"/><feMergeNode in="grain"/><feMergeNode in="fibre"/></feMerge></filter>`;
  }
  const DEFS = `<defs>
    <linearGradient id="sA-sheen" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#fff" stop-opacity=".28"/><stop offset=".5" stop-color="#fff" stop-opacity="0"/><stop offset="1" stop-color="#2a1208" stop-opacity=".17"/></linearGradient>
    <radialGradient id="sA-gok"><stop offset="0" stop-color="${C.tealLight}" stop-opacity=".9"/><stop offset="1" stop-color="${C.tealLight}" stop-opacity="0"/></radialGradient>
    <radialGradient id="sA-gno"><stop offset="0" stop-color="${C.led}" stop-opacity=".9"/><stop offset="1" stop-color="${C.led}" stop-opacity="0"/></radialGradient>
    <radialGradient id="sA-glow"><stop offset="0" stop-color="#fff3c4" stop-opacity=".85"/><stop offset="1" stop-color="#fff3c4" stop-opacity="0"/></radialGradient>
    ${layerFilter("sA-L0", 0, 0, 0.1, 0, 1.6)}${layerFilter("sA-L1", 2, 3.6, 2, 0.3, 1.9)}${layerFilter("sA-L2", 3.6, 7, 3.8, 0.34, 2.1)}${layerFilter("sA-L3", 6, 12, 6.5, 0.36, 2.4)}
    <filter id="sA-blur" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="7"/></filter>
    <filter id="sA-blur2" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="16"/></filter>
  </defs>`;

  // ------------------------------------------------------------------ primitive di carta
  const rr = (x, y, w, h, r) => { r = Math.min(r, w / 2, h / 2); return `M${f(x + r)},${f(y)} H${f(x + w - r)} Q${f(x + w)},${f(y)} ${f(x + w)},${f(y + r)} V${f(y + h - r)} Q${f(x + w)},${f(y + h)} ${f(x + w - r)},${f(y + h)} H${f(x + r)} Q${f(x)},${f(y + h)} ${f(x)},${f(y + h - r)} V${f(y + r)} Q${f(x)},${f(y)} ${f(x + r)},${f(y)}Z`; };
  const ell = (cx, cy, rx, ry) => `M${f(cx - rx)},${f(cy)} A${f(rx)},${f(ry)} 0 1,0 ${f(cx + rx)},${f(cy)} A${f(rx)},${f(ry)} 0 1,0 ${f(cx - rx)},${f(cy)}Z`;
  // pezzo di cartoncino: bordo luce (alto-sx), spessore scuro (basso-dx), colore, velatura
  function piece(d, fill, o = {}) {
    const t = o.t ?? 1;
    return `<g class="pc"><path d="${d}" fill="${o.dark ?? dk(fill)}" transform="translate(${f(1.1 * t)},${f(2.1 * t)})"/>` +
      `<path d="${d}" fill="${o.light ?? lt(fill)}" transform="translate(${f(-0.8 * t)},${f(-0.9 * t)})"/>` +
      `<path d="${d}" fill="${fill}"${o.op ? ` opacity="${o.op}"` : ""}/>` + (o.sheen === false ? "" : `<path d="${d}" fill="url(#sA-sheen)"/>`) + `</g>`;
  }
  // nastro di carta lungo un tracciato (braccia, cavo, tracolla)
  function strip(d, w, fill, o = {}) {
    return `<g class="st"><path d="${d}" fill="none" stroke="${o.dark ?? dk(fill, 0.38)}" stroke-width="${w}" stroke-linecap="${o.cap ?? "round"}" transform="translate(1.3,2.3)"/>` +
      `<path d="${d}" fill="none" stroke="${fill}" stroke-width="${w}" stroke-linecap="${o.cap ?? "round"}"/>` +
      `<path d="${d}" fill="none" stroke="${o.light ?? lt(fill, 0.34)}" stroke-width="${f(w * 0.26)}" stroke-linecap="round" opacity=".55" transform="translate(${f(-w * 0.2)},${f(-w * 0.2)})"/></g>`;
  }
  // linea "incisa" (bocca, pieghe): inchiostro con filo di luce sotto
  const slit = (d, w = 5, col = C.ink) => `<path d="${d}" fill="none" stroke="#fff" stroke-opacity=".5" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round" transform="translate(0,1.8)"/><path d="${d}" fill="none" stroke="${col}" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round"/>`;
  const crease = (d, w = 2.2) => `<path d="${d}" fill="none" stroke="#2a1208" stroke-opacity=".28" stroke-width="${w}" stroke-linecap="round"/><path d="${d}" fill="none" stroke="#fff" stroke-opacity=".45" stroke-width="${f(w * 0.8)}" stroke-linecap="round" transform="translate(.8,1.4)"/>`;
  const lay = (inner, e = 1, attrs = "") => `<g filter="url(#sA-L${e})"${attrs ? " " + attrs : ""}>${inner}</g>`;
  const dg = (depth, id, cls, inner, extra = "") => `<g${id ? ` id="${id}"` : ""} class="${cls}" data-part="${cls.split(" ")[0]}" data-depth="${depth}"${extra ? " " + extra : ""}>${inner}</g>`;
  const dgShadow = (rx, ry = 13, op = 0.32) => `<ellipse cx="0" cy="3" rx="${rx}" ry="${ry}" fill="#3a1c0a" opacity="${op}" filter="url(#sA-blur)"/>`;

  // Catmull-Rom -> Bezier
  function crPath(pts, closed) {
    const n = pts.length, g = (i) => closed ? pts[(i + n) % n] : pts[Math.max(0, Math.min(n - 1, i))];
    let d = `M${f(pts[0][0])},${f(pts[0][1])}`;
    for (let i = 0; i < (closed ? n : n - 1); i++) { const p0 = g(i - 1), p1 = g(i), p2 = g(i + 1), p3 = g(i + 2);
      d += ` C${f(p1[0] + (p2[0] - p0[0]) / 6)},${f(p1[1] + (p2[1] - p0[1]) / 6)} ${f(p2[0] - (p3[0] - p1[0]) / 6)},${f(p2[1] - (p3[1] - p1[1]) / 6)} ${f(p2[0])},${f(p2[1])}`; }
    return d + (closed ? "Z" : "");
  }
  function sampleCR(pts, per = 8) { // punti lungo la spline
    const n = pts.length, g = (i) => pts[Math.max(0, Math.min(n - 1, i))], out = [];
    for (let i = 0; i < n - 1; i++) { const p0 = g(i - 1), p1 = g(i), p2 = g(i + 1), p3 = g(i + 2);
      for (let k = 0; k < per; k++) { const t = k / per, t2 = t * t, t3 = t2 * t;
        out.push([0, 1].map((a) => 0.5 * ((2 * p1[a]) + (-p0[a] + p2[a]) * t + (2 * p0[a] - 5 * p1[a] + 4 * p2[a] - p3[a]) * t2 + (-p0[a] + 3 * p1[a] - 3 * p2[a] + p3[a]) * t3))); } }
    out.push(pts[n - 1]); return out;
  }
  // sagoma di fumo: linea centrale + profilo di larghezza w(t)
  function smoke(center, wfun, step = 5) {
    const s = sampleCR(center, 10), L = [], R = [];
    s.forEach((p, i) => { const a = s[Math.max(0, i - 1)], b = s[Math.min(s.length - 1, i + 1)], dx = b[0] - a[0], dy = b[1] - a[1], l = Math.hypot(dx, dy) || 1, nx = -dy / l, ny = dx / l, w = wfun(i / (s.length - 1)) / 2;
      L.push([p[0] + nx * w, p[1] + ny * w]); R.push([p[0] - nx * w, p[1] - ny * w]); });
    const pick = (a) => a.filter((_, i) => i % step === 0 || i === a.length - 1);
    return crPath([...pick(L), ...pick(R).reverse()], true);
  }

  // ------------------------------------------------------------------ occhi, sopracciglio, bocche
  function eyeA(cx, cy, o = {}) {
    const rx = o.rx ?? 28, open = o.open ?? 1, ry = (o.ry ?? 33) * open;
    if (open < 0.12) return `<g class="eye closed">${slit(`M${f(cx - rx)},${f(cy)} Q${f(cx)},${f(cy + (o.happy ? -20 : 9))} ${f(cx + rx)},${f(cy)}`, 6)}</g>`;
    const px = (o.px ?? 0) * rx * 0.4, py = (o.py ?? 0) * ry * 0.38, lid = o.lid ?? 0, cid = uid("ey");
    const ir = rx * 0.64, iry = Math.min(ry * 0.72, ir * 1.18), E = ell(0, 0, rx, ry);
    const lidC = o.lidColor ?? C.cream, lidY = -ry + 2 * ry * lid;
    return `<g class="eye" transform="translate(${f(cx)},${f(cy)}) scale(${o.sx ?? 1},1)"><clipPath id="${cid}"><path d="${E}"/></clipPath>
      ${piece(E, "#fffdf4", { t: 0.8, sheen: false })}
      <g clip-path="url(#${cid})">
        <ellipse cx="${f(px)}" cy="${f(py)}" rx="${f(ir)}" ry="${f(iry)}" fill="${C.navyMid}"/>
        <ellipse cx="${f(px)}" cy="${f(py - iry * 0.1)}" rx="${f(ir * 0.94)}" ry="${f(iry * 0.88)}" fill="${C.ink}"/>
        <ellipse cx="${f(px)}" cy="${f(py + iry * 0.62)}" rx="${f(ir * 0.62)}" ry="${f(iry * 0.2)}" fill="#6f84b3" opacity=".55"/>
        <circle cx="${f(px + ir * 0.36)}" cy="${f(py - iry * 0.4)}" r="${f(ir * 0.33)}" fill="#fff"/>
        <circle cx="${f(px - ir * 0.34)}" cy="${f(py + iry * 0.36)}" r="${f(ir * 0.14)}" fill="#fff" opacity=".85"/>
        <rect x="${-rx}" y="${-ry}" width="${2 * rx}" height="${f(ry * 0.34)}" fill="#2a1208" opacity=".1"/>
        ${lid > 0 ? `<rect x="${-rx - 2}" y="${-ry - 2}" width="${2 * rx + 4}" height="${f(2 * ry * lid + 2)}" fill="${lidC}"/><path d="M${-rx},${f(lidY)} L${rx},${f(lidY)}" stroke="${C.ink}" stroke-width="3.4"/><path d="M${-rx},${f(lidY + 3)} L${rx},${f(lidY + 3)}" stroke="#2a1208" stroke-opacity=".18" stroke-width="3"/>` : ""}
      </g><path d="${E}" fill="none" stroke="${C.ink}" stroke-opacity=".55" stroke-width="1.6"/></g>`;
  }
  // sopracciglio: mezzaluna di carta nera (corpo unico, espressivo)
  function browA(cx, y, tilt = 0, curve = 6, hw = 58, th = 12) {
    const x0 = cx - hw, x1 = cx + hw, y0 = y + tilt, y1 = y - tilt, cy = (y0 + y1) / 2 - curve - 4;
    const d = `M${f(x0)},${f(y0)} Q${f(cx)},${f(cy - th)} ${f(x1)},${f(y1)} Q${f(cx)},${f(cy + th * 0.55)} ${f(x0)},${f(y0)}Z`;
    return `<g class="brow"><path d="${d}" fill="${C.ink}" stroke="${C.ink}" stroke-width="4" stroke-linejoin="round"/><path d="M${f(x0 + 8)},${f(y0 - 1)} Q${f(cx)},${f(cy - th + 3)} ${f(x1 - 8)},${f(y1 - 1)}" fill="none" stroke="#fff" stroke-opacity=".22" stroke-width="2" stroke-linecap="round"/></g>`;
  }
  // bocche (visemi italiani + espressioni), centrate in (0,0), larghezza ~44. Cavita' scura + lingua rosa + denti di carta.
  function mouthA(kind, sc = 1) {
    const cav = (d, extra = "") => { const c = uid("mc"); return `<clipPath id="${c}"><path d="${d}"/></clipPath><path d="${d}" fill="#fff" fill-opacity=".55" transform="translate(0,2.2)"/><path d="${d}" fill="${C.mouth}"/><g clip-path="url(#${c})">${extra}<path d="${d}" fill="none" stroke="#1a0a10" stroke-opacity=".5" stroke-width="5" transform="translate(0,-2)"/></g>`; };
    const tongue = (y, w) => `<path d="M${-w},${y + 6} Q0,${y - 12} ${w},${y + 6} L${w + 8},${y + 22} L${-w - 8},${y + 22}Z" fill="${C.tongue}"/><path d="M0,${y - 2} L0,${y + 7}" stroke="#b9484f" stroke-width="2.4" stroke-linecap="round"/>`;
    const teeth = (x0, w, h = 7) => `<path d="${rr(x0, -13, w, h + 3, 2.5)}" fill="#fffdf4" stroke="#2a1208" stroke-opacity=".25" stroke-width="1.4"/>`;
    const m = {
      rest: slit("M-15,-2 Q0,8 15,-2", 5), smile: slit("M-23,-7 Q0,15 23,-7", 5.5),
      A: cav("M-22,-8 Q0,-11 22,-8 Q18,20 0,22 Q-18,20 -22,-8Z", tongue(8, 12) + teeth(-15, 30, 5)),
      E: cav("M-21,-5 Q0,-8 21,-5 Q16,9 0,10 Q-16,9 -21,-5Z", teeth(-14, 28, 3)),
      I: cav("M-19,-4 Q0,-6 19,-4 Q14,5 0,6 Q-14,5 -19,-4Z", `<path d="M-15,-5 L15,-5 L15,0 L-15,0Z" fill="#fffdf4"/>`),
      O: cav("M0,-14 Q13,-14 13,2 Q13,17 0,17 Q-13,17 -13,2 Q-13,-14 0,-14Z", tongue(8, 8)),
      U: cav("M0,-8 Q8,-8 8,1 Q8,10 0,10 Q-8,10 -8,1 Q-8,-8 0,-8Z"),
      MBP: slit("M-15,0 Q0,3 15,0", 5),
      FV: cav("M-17,-3 Q0,-5 17,-3 Q12,7 0,8 Q-12,7 -17,-3Z", `<path d="${rr(-11, -6, 22, 7, 2.5)}" fill="#fffdf4"/>`),
      frown: slit("M-18,6 Q0,-8 18,6", 5), flat: slit("M-14,1 L14,1", 5),
      smirk: slit("M-14,2 Q4,4 18,-7", 5), worried: slit("M-19,5 Q-9,-4 0,3 Q9,10 19,1", 5),
      grin: cav("M-30,-8 Q0,-12 30,-8 Q24,22 0,24 Q-24,22 -30,-8Z", `<path d="M-32,-10 L32,-10 L32,3 L-32,3Z" fill="#fffdf4"/><path d="M-30,3 L30,3 M-15,-10 L-15,3 M0,-10 L0,3 M15,-10 L15,3" stroke="#2a1208" stroke-opacity=".3" stroke-width="2"/>${tongue(14, 14)}`),
      shout: cav("M-18,-12 Q0,-16 18,-12 Q16,22 0,26 Q-16,22 -18,-12Z", teeth(-13, 26, 4) + tongue(15, 11)),
    };
    return `<g class="mouthshape" transform="scale(${sc})">${m[kind] ?? m.rest}</g>`;
  }

  // ------------------------------------------------------------------ mani, braccia, scarpe
  function hand(hx, hy, kind, dir, ang) {
    const E = (cx, cy, rx, ry) => ell(cx, cy, rx, ry), W = "#fbf3e2";
    if (kind === "point") return `<g transform="translate(${f(hx)},${f(hy)}) rotate(${f(ang)})">${piece(rr(-2, -8, 34, 14, 7), W)}${piece(E(0, 0, 15, 14), W)}</g>`;
    if (kind === "thumb") return `<g transform="translate(${f(hx)},${f(hy)})">${piece(rr(-6, -34, 13, 30, 6.5), W)}${piece(E(0, 0, 15, 14), W)}</g>`;
    return `<g transform="translate(${f(hx)},${f(hy)})">${piece(E(0, 0, 15, 14), W)}${piece(E(dir > 0 ? -9 : 9, -9, 7, 7.5), W, { t: 0.7 })}</g>`;
  }
  function arm(sx, sy, hx, hy, bend = 0.25, kind = "open", col = C.navy, w = 12) {
    const dx = hx - sx, dy = hy - sy, cx = (sx + hx) / 2 - dy * bend, cy = (sy + hy) / 2 + dx * bend;
    return strip(`M${f(sx)},${f(sy)} Q${f(cx)},${f(cy)} ${f(hx)},${f(hy)}`, w, col) + hand(hx, hy, kind, dx, Math.atan2(dy, dx) * 57.3);
  }
  function shoe(x, y, dir, col = C.navyMid) {
    const s = (v) => f(x + v * dir);
    const body = `M${s(-17)},${f(y - 13)} Q${s(-17)},${f(y - 23)} ${s(-3)},${f(y - 23)} Q${s(11)},${f(y - 25)} ${s(23)},${f(y - 13)} Q${s(33)},${f(y - 9)} ${s(33)},${f(y - 4)} L${s(33)},${f(y - 2)} L${s(-19)},${f(y - 2)} Q${s(-21)},${f(y - 2)} ${s(-21)},${f(y - 8)}Z`;
    return piece(body, col) + piece(`M${s(-21)},${f(y - 6)} L${s(33)},${f(y - 6)} Q${s(35)},${f(y)} ${s(30)},${f(y)} L${s(-19)},${f(y)} Q${s(-22)},${f(y)} ${s(-21)},${f(y - 6)}Z`, "#f4ecd9", { t: 0.7 }) +
      `<path d="M${s(15)},${f(y - 12)} Q${s(26)},${f(y - 11)} ${s(30)},${f(y - 6)} L${s(14)},${f(y - 6)}Z" fill="${lt(col, 0.2)}" opacity=".7"/>`;
  }
  const leg = (x0, y0, x1, y1, dir, col = C.navy) => strip(`M${x0},${y0} Q${f((x0 + x1) / 2 + 5 * dir)},${f((y0 + y1) / 2)} ${x1},${f(y1 - 8)}`, 15, col) + shoe(x1, y1, dir);
  const drop = (x, y, s = 1) => `<g transform="translate(${x},${y}) scale(${s})">${piece("M0,-14 Q11,2 8,10 Q5,17 0,17 Q-5,17 -8,10 Q-11,2 0,-14Z", "#9fd8ff")}<path d="M-3,2 Q-5,8 -2,11" stroke="#fff" stroke-width="2.4" fill="none" stroke-linecap="round" opacity=".8"/></g>`;

  // logo ufficiale intero in un riquadro size x size (mai deformato ne' ricolorato)
  function logo(kind, x, y, size) {
    const L = root.Loghi && root.Loghi[kind]; if (!L) return "";
    return `<svg class="logo logo-${kind}" x="${f(x - size / 2)}" y="${f(y - size / 2)}" width="${size}" height="${size}" viewBox="${L.vb}" preserveAspectRatio="xMidYMid meet" overflow="visible">${L.g}</svg>`;
  }

  // ================================================================== IL BADGE
  // tessera plastificata, fascia aziendale avvolta, clip d'ottone, chip, codice a barre, LED. Piedi chunky.
  function badge(o = {}) {
    const id = o.id ?? "badge", W = 98, T = -368, B = -64, R = 30, sq = o.squash ?? 1, br = o.brow ?? {}, ey = o.eyes ?? {};
    const aL = o.armL ?? [-156, -168, 0.3, "open"], aR = o.armR ?? [156, -168, -0.3, "open"];
    const strap = o.strap ?? 0, card = rr(-W, T, 2 * W, B - T, R), cid = uid("bd");
    const bandY0 = T + 46, bandY1 = T + 102;
    const lanyard = strap > 0 ? `<g class="lanyard">${strip(`M-5,${T - 38} L-30,${T - 38 - strap}`, 15, C.navyMid, { cap: "butt" })}${strip(`M5,${T - 38} L30,${T - 38 - strap}`, 15, C.navy, { cap: "butt" })}</g>` : "";
    const bars = [4, 2, 6, 3, 2, 5, 2, 4, 3, 6, 2, 3].reduce((a, w, i) => { a.s += i % 2 ? "" : `<rect x="${a.x}" y="${B - 62}" width="${w}" height="30"/>`; a.x += w + 3; return a; }, { x: -20, s: "" }).s;
    const ledC = o.led === "ok" ? "gok" : "gno", ledF = o.led === "ok" ? C.tealLight : C.led;
    const led = o.led ? `<circle cx="66" cy="${B - 47}" r="26" fill="url(#sA-${ledC})"/>${piece(ell(66, B - 47, 10, 10), ledF, { t: 0.8 })}<circle cx="63" cy="${B - 50}" r="3" fill="#fff" opacity=".85"/>` : "";
    const chip = piece(rr(-78, B - 66, 40, 36, 7), C.brass) + `<path d="M-78,${B - 54} H-38 M-78,${B - 42} H-38 M-62,${B - 66} V${B - 30} M-52,${B - 66} V${B - 30}" stroke="#7a4f12" stroke-opacity=".55" stroke-width="2.2" fill="none"/>${piece(rr(-65, B - 54, 14, 12, 3), lt(C.brass, 0.25), { t: 0.5, sheen: false })}`;
    const bodyShade = `<path d="${rr(W - 34, T + 20, 40, B - T - 40, 18)}" fill="#8f6d3f" opacity=".11"/>`;
    // 1) ombra a terra, gambe/piedi, braccia dietro
    const legs = dg(0.46, id + "-legs", "legs", lay(leg(-34, B - 2, -42, 0, -1) + leg(34, B - 2, 42, 0, 1), 1));
    const armsBack = (side, a, flag) => dg(flag ? 0.62 : 0.46, id + "-arm" + side, "arm" + side, lay(arm(side === "L" ? -W + 6 : W - 6, -204, a[0], a[1], a[2], a[3]), 1));
    // 2) corpo: tessera plastificata
    const body = dg(0.5, id + "-body", "body", lay(
      `<path d="${rr(-W - 6, T - 6, 2 * W + 12, B - T + 12, R + 5)}" fill="#f2fbfc" fill-opacity=".5" stroke="#fff" stroke-opacity=".85" stroke-width="1.6"/>` +
      piece(card, C.cream, { dark: "#cdb98f", light: "#fffdf6", t: 1.2 }) +
      `<clipPath id="${cid}"><path d="${card}"/></clipPath><g clip-path="url(#${cid})">${bodyShade}
        <path d="M24,${T} L62,${T} L-48,${B} L-86,${B}Z" fill="#fff" opacity=".2"/><path d="M72,${T} L84,${T} L-26,${B} L-38,${B}Z" fill="#fff" opacity=".14"/></g>` +
      `<path d="${rr(-W + 8, T + 8, 2 * W - 16, B - T - 16, R - 7)}" fill="none" stroke="#b79d6c" stroke-opacity=".35" stroke-width="1.6" stroke-dasharray="6 5"/>` +
      // asola + fascia avvolta
      `<path d="M${-W - 7},${bandY1} L${-W},${bandY1} L${-W},${bandY1 + 9}Z M${W + 7},${bandY1} L${W},${bandY1} L${W},${bandY1 + 9}Z" fill="${C.copperDark}"/>` +
      piece(rr(-W - 7, bandY0, 2 * W + 14, bandY1 - bandY0, 5), C.copper, { dark: C.copperDark, light: C.copperLight }) +
      `<path d="M${-W + 6},${bandY0 + 6} H${W - 6} M${-W + 6},${bandY1 - 6} H${W - 6}" stroke="${C.creamLight}" stroke-opacity=".5" stroke-width="1.6" stroke-dasharray="5 4" fill="none"/>` +
      `<text x="0" y="${bandY0 + 35}" text-anchor="middle" font-family="Inter" font-weight="800" font-size="21" letter-spacing="2.6" fill="#5c230d" opacity=".5" transform="translate(0,1.5)">NODOMATIC</text><text x="0" y="${bandY0 + 35}" text-anchor="middle" font-family="Inter" font-weight="800" font-size="21" letter-spacing="2.6" fill="#fff6e6">NODOMATIC</text>` +
      `<path d="${rr(-24, T + 14, 48, 14, 7)}" fill="#fff" fill-opacity=".55" transform="translate(0,1.8)"/><path d="${rr(-24, T + 14, 48, 14, 7)}" fill="${C.ink}"/>` +
      chip + `<g fill="${C.ink}">${bars}</g>` + led +
      (o.name ? `<text x="0" y="${B - 76}" text-anchor="middle" font-family="IBM Plex Mono" font-weight="700" font-size="16" fill="${C.ink}">${o.name}</text>` : "") +
      crease(`M${-W + 12},${bandY1 + 14} Q0,${bandY1 + 20} ${W - 12},${bandY1 + 14}`, 1.6), 2));
    // 3) clip d'ottone (con anello) davanti allo spacco
    const clip = dg(0.52, id + "-clip", "clip", lay(piece(rr(-8, T - 2, 16, 30, 5), dk(C.brass, 0.15)) + piece(rr(-21, T - 34, 42, 30, 9), C.brass) + `<path d="M-13,${T - 27} H13" stroke="#fff" stroke-opacity=".55" stroke-width="3" stroke-linecap="round"/>` + piece(rr(-6, T - 40, 12, 9, 4), dk(C.brass, 0.28), { t: 0.6, sheen: false }), 2));
    // 4) volto: guance, sopracciglio, occhi, bocca (strato rialzato)
    const face = dg(0.56, id + "-face", "face",
      `<g class="cheeks"><circle cx="-66" cy="-186" r="14" fill="${C.copperLight}" opacity=".42"/><circle cx="66" cy="-186" r="14" fill="${C.copperLight}" opacity=".42"/></g>` +
      lay(`<g id="${id}-brow" class="brow-g">${browA(0, br.y ?? -268, br.tilt ?? 0, br.curve ?? 6)}</g><g id="${id}-eyes" class="eyes">${eyeA(-36, -226, { rx: 26, ry: 32, lidColor: C.cream, ...ey })}${eyeA(36, -226, { rx: 26, ry: 32, lidColor: C.cream, ...ey })}</g>` +
        `<g id="${id}-mouth" class="mouth" transform="translate(0,-158)">${mouthA(o.mouth ?? "rest", 1.1)}</g>`, 1));
    const sweat = o.sweat ? dg(0.6, id + "-sweat", "sweat", lay(drop(W + 10, T + 40, 1.2), 1)) : "";
    return `<g class="badge" data-char="badge" id="${id}">${o.noDefs ? "" : DEFS}<g class="sq" transform="scale(${f(1 / Math.sqrt(sq))},${sq})">
      ${dg(0.4, null, "shadow", dgShadow(104))}${lanyard}${legs}${o.lf ? "" : armsBack("L", aL)}${o.rf ? "" : armsBack("R", aR)}${body}${clip}${face}
      ${o.lf ? armsBack("L", aL, 1) : ""}${o.rf ? armsBack("R", aR, 1) : ""}${sweat}</g></g>`;
  }

  // ================================================================== IL MOUSE
  function mouse(o = {}) {
    const id = o.id ?? "mouse", sq = o.squash ?? 1, ey = o.eyes ?? { open: 1.06, px: 0.35, py: -0.1 }, br = o.brow ?? { y: -160, curve: 4, tilt: -6 }, cid = uid("ms");
    const body = "M0,-238 C66,-238 88,-184 88,-122 C88,-60 56,-32 0,-32 C-56,-32 -88,-60 -88,-122 C-88,-184 -66,-238 0,-238Z";
    const aL = o.armL ?? [-136, -150, 0.35, "open"], aR = o.armR ?? [124, -96, -0.2, "open"];
    const cup = o.cup !== false ? `<g class="cup" transform="translate(${aR[0] + 4},${aR[1] - 26}) rotate(${o.cupTilt ?? 8})">
        <path d="M-6,-44 Q-16,-58 -6,-70 Q4,-82 -4,-94" fill="none" stroke="#fff" stroke-opacity=".7" stroke-width="5" stroke-linecap="round"/><path d="M10,-44 Q2,-56 12,-68 Q20,-78 12,-90" fill="none" stroke="#fff" stroke-opacity=".5" stroke-width="4" stroke-linecap="round"/>
        ${piece("M-17,-28 L17,-28 L12,20 L-12,20Z", "#fbf3e2")}${piece("M-16,-12 L16,-12 L14,8 L-14,8Z", C.kraft)}${piece(rr(-19, -36, 38, 10, 4), C.copper)}</g>` : "";
    const sweat = o.sweat !== false ? dg(0.6, id + "-sweat", "sweat", lay(drop(100, -214, 1.1) + drop(-104, -172, 0.9), 1)) : "";
    const stride = o.stride, lx = stride == null ? 0 : Math.sin(stride) * 22, l1 = stride == null ? 0 : Math.max(0, Math.sin(stride)) * 16, l2 = stride == null ? 0 : Math.max(0, -Math.sin(stride)) * 16;
    const cable = `M0,-232 C8,-266 -34,-292 -58,-272 C-80,-254 -64,-222 -92,-210 C-110,-204 -122,-220 -124,-234`;
    const legs = dg(0.46, id + "-legs", "legs", lay(leg(-30, -42, -38 - lx, -l1, -1) + leg(30, -42, 40 + lx, -l2, 1), 1));
    const ears = `${piece(ell(-60, -214, 25, 26), C.mouseDark)}${piece(ell(-60, -214, 14, 15), "#f3a9b7", { t: 0.5 })}${piece(ell(60, -214, 25, 26), C.mouseDark)}${piece(ell(60, -214, 14, 15), "#f3a9b7", { t: 0.5 })}`;
    const bodyL = dg(0.5, id + "-body", "body", lay(
      `<g class="cable">${strip(cable, 8, C.mouseDark, { dark: C.navyDark })}${piece(rr(-142, -250, 26, 20, 4), C.mouseLight, { t: 0.7 })}${piece(rr(-136, -246, 14, 5, 1.5), C.brass, { t: 0.3, sheen: false })}</g>` +
      ears + piece(body, C.mouse, { dark: C.mouseDark, light: C.mouseLight, t: 1.3 }) +
      `<clipPath id="${cid}"><path d="${body}"/></clipPath><g clip-path="url(#${cid})">
        ${piece(rr(-96, -250, 94, 82, 20), C.mouseLight, { t: 0.9 })}${piece(rr(2, -250, 94, 82, 20), C.mouseLight, { t: 0.9 })}
        <path d="M60,-238 Q100,-170 62,-60 L100,-60 L100,-240Z" fill="#2a1208" opacity=".1"/></g>` +
      piece(rr(-7, -226, 14, 28, 7), C.mouseDark, { t: 0.7 }) + `<path d="M-7,-218 H7 M-7,-211 H7 M-7,-204 H7" stroke="#fff" stroke-opacity=".35" stroke-width="1.6"/>` +
      piece(ell(0, -104, 66, 58), "#d6e0ee", { dark: C.mouseDark, light: "#fff", t: 0.9 }), 2));
    const face = dg(0.56, id + "-face", "face",
      `<g class="cheeks"><circle cx="-52" cy="-82" r="11" fill="#f08aa0" opacity=".4"/><circle cx="52" cy="-82" r="11" fill="#f08aa0" opacity=".4"/></g>` +
      lay(`<g id="${id}-brow" class="brow-g">${browA(0, br.y ?? -160, br.tilt ?? -6, br.curve ?? 4, 48, 11)}</g><g id="${id}-eyes" class="eyes">${eyeA(-27, -122, { rx: 22, ry: 28, lidColor: "#d6e0ee", ...ey })}${eyeA(27, -122, { rx: 22, ry: 28, lidColor: "#d6e0ee", ...ey })}</g>` +
        `<g id="${id}-mouth" class="mouth" transform="translate(0,-70)">${mouthA(o.mouth ?? "worried", 0.95)}</g>`, 1));
    const armL = dg(0.62, id + "-armL", "armL", lay(arm(-80, -138, aL[0], aL[1], aL[2], aL[3]), 1));
    const armR = dg(0.62, id + "-armR", "armR", lay(arm(80, -138, aR[0], aR[1], aR[2], aR[3]) + cup, 1));
    return `<g class="mouse" data-char="mouse" id="${id}">${o.noDefs ? "" : DEFS}<g class="sq" transform="scale(${f(1 / Math.sqrt(sq))},${sq})">
      ${dg(0.4, null, "shadow", dgShadow(100))}${legs}${bodyL}${face}${armL}${armR}${sweat}</g></g>`;
  }

  // ================================================================== I GENI (con logo ufficiale sul medaglione)
  const starburst = (r, n = 12) => { const p = []; for (let i = 0; i < n * 2; i++) { const a = (i / (n * 2)) * Math.PI * 2, rad = i % 2 ? r * 0.78 : r; p.push(`${f(Math.cos(a) * rad)},${f(Math.sin(a) * rad)}`); } return "M" + p.join(" L") + "Z"; };
  function genie(kind, o = {}) {
    const ok = o.approved ?? false, id = o.id ?? "genie-" + kind, sq = o.squash ?? 1, ey = o.eyes ?? {};
    const col = ok ? C.teal : C.gratis, cL = ok ? C.tealLight : C.gratisLight, cD = ok ? C.tealDark : C.gratisDark;
    const aL = o.armL ?? (ok ? [-150, -206, 0.4, "open"] : [-168, -318, 0.25, "open"]), aR = o.armR ?? (ok ? [160, -332, -0.25, "thumb"] : [164, -216, -0.3, "thumb"]);
    const cid = uid("gn");
    // coda di fumo: strati di carta sovrapposti (curva piu' composta se approvato)
    const ctr = ok ? [[0, -204], [12, -160], [-10, -118], [-22, -78], [-4, -40], [22, -20], [40, -34]] : [[0, -204], [26, -164], [-18, -126], [-34, -84], [0, -48], [38, -52], [44, -22], [20, -6]];
    const w1 = (t) => 14 + 100 * Math.pow(1 - t, 1.25), w2 = (t) => 6 + 70 * Math.pow(1 - t, 1.3), w3 = (t) => 3 + 38 * Math.pow(1 - t, 1.4);
    const tail = dg(0.46, id + "-legs", "legs tail", lay(
      piece(smoke(ctr, w1), cD, { dark: dk(cD), light: col }) + piece(smoke(ctr.map(([x, y]) => [x - 2, y - 2]), w2), col, { t: 1.1 }) + piece(smoke(ctr.map(([x, y]) => [x - 4, y - 4]), w3), cL, { t: 0.9 }) +
      (ok ? "" : `${piece(ell(-52, -40, 13, 13), col, { t: 0.8 })}${piece(ell(58, -86, 9, 9), cL, { t: 0.7 })}`), 1));
    const torsoD = "M-86,-292 C-106,-250 -76,-212 -54,-186 L54,-186 C76,-212 106,-250 86,-292 C52,-312 -52,-312 -86,-292Z";
    const torso = dg(0.5, id + "-body", "body", lay(
      piece(torsoD, col, { dark: cD, light: cL, t: 1.2 }) +
      `<clipPath id="${cid}"><path d="${torsoD}"/></clipPath><g clip-path="url(#${cid})"><path d="M52,-312 Q104,-250 70,-180 L120,-180 L120,-312Z" fill="${cD}" opacity=".35"/></g>` +
      // cintura
      (ok ? piece(rr(-62, -212, 124, 24, 5), C.tealDark, { t: 0.9 }) + `<text x="-10" y="-194.5" text-anchor="middle" font-family="Inter" font-weight="800" font-size="12.5" letter-spacing="1.6" fill="${C.cream}">IN SEDE</text>` +
        `<g transform="translate(40,-200)"><path d="M-8,-5 L-8,-13 Q-8,-19 0,-19 Q8,-19 8,-13 L8,-5" fill="none" stroke="#b78a2e" stroke-width="3.6"/>${piece(rr(-13, -6, 26, 21, 5), C.brass, { t: 0.8 })}<circle cx="0" cy="3" r="3" fill="#6b4a0d"/></g>`
        : piece(rr(-62, -212, 124, 22, 4), C.pink, { dark: C.pinkDark, t: 0.9 }) + `<path d="M-52,-201 H52" stroke="#fff" stroke-opacity=".55" stroke-width="2" stroke-dasharray="3 5"/>`) +
      // medaglione: anello + disco bianco (il logo sta fuori dal filtro, intero)
      piece(ell(0, -246, 53, 53), ok ? C.brass : C.pink, { t: 1 }) + piece(ell(0, -246, 46, 46), "#fffefa", { t: 0.8, sheen: false }), 2));
    const medalLogo = `<g class="medal-logo" data-depth="0.54">${logo(kind, 0, -246, 62)}</g>`;
    // testa
    const headD = "M0,-428 C46,-428 82,-398 82,-355 C82,-312 46,-280 0,-280 C-46,-280 -82,-312 -82,-355 C-82,-398 -46,-428 0,-428Z";
    const topknot = ok
      ? piece("M-14,-424 C-26,-452 -2,-470 12,-452 C20,-442 8,-432 0,-432 C12,-440 6,-452 -4,-448 C-10,-444 -8,-434 -2,-428Z", col, { t: 0.8 }) + piece(rr(-20, -432, 40, 11, 5), C.brass, { t: 0.8 })
      : piece("M-40,-418 L-52,-462 L-22,-432 L-6,-472 L8,-432 L34,-460 L38,-416Z", C.pink, { t: 0.8 });
    const bow = ok ? `<g class="bowtie">${piece("M0,-284 L-34,-304 Q-42,-284 -34,-264Z", C.copper, { t: 0.9 })}${piece("M0,-284 L34,-304 Q42,-284 34,-264Z", C.copper, { t: 0.9 })}${piece(rr(-9, -296, 18, 24, 6), C.copperLight, { t: 0.8 })}${crease("M-32,-296 L-12,-286 M32,-296 L12,-286", 1.6)}</g>` : "";
    const head = dg(0.54, id + "-head", "head", lay(topknot + piece(headD, col, { dark: cD, light: cL, t: 1.2 }) + `<path d="M-52,-398 Q-30,-418 4,-418" fill="none" stroke="#fff" stroke-opacity=".28" stroke-width="7" stroke-linecap="round"/>` + bow, 2));
    const brows = ok
      ? `${browA(-34, -394, 5, 8, 24, 10)}${browA(34, -394, -5, 8, 24, 10)}`
      : `${browA(-34, -396, 8, 10, 25, 10)}${browA(34, -406, -8, 14, 25, 10)}`;
    const eyeO = ok ? { rx: 21, ry: 26, open: 0.95, lidColor: col, ...ey } : { rx: 21, ry: 26, lid: 0.38, px: 0.8, lidColor: col, ...ey };
    const face = dg(0.58, id + "-face", "face",
      `<g class="cheeks"><circle cx="-54" cy="-336" r="13" fill="${ok ? "#f4a58d" : C.pink}" opacity="${ok ? 0.38 : 0.5}"/><circle cx="54" cy="-336" r="13" fill="${ok ? "#f4a58d" : C.pink}" opacity="${ok ? 0.38 : 0.5}"/></g>` +
      lay(`<g id="${id}-brow" class="brow-g">${brows}</g><g id="${id}-eyes" class="eyes">${eyeA(-33, -362, eyeO)}${eyeA(33, -362, eyeO)}</g><g id="${id}-mouth" class="mouth" transform="translate(0,-318)">${mouthA(o.mouth ?? (ok ? "smile" : "grin"), 1.0)}</g>`, 1));
    const armsL = dg(0.46, id + "-armL", "armL", lay(arm(-80, -274, aL[0], aL[1], aL[2], aL[3], col, 20), 1));
    const armsR = dg(0.46, id + "-armR", "armR", lay(arm(80, -274, aR[0], aR[1], aR[2], aR[3], col, 20), 1));
    const deco = ok ? "" : dg(0.62, id + "-deco", "deco", lay(
      `<g transform="translate(96,-398) rotate(14)">${piece(starburst(40), C.pink, { dark: C.pinkDark, light: "#ff86b0" })}<text y="5" text-anchor="middle" font-family="Inter" font-weight="800" font-size="14" fill="#fff" letter-spacing=".5">GRATIS</text></g>` +
      [[-120, -430, 12, 18], [-140, -330, 9, -10], [130, -300, 11, 24], [-110, -258, 8, 30], [118, -452, 8, 8]].map(([x, y, s, r], i) => `<g transform="translate(${x},${y}) rotate(${r})">${piece(rr(-s, -s, 2 * s, 2 * s, 2), i % 2 ? C.pink : "#ffffff", { t: 0.7 })}</g>`).join(""), 2));
    return `<g class="genie genie-${kind} ${ok ? "approved" : "unauthorized"}" data-char="genie" data-kind="${kind}" id="${id}">${o.noDefs ? "" : DEFS}<g class="sq" transform="scale(${f(1 / Math.sqrt(sq))},${sq})">
      ${dg(0.4, null, "shadow", dgShadow(88, 12, 0.28))}${tail}${armsL}${armsR}${torso}${medalLogo}${head}${face}${deco}</g></g>`;
  }

  // ================================================================== L'UFFICIO a strati di profondita'
  // Coordinate: 1920x1080, origine in alto a sinistra. Piano della scrivania (dove stanno i personaggi): y 790..1010.
  // Livelli (data-depth): 0.06 parete · 0.2 finestra/orologio/scaffale · 0.36 porta+lettore · 0.52 scrivania+laptop · [0.7 personaggi] · 0.88 bordo scrivania · 1.0 primo piano.
  function office(o = {}) {
    const led = o.led ?? "ok", scr = o.screen ?? "Chiedi qualcosa...", title = o.title ?? "Assistente";
    const ledF = led === "ok" ? C.tealLight : led === "no" ? C.led : "#8b9bb4";
    const P = piece;
    // ---- parete lontana: carta da parati a strisce + zoccolo a pannelli
    let stripes = ""; for (let i = -2; i < 18; i++) stripes += `<rect x="${i * 130}" y="-120" width="65" height="940" fill="${i % 2 ? "#e6d3b1" : "#eddcbf"}"/>`;
    const dado = Array.from({ length: 12 }, (_, i) => P(rr(i * 175 - 120, 630, 160, 160, 6), "#c9a273", { t: 0.8 })).join("");
    const far = `<g class="layer layer-far" data-layer="far" data-depth="0.06">${lay(
      `<rect x="-220" y="-140" width="2360" height="940" fill="${C.wall}"/>${stripes}<rect x="-220" y="610" width="2360" height="30" fill="#b98f5e"/>` + P(rr(-220, 636, 2360, 14, 3), "#d9b78a") + dado +
      `<path d="M-60,0 L560,0 L1180,820 L560,820Z" fill="url(#sA-glow)" opacity=".5"/>`, 0)}</g>`;
    // ---- finestra con tapparella, orologio, scaffale
    const hills = `<path d="M0,300 Q120,210 230,262 T420,236 L420,420 L0,420Z" fill="#9dc1ab"/><path d="M0,350 Q140,290 250,332 T420,312 L420,420 L0,420Z" fill="#7da98f"/><ellipse cx="300" cy="120" rx="62" ry="22" fill="#fff" opacity=".85"/><ellipse cx="340" cy="106" rx="38" ry="18" fill="#fff" opacity=".85"/><ellipse cx="90" cy="160" rx="46" ry="16" fill="#fff" opacity=".6"/>`;
    const slats = Array.from({ length: 9 }, (_, i) => P(rr(-4, i * 22, 428, 20, 4), i % 2 ? "#e7d3ad" : "#efdfbf", { t: 0.7 })).join("");
    const win = `<g transform="translate(120,90)">${P(rr(-26, -26, 472, 470, 10), "#f6efe0", { t: 1.2 })}<rect x="0" y="0" width="420" height="420" fill="#bcd9e2"/>${hills}<path d="M0,0 L420,0 L420,34 L0,34Z" fill="#000" opacity=".06"/>${slats}${P(rr(-4, 196, 428, 14, 5), C.kraftDark, { t: 0.8 })}${P(rr(-36, 436, 492, 26, 6), "#f6efe0", { t: 1 })}</g>`;
    const tick = Array.from({ length: 12 }, (_, i) => { const a = i * Math.PI / 6; return `<path d="M${f(Math.sin(a) * 74)},${f(-Math.cos(a) * 74)} L${f(Math.sin(a) * 84)},${f(-Math.cos(a) * 84)}" stroke="${C.ink}" stroke-width="${i % 3 ? 3 : 5.5}" stroke-linecap="round"/>`; }).join("");
    const clock = `<g transform="translate(1030,170)">${P(ell(0, 0, 108, 108), C.kraftDark, { t: 1.2 })}${P(ell(0, 0, 98, 98), C.brass, { t: 1 })}${P(ell(0, 0, 88, 88), "#fbf4e4", { t: 0.8, sheen: false })}${tick}<path d="M0,0 L-34,-22" stroke="${C.ink}" stroke-width="8" stroke-linecap="round"/><path d="M0,0 L56,6" stroke="${C.ink}" stroke-width="5.5" stroke-linecap="round"/><circle r="8" fill="${C.copper}"/></g>`;
    const books = [[C.copper, 70, 108], [C.kraftDark, 54, 92], [C.navyMid, 62, 118], [C.tealDark, 50, 100], [C.copperLight, 66, 86], ["#9aa7bd", 56, 112]];
    let bx = 1390; const shelf = `<g>${books.map(([c, w, h]) => { const s = P(rr(bx, 442 - h, w, h, 5), c, { t: 0.9 }) + `<path d="M${bx + 10},${442 - h + 18} H${bx + w - 10}" stroke="#fff" stroke-opacity=".55" stroke-width="4" stroke-linecap="round"/>`; bx += w + 8; return s; }).join("")}${P(rr(1360, 442, 520, 20, 4), C.kraftDark, { t: 1.2 })}${P(rr(1380, 462, 480, 8, 2), dk(C.kraftDark, 0.2), { t: 0.6, sheen: false })}</g>`;
    const mid1 = `<g class="layer layer-wall" data-layer="wall" data-depth="0.2">${lay(win + clock + shelf, 1)}</g>`;
    // ---- porta con lettore badge
    const door = `<g transform="translate(1390,200)">${P(rr(-30, -30, 420, 640, 8), C.kraftDark, { t: 1.4 })}<rect x="0" y="0" width="360" height="600" fill="#3a2a1a"/>${P(rr(0, 0, 360, 610, 3), C.kraft, { t: 1.2 })}${P(rr(40, 50, 280, 220, 8), C.kraftLight, { t: 0.9, sheen: false })}${P(rr(40, 310, 280, 240, 8), C.kraftLight, { t: 0.9, sheen: false })}${P(ell(318, 330, 18, 18), C.brass, { t: 1 })}</g>
      <g transform="translate(1830,360)">${P(rr(-34, -62, 68, 124, 14), C.navyMid, { t: 1.1 })}${P(rr(-22, -42, 44, 44, 8), C.server, { t: 0.7, sheen: false })}<path d="M-12,-26 q12,-9 24,0 M-7,-17 q7,-5 14,0" fill="none" stroke="${C.mouseLight}" stroke-width="3" stroke-linecap="round"/><circle cy="32" r="22" fill="url(#sA-g${led === "no" ? "no" : "ok"})"/>${P(ell(0, 32, 10, 10), ledF, { t: 0.8 })}</g>`;
    const plantBack = `<g transform="translate(120,780)">${[[-60, -200, -26], [-18, -250, -8], [26, -230, 14], [58, -180, 30], [-30, -150, -38]].map(([x, y, r]) => `<g transform="rotate(${r} 0 -80)">${P(`M0,0 Q${x - 26},${y / 2} ${x},${y} Q${x + 28},${y / 2} 0,0Z`, r % 2 ? C.tealDark : "#3f9a72", { t: 0.9 })}${crease(`M0,-6 Q${x * 0.4},${y / 2} ${x * 0.9},${y * 0.9}`, 1.4)}</g>`).join("")}${P("M-52,-84 L52,-84 L40,0 L-40,0Z", C.copper, { t: 1.1 })}${P(rr(-58, -98, 116, 20, 5), C.copperLight, { t: 1 })}</g>`;
    const mid2 = `<g class="layer layer-door" data-layer="door" data-depth="0.36">${lay(door, 2)}</g>`;
    // ---- scrivania: piano, laptop, tazza, portapenne
    const grain = Array.from({ length: 7 }, (_, i) => `<path d="M-200,${812 + i * 28} Q500,${806 + i * 28 + (i % 2 ? 8 : -6)} 1100,${814 + i * 28} T2160,${810 + i * 28}" fill="none" stroke="${i % 2 ? "#fff" : "#7a4f22"}" stroke-opacity=".13" stroke-width="${3 + (i % 3)}"/>`).join("");
    const deskTop = P("M-220,790 L2140,790 L2140,1010 L-220,1010Z", C.kraft, { t: 1.4, dark: C.kraftDark, light: C.kraftLight }) + grain + `<path d="M-220,790 L2140,790 L2140,800 L-220,800Z" fill="#2a1208" opacity=".14"/><path d="M80,800 L660,800 L420,1010 L-120,1010Z" fill="url(#sA-glow)" opacity=".35"/>`;
    const lap = `<g transform="translate(250,0)">${P("M30,790 L520,790 L560,832 L-10,832Z", C.mouseDark, { t: 1.2 })}${P(rr(40, 796, 470, 12, 3), C.mouse, { t: .6 })}${P(rr(20, 480, 510, 318, 16), C.navyMid, { t: 1.3 })}${P(rr(38, 498, 474, 280, 8), "#e9f2f0", { t: 0.8, sheen: false })}
      ${P(rr(38, 498, 474, 44, 8), C.teal, { t: 0.6 })}<text x="60" y="527" font-family="Inter" font-weight="800" font-size="22" fill="#fff" letter-spacing=".5">${title}</text>${P(ell(472, 520, 10, 10), "#fff", { t: 0.4, sheen: false })}
      ${P(rr(60, 568, 250, 44, 14), "#fff", { t: 0.6, sheen: false })}${P(rr(240, 628, 252, 44, 14), C.tealLight, { t: 0.6, sheen: false })}<path d="M80,590 H260 M260,650 H468" stroke="#8aa09b" stroke-width="7" stroke-linecap="round" opacity=".6"/>
      ${P(rr(60, 712, 392, 40, 20), "#fff", { t: 0.6, sheen: false })}<text x="84" y="739" font-family="Inter" font-weight="600" font-size="19" fill="#7f8f8b">${scr}</text>${P(ell(430, 732, 14, 14), C.teal, { t: 0.5 })}</g>`;
    const mug = `<g transform="translate(1180,930)">${P("M-44,-90 L44,-90 L36,0 L-36,0Z", C.copper, { t: 1.2 })}<path d="M44,-70 Q82,-70 74,-36 Q70,-14 40,-18" fill="none" stroke="${C.copper}" stroke-width="12" stroke-linecap="round"/><ellipse cx="0" cy="-90" rx="44" ry="10" fill="#4a2410"/></g>`;
    const pens = `<g transform="translate(1700,930)">${["#2d4b6e", C.copper, C.teal].map((c, i) => `<g transform="rotate(${-14 + i * 12} 0 -60)">${P(rr(-4 + i * 4, -170 + i * 10, 8, 120, 3), c, { t: 0.6 })}</g>`).join("")}${P("M-40,-72 L40,-72 L34,0 L-34,0Z", C.tealDark, { t: 1.1 })}</g>`;
    const desk = `<g class="layer layer-desk" data-layer="desk" data-depth="0.52">${lay(deskTop, 1)}${lay(plantBack + lap + mug + pens, 2)}</g>`;
    // ---- personaggi (slot) e bordo scrivania
    const slot = o.mid ? `<g class="layer layer-chars" data-layer="chars" data-depth="0.7">${o.mid}</g>` : "";
    const front = `<g class="layer layer-front" data-layer="front" data-depth="0.88">${lay(P("M-220,1004 L2140,1004 L2140,1180 L-220,1180Z", C.kraftDark, { t: 1.6, dark: dk(C.kraftDark, 0.4), light: C.kraft }) + `<path d="M-220,1004 L2140,1004 L2140,1020 L-220,1020Z" fill="${C.kraftLight}" opacity=".7"/>` + P(rr(220, 1052, 300, 22, 8), dk(C.kraftDark, 0.2), { t: 0.8 }), 3)}</g>`;
    const leaf = (x, y, r, s, c) => `<g transform="translate(${x},${y}) rotate(${r}) scale(${s})">${P("M0,0 C-60,-60 -70,-170 0,-260 C70,-170 60,-60 0,0Z", c, { t: 1.2 })}${crease("M0,-6 L0,-236 M0,-70 L-34,-110 M0,-110 L36,-150 M0,-150 L-30,-186", 2)}</g>`;
    const lampCord = `<g transform="translate(1780,0)"><path d="M0,-10 V150" stroke="${C.ink}" stroke-width="4"/>${P("M-70,260 Q-50,150 0,150 Q50,150 70,260Z", C.copper, { t: 1.4, dark: C.copperDark, light: C.copperLight })}<ellipse cx="0" cy="262" rx="60" ry="8" fill="url(#sA-glow)"/></g>`;
    const fg = `<g class="layer layer-fg" data-layer="fg" data-depth="1">${lay(leaf(70, 1110, -18, 1.25, "#3f9a72") + leaf(170, 1130, 14, 1.05, C.tealDark) + leaf(-10, 1140, -48, 0.9, "#4aa57d") + lampCord, 3)}</g>`;
    return `<g class="office" data-char="office">${o.noDefs ? "" : DEFS}${far}${mid1}${mid2}${desk}${slot}${front}${fg}</g>`;
  }

  // ------------------------------------------------------------------ camera 2.5D (browser): sposta i gruppi per data-depth
  // cam = spostamento orizzontale della camera in px, k = intensita'. I gruppi annidati ricevono solo la differenza col genitore.
  function applyCamera(el, cam, k = 1, camY = 0) {
    const eff = (d) => ({ x: -(d - 0.5) * cam * k, y: -(d - 0.5) * camY * k });
    el.querySelectorAll("[data-depth]").forEach((g) => {
      const p = g.parentElement && g.parentElement.closest("[data-depth]"), e = eff(+g.dataset.depth), pe = p ? eff(+p.dataset.depth) : { x: 0, y: 0 };
      g.setAttribute("transform", `translate(${f(e.x - pe.x)},${f(e.y - pe.y)})`);
    });
  }
  const svg = (inner, w, h, vb) => `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="${vb ?? `0 0 ${w} ${h}`}">${inner}</svg>`;

  root.StileA = { C, DEFS, badge, mouse, genie, office, applyCamera, svg, piece, strip, eyeA, browA, mouthA, rr, ell, lay, logo, version: "A-carta-1" };
})(typeof window !== "undefined" ? window : globalThis);
