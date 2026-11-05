// Stile B "Morbido premium": vettoriale 3D-ish. Volumi con gradienti a 2-3 toni, luce di bordo, occlusione ambientale,
// contorni colorati a spessore variabile, occhi lucidi a due riflessi. Origine ai piedi (y negativo = verso l'alto).
// Browser + Node: root.StileB = { badge, mouse, genie, office, mouth, palette }.
(function (root) {
  "use strict";
  let N = 0; const uid = (p) => `${p}${++N}`;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const r2 = (v) => Math.round(v * 100) / 100;

  // ------------------------------------------------------------------ colore
  const h2r = (h) => { h = h.replace("#", ""); if (h.length === 3) h = [...h].map((c) => c + c).join(""); return [0, 2, 4].map((i) => parseInt(h.substr(i, 2), 16)); };
  const r2h = (a) => "#" + a.map((v) => clamp(Math.round(v), 0, 255).toString(16).padStart(2, "0")).join("");
  const mix = (a, b, t) => { const A = h2r(a), B = h2r(b); return r2h(A.map((v, i) => v + (B[i] - v) * t)); };
  const lt = (c, t) => mix(c, "#ffffff", t);
  const dk = (c, t) => mix(c, r2h(h2r(c).map((v, i) => v * 0.26 + [10, 3, 26][i])), t);   // ombra: scura e leggermente viola
  const ol = (c) => dk(c, 0.62);                                                           // contorno = tinta scura del riempimento

  // Tavolozza (stesso significato dell'episodio 1, tonalita' raffinate)
  const PAL = {
    cream: "#fff5e3", creamS: "#ead6b4", kraft: "#d9ad78", copper: "#c4602c", copperL: "#ec9660", brass: "#e8b14e",
    navy: "#1b2844", navyMid: "#2d4268", led: "#ff4a3a", ledOk: "#4be3b6", ink: "#2b1d2e",
    gratis: "#ffd84a", gratisD: "#f4b400", pink: "#f0508a", teal: "#2fa79a", tealL: "#72cdbf", tealD: "#1d7468",
    mouse: "#9db2d6", glove: "#fff3e0", rose: "#ff93a6",
  };

  // ------------------------------------------------------------------ contesto per istanza: gradienti con id unici
  function ctx() {
    const id = uid("sb"); let n = 0; const defs = [], cache = {};
    const once = (key, fn) => (cache[key] ??= fn(`${id}_${n++}`));
    const K = {
      id, defs, uid: (p) => `${id}${p}${n++}`,
      fill: (c) => `url(#${once("f" + c, (i) => { defs.push(`<radialGradient id="${i}" cx=".33" cy=".25" r="1"><stop offset="0" stop-color="${lt(c, 0.45)}"/><stop offset=".38" stop-color="${c}"/><stop offset="1" stop-color="${dk(c, 0.42)}"/></radialGradient>`); return i; })})`,
      rim: (c) => `url(#${once("r" + c, (i) => { defs.push(`<linearGradient id="${i}" x1="0" y1="0" x2="1" y2="1"><stop offset=".45" stop-color="#fff" stop-opacity="0"/><stop offset="1" stop-color="${mix(lt(c, 0.62), "#fff1d8", 0.4)}" stop-opacity=".95"/></linearGradient>`); return i; })})`,
      edge: (c) => `url(#${once("e" + c, (i) => { defs.push(`<linearGradient id="${i}" x1="0" y1="0" x2="1" y2="1"><stop offset=".2" stop-color="${ol(c)}" stop-opacity="0"/><stop offset="1" stop-color="${dk(c, 0.9)}"/></linearGradient>`); return i; })})`,
      lin: (c1, c2) => `url(#${once("l" + c1 + c2, (i) => { defs.push(`<linearGradient id="${i}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${c1}"/><stop offset="1" stop-color="${c2}"/></linearGradient>`); return i; })})`,
      fade: (c, o0, o1) => `url(#${once("d" + c + o0 + o1, (i) => { defs.push(`<linearGradient id="${i}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${c}" stop-opacity="${o0}"/><stop offset="1" stop-color="${c}" stop-opacity="${o1}"/></linearGradient>`); return i; })})`,
      sheen: () => `url(#${once("sh", (i) => { defs.push(`<linearGradient id="${i}" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset=".5" stop-color="#fff" stop-opacity=".55"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>`); return i; })})`,
      soft: (c) => `url(#${once("o" + c, (i) => { defs.push(`<radialGradient id="${i}"><stop offset="0" stop-color="${c}" stop-opacity="1"/><stop offset=".5" stop-color="${c}" stop-opacity=".55"/><stop offset="1" stop-color="${c}" stop-opacity="0"/></radialGradient>`); return i; })})`,
      iris: (a, b) => `url(#${once("i" + a + b, (i) => { defs.push(`<radialGradient id="${i}" cx=".5" cy=".42" r=".62"><stop offset="0" stop-color="${a}"/><stop offset=".55" stop-color="${mix(a, b, 0.4)}"/><stop offset="1" stop-color="${b}"/></radialGradient>`); return i; })})`,
      sclera: () => `url(#${once("sc", (i) => { defs.push(`<radialGradient id="${i}" cx=".45" cy=".4" r=".75"><stop offset=".5" stop-color="#ffffff"/><stop offset="1" stop-color="#cfd8ea"/></radialGradient>`); return i; })})`,
      // morbida macchia di colore (ombre di contatto, occlusione ambientale, bagliori)
      blob: (cx, cy, rx, ry, c, op = 0.3, rot = 0) => `<ellipse cx="${r2(cx)}" cy="${r2(cy)}" rx="${rx}" ry="${ry}" fill="${K.soft(c)}" opacity="${op}"${rot ? ` transform="rotate(${rot} ${cx} ${cy})"` : ""}/>`,
      // volume: ombra-contorno spessa lato ombra + riempimento sfumato + luce di bordo + contorno colorato
      shape(d, c, o = {}) {
        const cp = K.uid("c"), sw = o.sw ?? 4.2, rw = o.rim ?? 11;
        return `<clipPath id="${cp}"><path d="${d}"/></clipPath>` +
          `<path d="${d}" fill="none" stroke="${K.edge(c)}" stroke-width="${sw + 4.5}" stroke-linejoin="round" stroke-linecap="round"/>` +
          `<path d="${d}" fill="${o.flat ? c : K.fill(c)}"/>` +
          `<g clip-path="url(#${cp})">${o.inner || ""}${rw ? `<path d="${d}" fill="none" stroke="${K.rim(c)}" stroke-width="${rw}" stroke-linejoin="round"/>` : ""}</g>` +
          `<path d="${d}" fill="none" stroke="${ol(c)}" stroke-width="${sw}" stroke-linejoin="round" stroke-linecap="round"/>`;
      },
    };
    return K;
  }
  const wrapSvg = (K, id, inner) => `<g class="sb-char" data-id="${id}"><defs>${K.defs.join("")}</defs>${inner}</g>`;

  // ------------------------------------------------------------------ geometria
  const circ = (x, y, r) => `M${x - r},${y} a${r},${r} 0 1,0 ${2 * r},0 a${r},${r} 0 1,0 ${-2 * r},0Z`;
  const pill = (x0, y0, x1, y1, r) => { const k = r * 0.5523; return `M${x0 + r},${y0} L${x1 - r},${y0} C${x1 - r + k},${y0} ${x1},${y0 + r - k} ${x1},${y0 + r} L${x1},${y1 - r} C${x1},${y1 - r + k} ${x1 - r + k},${y1} ${x1 - r},${y1} L${x0 + r},${y1} C${x0 + r - k},${y1} ${x0},${y1 - r + k} ${x0},${y1 - r} L${x0},${y0 + r} C${x0},${y0 + r - k} ${x0 + r - k},${y0} ${x0 + r},${y0}Z`; };
  // spline chiusa Catmull-Rom attraverso i punti
  const spline = (P) => { const n = P.length; let d = `M${r2(P[0][0])},${r2(P[0][1])}`; for (let i = 0; i < n; i++) { const p0 = P[(i - 1 + n) % n], p1 = P[i], p2 = P[(i + 1) % n], p3 = P[(i + 2) % n]; d += ` C${r2(p1[0] + (p2[0] - p0[0]) / 6)},${r2(p1[1] + (p2[1] - p0[1]) / 6)} ${r2(p2[0] - (p3[0] - p1[0]) / 6)},${r2(p2[1] - (p3[1] - p1[1]) / 6)} ${r2(p2[0])},${r2(p2[1])}`; } return d + "Z"; };
  // sagoma "nastro": linea centrale [x,y,r] dalla coda alla testa + calotta ellittica sulla testa
  function ribbon(C, cap) {
    const A = [], B = [];
    C.forEach((p, i) => { const a = C[Math.max(0, i - 1)], b = C[Math.min(C.length - 1, i + 1)]; let tx = b[0] - a[0], ty = b[1] - a[1]; const l = Math.hypot(tx, ty) || 1; tx /= l; ty /= l; const nx = -ty, ny = tx; A.push([p[0] + nx * p[2], p[1] + ny * p[2]]); B.push([p[0] - nx * p[2], p[1] - ny * p[2]]); });
    // tangente verso l'alto: la normale (-ty,tx) punta a destra sullo schermo, quindi A e' il lato destro
    const top = C[C.length - 1], arc = [];
    for (let k = 1; k < 8; k++) { const th = (k / 8) * Math.PI; arc.push([top[0] + cap.rx * Math.cos(th), top[1] - cap.ry * Math.sin(th)]); }
    return spline([...A, ...arc, ...B.reverse()]);
  }

  // ------------------------------------------------------------------ parte per parte: parallasse a profondita'
  const par = (o, d) => { const c = o.cam; if (!c) return ""; const k = o.camK ?? 14; return ` transform="translate(${r2(-c[0] * d * k)},${r2(-(c[1] || 0) * d * k)})"`; };
  const part = (o, cls, d, inner, extra = "") => `<g class="${cls}" data-depth="${d}"${extra}${par(o, d)}>${inner}</g>`;

  // ------------------------------------------------------------------ occhi grandi e lucidi (due riflessi)
  function eye(K, x, y, o = {}) {
    const rx = o.rx ?? 26, ry = o.ry ?? 32, open = clamp(o.open ?? 1, 0, 1.4), lid = clamp(o.lid ?? 0, 0, 1);
    const px = o.px ?? 0, py = o.py ?? 0, ir = o.iris ?? ["#2b1d33", "#6b5186"], lidCol = o.lidCol ?? "#fff3e0", inkC = o.ink ?? "#2b1d2e";
    if (o.happy) return `<g class="eye"><path d="M${x - rx},${y + ry * 0.3} Q${x},${y - ry * 1.0} ${x + rx},${y + ry * 0.3}" fill="none" stroke="${inkC}" stroke-width="7.5" stroke-linecap="round"/></g>`;
    if (open < 0.14) return `<g class="eye"><path d="M${x - rx},${y - 1} Q${x},${y + ry * 0.42} ${x + rx},${y - 1}" fill="none" stroke="${inkC}" stroke-width="6.5" stroke-linecap="round"/></g>`;
    const h = ry * Math.min(open, 1.2), cp = K.uid("e");
    const R = Math.min(rx * 0.8, h * 0.88), ex = x + px * rx * 0.3, ey = y + py * h * 0.3;
    const lidY = y - h + lid * 2 * h;
    const lidSh = lid > 0 ? `<path d="M${x - rx - 3},${y - h - 3} L${x + rx + 3},${y - h - 3} L${x + rx + 3},${lidY - 2} Q${x},${lidY + 7 + lid * 6} ${x - rx - 3},${lidY - 2}Z" fill="${lidCol}"/><path d="M${x - rx - 3},${lidY - 2} Q${x},${lidY + 7 + lid * 6} ${x + rx + 3},${lidY - 2}" fill="none" stroke="${inkC}" stroke-width="4.5"/>` : "";
    return `<g class="eye"><clipPath id="${cp}"><ellipse cx="${x}" cy="${y}" rx="${rx}" ry="${h}"/></clipPath>
      <ellipse cx="${x}" cy="${y}" rx="${rx}" ry="${h}" fill="${K.sclera()}"/>
      <g clip-path="url(#${cp})">
        <circle cx="${r2(ex)}" cy="${r2(ey)}" r="${r2(R)}" fill="${K.iris(ir[0], ir[1])}"/>
        <circle cx="${r2(ex)}" cy="${r2(ey)}" r="${r2(R)}" fill="none" stroke="${dk(ir[0], 0.5)}" stroke-width="2.4"/>
        <circle cx="${r2(ex)}" cy="${r2(ey)}" r="${r2(R * 0.5)}" fill="#0d0813"/>
        <path d="M${r2(ex - R * 0.62)},${r2(ey + R * 0.35)} Q${r2(ex)},${r2(ey + R * 1.02)} ${r2(ex + R * 0.62)},${r2(ey + R * 0.35)}" fill="none" stroke="${lt(ir[1], 0.55)}" stroke-width="${r2(R * 0.16)}" stroke-linecap="round" opacity=".55"/>
        <ellipse cx="${r2(ex - R * 0.36)}" cy="${r2(ey - R * 0.4)}" rx="${r2(R * 0.36)}" ry="${r2(R * 0.31)}" fill="#fff" transform="rotate(-22 ${r2(ex - R * 0.36)} ${r2(ey - R * 0.4)})"/>
        <circle cx="${r2(ex + R * 0.42)}" cy="${r2(ey + R * 0.36)}" r="${r2(R * 0.15)}" fill="#fff" opacity=".92"/>
        <rect x="${x - rx}" y="${y - h}" width="${rx * 2}" height="${r2(h * 0.7)}" fill="${K.fade("#1a0f24", 0.34, 0)}"/>
        ${lidSh}
      </g>
      <ellipse cx="${x}" cy="${y}" rx="${rx}" ry="${h}" fill="none" stroke="${inkC}" stroke-width="3" opacity=".55"/>
      <path d="M${x - rx + 1},${y - h * 0.32} Q${x - rx * 0.6},${y - h * 1.02} ${x + rx * 0.2},${y - h * 1.0} Q${x + rx * 0.86},${y - h * 0.92} ${x + rx - 1},${y - h * 0.3}" fill="none" stroke="${inkC}" stroke-width="5.2" stroke-linecap="round"/></g>`;
  }

  // sopracciglio affusolato: da (x0,y0) a (x1,y1), curve>0 = arco verso l'alto, th = spessore al centro
  function browShape(x0, y0, x1, y1, curve, th, col) {
    const mx = (x0 + x1) / 2, my = (y0 + y1) / 2, dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy) || 1, ux = dy / L, uy = -dx / L;   // (ux,uy) = "su"
    const cT = [mx + ux * (curve + th), my + uy * (curve + th)], cB = [mx + ux * (curve - th), my + uy * (curve - th)];
    return `<path d="M${r2(x0)},${r2(y0)} Q${r2(cT[0])},${r2(cT[1])} ${r2(x1)},${r2(y1)} Q${r2(cB[0])},${r2(cB[1])} ${r2(x0)},${r2(y0)}Z" fill="${col}" stroke="${col}" stroke-width="3.2" stroke-linejoin="round"/>`;
  }

  // ------------------------------------------------------------------ bocche (visemi)
  function mouth(K, kind = "rest", s = 1, lip = "#80303f") {
    const inner = "#4a1826", tongue = "#f47c8f", teeth = "#fffaf0";
    const st = (d, w = 4.6, op = 1) => `<path d="${d}" fill="none" stroke="${lip}" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round" opacity="${op}"/>`;
    const cav = (d, top, bot, hw, t = 0, tg = true, grid = false) => {
      const cp = K.uid("m");
      return `<clipPath id="${cp}"><path d="${d}"/></clipPath><path d="${d}" fill="${K.lin("#5a2030", inner)}"/><g clip-path="url(#${cp})">
        ${tg ? `<ellipse cx="0" cy="${bot + 1}" rx="${r2(hw * 0.6)}" ry="${r2((bot - top) * 0.42)}" fill="${tongue}"/><ellipse cx="0" cy="${r2(bot - (bot - top) * 0.28)}" rx="${r2(hw * 0.34)}" ry="${r2((bot - top) * 0.12)}" fill="#fff" opacity=".28"/>` : ""}
        ${t ? `<rect x="${-hw - 2}" y="${top - 2}" width="${hw * 2 + 4}" height="${t + 2}" fill="${teeth}"/><rect x="${-hw - 2}" y="${top + t - 2}" width="${hw * 2 + 4}" height="3" fill="#d9cbb8" opacity=".8"/>${grid ? `<path d="M0,${top} L0,${top + t}" stroke="#d9cbb8" stroke-width="1.6"/>` : ""}` : ""}
      </g><path d="${d}" fill="none" stroke="${lip}" stroke-width="3.8" stroke-linejoin="round"/>`;
    };
    const M = {
      rest: () => st("M-18,0 Q0,11 18,0") + st("M-20,-1 q-3,-1 -4,-5 M20,-1 q3,-1 4,-5", 2.6, 0.55),
      smile: () => cav("M-28,-4 Q0,6 28,-4 Q25,27 0,28 Q-25,27 -28,-4Z", -4, 28, 28, 9),
      grin: () => cav("M-33,-6 Q0,5 33,-6 Q33,29 0,31 Q-33,29 -33,-6Z", -6, 31, 33, 14, false, true),
      A: () => cav("M-19,-6 Q0,-12 19,-6 Q25,19 0,29 Q-25,19 -19,-6Z", -10, 29, 22, 0),
      E: () => cav("M-27,-5 Q0,-11 27,-5 Q25,14 0,15 Q-25,14 -27,-5Z", -8, 15, 26, 7, false),
      I: () => cav("M-26,-2 Q0,-7 26,-2 Q23,9 0,10 Q-23,9 -26,-2Z", -6, 10, 25, 6, false),
      O: () => cav("M0,-13 C15,-13 16,0 15,8 C14,23 -14,23 -15,8 C-16,0 -15,-13 0,-13Z", -13, 21, 15, 0),
      U: () => cav("M0,-5 C10,-5 11,4 10,10 C9,20 -9,20 -10,10 C-11,4 -10,-5 0,-5Z", -5, 17, 10, 0),
      MBP: () => st("M-18,2 Q0,5 18,2", 5.4) + st("M-9,8 Q0,12 9,8", 3, 0.6),
      FV: () => cav("M-25,-2 Q0,-8 25,-2 Q22,9 0,10 Q-22,9 -25,-2Z", -6, 10, 24, 8, false) + `<path d="M-15,10 Q0,19 15,10 Q0,13 -15,10Z" fill="#e48a98" stroke="${lip}" stroke-width="3" stroke-linejoin="round"/>`,
      frown: () => st("M-19,9 Q0,-4 19,9"),
      flat: () => st("M-18,4 L18,4", 5),
      smirk: () => st("M-17,8 Q3,9 21,-5") + st("M21,-5 q4,-1 6,-6", 3.2),
      worried: () => st("M-21,8 Q-10,-3 0,5 T21,4", 4.4),
      shout: () => cav("M-35,-10 Q0,-23 35,-10 Q46,35 0,45 Q-46,35 -35,-10Z", -14, 45, 40, 12),
    };
    return `<g class="mouth-shape" transform="scale(${s})">${(M[kind] || M.rest)()}</g>`;
  }

  // ------------------------------------------------------------------ faccia completa (occhi, sopracciglia, bocca, guance)
  function face(K, cfg, o) {
    const e = o.eyes ?? {}, b = o.brow ?? {};
    const E = (x, over = {}) => eye(K, x, cfg.ey, { rx: cfg.rx, ry: cfg.ry, lidCol: cfg.lidCol, iris: cfg.iris, ink: cfg.ink, ...e, ...over });
    const by = b.y ?? cfg.ey - cfg.ry - 15, cv = b.curve ?? 7, tl = b.tilt ?? 0, wy = b.worry ?? 0, rr = b.raiseR ?? 0, th = cfg.bt ?? 5.2;
    const brL = browShape(-cfg.ex - cfg.rx * 0.95, by + tl + wy * 7, -cfg.ex + cfg.rx * 0.9, by + tl - wy * 11, cv, th, cfg.browCol);
    const brR = browShape(cfg.ex - cfg.rx * 0.9, by - tl - wy * 11 - rr, cfg.ex + cfg.rx * 0.95, by - tl + wy * 7 - rr, cv + (b.curveR ?? 0), th, cfg.browCol);
    const ch = cfg.cheek === false ? "" : K.blob(-cfg.ex - cfg.rx * 0.7, cfg.cy, 20, 12, PAL.rose, 0.5) + K.blob(cfg.ex + cfg.rx * 0.7, cfg.cy, 20, 12, PAL.rose, 0.5);
    return {
      cheeks: `<g class="cheeks">${ch}</g>`,
      eyes: `<g class="eyes">${E(-cfg.ex, e.L)}${E(cfg.ex, e.R)}</g>`,
      brows: `<g class="brows">${brL}${brR}</g>`,
      mouth: `<g class="mouth" transform="translate(0,${cfg.my})">${mouth(K, o.mouth ?? "rest", cfg.ms, cfg.lip)}</g>`,
    };
  }

  // ------------------------------------------------------------------ arti: braccio a tubo + guanto a manopola
  function limb(K, d, w, c) {
    return `<path d="${d}" fill="none" stroke="${ol(c)}" stroke-width="${w + 5}" stroke-linecap="round"/><path d="${d}" fill="none" stroke="${c}" stroke-width="${w}" stroke-linecap="round"/>` +
      `<path d="${d}" fill="none" stroke="${dk(c, 0.34)}" stroke-width="${r2(w * 0.3)}" stroke-linecap="round" opacity=".42" transform="translate(${r2(w * 0.22)},${r2(w * 0.22)})"/>` +
      `<path d="${d}" fill="none" stroke="${lt(c, 0.55)}" stroke-width="${r2(w * 0.26)}" stroke-linecap="round" opacity=".55" transform="translate(${r2(-w * 0.2)},${r2(-w * 0.22)})"/>`;
  }
  function hand(K, x, y, ang, kind = "open", sc = 1) {
    const c = PAL.glove; let P = `<circle cx="7" cy="0" r="15"/>`;
    if (kind === "point") P += `<ellipse cx="31" cy="-2" rx="16" ry="6.2"/><circle cx="9" cy="10" r="9"/><ellipse cx="3" cy="-12" rx="8" ry="6" transform="rotate(-30 3 -12)"/>`;
    else if (kind === "thumb") P += `<ellipse cx="7" cy="-23" rx="6.8" ry="13" transform="rotate(${r2(-ang)} 7 0)"/><circle cx="14" cy="8" r="9"/>`;
    else P += [-38, 0, 38].map((a) => { const fx = r2(7 + 20 * Math.cos((a * Math.PI) / 180)), fy = r2(20 * Math.sin((a * Math.PI) / 180)); return `<ellipse cx="${fx}" cy="${fy}" rx="9.5" ry="5.8" transform="rotate(${a} ${fx} ${fy})"/>`; }).join("") + `<ellipse cx="3" cy="-15" rx="9" ry="6" transform="rotate(-48 3 -15)"/>`;
    return `<g class="hand" transform="translate(${r2(x)},${r2(y)}) rotate(${r2(ang)}) scale(${sc})">
      <g fill="${ol(c)}" stroke="${ol(c)}" stroke-width="7.5" stroke-linejoin="round">${P}</g><g fill="${c}">${P}</g>
      <circle cx="7" cy="0" r="15" fill="${K.fill(c)}"/><path d="M-3,-9 Q4,-14 12,-11" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" opacity=".8"/></g>`;
  }
  // braccio da (sx,sy) alla mano (hx,hy), curvatura bend (positivo: gomito verso l'alto-fuori a sinistra)
  function arm(K, sx, sy, hx, hy, bend, kind, c, w = 13) {
    const dx = hx - sx, dy = hy - sy, L = Math.hypot(dx, dy) || 1, cx = (sx + hx) / 2 + (-dy / L) * bend * L * 0.5, cy = (sy + hy) / 2 + (dx / L) * bend * L * 0.5;
    const ang = (Math.atan2(hy - cy, hx - cx) * 180) / Math.PI;
    return limb(K, `M${sx},${sy} Q${r2(cx)},${r2(cy)} ${hx},${hy}`, w, c) + hand(K, hx, hy, ang, kind || "open");
  }
  function leg(K, x, y0, dir, c, shoe, stride = 0, lift = 0) {
    const fx = x + dir * 6 + stride, fy = -14 - lift;
    const sh = dir > 0 ? "M-16,-13 Q-17,-31 4,-31 Q22,-30 31,-16 Q38,-10 35,-1 L-18,-1 Q-22,-6 -16,-13Z" : "M16,-13 Q17,-31 -4,-31 Q-22,-30 -31,-16 Q-38,-10 -35,-1 L18,-1 Q22,-6 16,-13Z";
    const sole = `<path d="M-40,-9 L40,-9 L40,4 L-40,4Z" fill="${dk(shoe, 0.45)}"/><path d="M-40,-9 L40,-9" stroke="${PAL.copper}" stroke-width="3"/>`;
    return limb(K, `M${x},${y0} L${fx},${fy}`, 14, c) + `<g transform="translate(${fx},${r2(fy + 14)})">${K.shape(sh, shoe, { sw: 3.6, rim: 8, inner: sole + `<ellipse cx="${dir * 6}" cy="-23" rx="14" ry="5" fill="#fff" opacity=".55" transform="rotate(${dir * -14} ${dir * 6} -23)"/>` })}</g>`;
  }
  const shadow = (K, rx = 96, ry = 15, dx = 0) => `<g class="shadow" data-depth="0.5">${K.blob(dx, 3, rx, ry, "#2a1830", 0.42)}${K.blob(dx, 2, rx * 0.55, ry * 0.55, "#1c1024", 0.38)}</g>`;
  const sweatDrop = (K, x, y, s = 1, rot = 0) => `<g class="sweat" transform="translate(${x},${y}) rotate(${rot}) scale(${s})"><path d="M0,-18 C10,-4 14,3 14,9 C14,18 -14,18 -14,9 C-14,3 -10,-4 0,-18Z" fill="${K.fill("#8fd0ff")}" stroke="${dk("#8fd0ff", 0.6)}" stroke-width="3" stroke-linejoin="round"/><ellipse cx="-4.5" cy="7" rx="3" ry="5" fill="#fff" opacity=".85" transform="rotate(14 -4.5 7)"/></g>`;
  const faceG = (o, id, f) => `<g class="face" id="${id}-face">${part(o, "cheeks", 0.55, f.cheeks)}${part(o, "eyes", 0.57, f.eyes, ` id="${id}-eyes"`)}${part(o, "brows", 0.58, f.brows)}${part(o, "mouth", 0.57, f.mouth, ` id="${id}-mouth"`)}</g>`;

  // ====================================================================== IL BADGE
  // tessera paffuta con fascia aziendale, clip metallica, chip, codice a barre, LED. Origine ai piedi.
  function badge(o = {}) {
    const K = ctx(), id = uid("badge"), W = 104, top = -350, bot = -56, sq = o.squash ?? 1;
    const aL = o.armL ?? [-176, -176, 0.32, "open"], aR = o.armR ?? [176, -176, -0.32, "open"];
    const armC = "#b97a4a", legC = "#8a5a3c", shoe = "#f4ede2";
    const body = pill(-W, top, W, bot, 40);
    const bars = [3, 5, 2, 4, 2, 6, 3, 2, 5, 3, 2, 4].reduce((a, w) => { a.s += `<rect x="${a.x}" y="-112" width="${w}" height="34" rx="1" fill="#4a2f24"/>`; a.x += w + 2.6; return a; }, { x: 6, s: "" }).s;
    const ledC = o.led === "ok" ? PAL.ledOk : PAL.led;
    const led = o.led ? K.blob(-22, -96, 22, 22, ledC, 0.7) + `<circle cx="-22" cy="-96" r="8.5" fill="${K.fill(ledC)}" stroke="${ol(ledC)}" stroke-width="2.6"/><circle cx="-24.5" cy="-99" r="2.6" fill="#fff" opacity=".9"/>` : `<circle cx="-22" cy="-96" r="8" fill="#9c8f86" stroke="#6f655e" stroke-width="2.4"/><circle cx="-24.5" cy="-99" r="2.4" fill="#fff" opacity=".6"/>`;
    const chip = K.shape(pill(-80, -114, -42, -80, 7), PAL.brass, { sw: 2.8, rim: 6, inner: `<path d="M-80,-103 L-42,-103 M-80,-91 L-42,-91 M-66,-114 L-66,-80 M-56,-114 L-56,-80" stroke="${dk(PAL.brass, 0.6)}" stroke-width="2" opacity=".75"/><rect x="-66" y="-103" width="10" height="12" fill="${lt(PAL.brass, 0.4)}" opacity=".7"/>` });
    const inner = `
      <rect x="${-W}" y="${top - 2}" width="${W * 2}" height="62" fill="${K.lin(PAL.copperL, PAL.copper)}"/>
      <rect x="${-W}" y="${top - 2}" width="${W * 2}" height="24" fill="${K.fade("#fff", 0.38, 0)}"/>
      <rect x="${-W}" y="${top + 56}" width="${W * 2}" height="4" fill="${dk(PAL.copper, 0.55)}" opacity=".6"/>
      <rect x="${-W}" y="${top + 60}" width="${W * 2}" height="30" fill="${K.fade("#3b1d28", 0.24, 0)}"/>
      <rect x="${-W}" y="${top + 60}" width="${W * 2}" height="2.4" fill="#fff" opacity=".6"/>
      <polygon points="-60,${top + 40} -14,${top + 40} -86,${bot + 10} -${W + 12},${bot + 10}" fill="${K.sheen()}" opacity=".7"/>
      <polygon points="-6,${top + 40} 10,${top + 40} -62,${bot + 10} -78,${bot + 10}" fill="${K.sheen()}" opacity=".35"/>
      ${K.blob(60, -140, 70, 120, "#c9a876", 0.22)}
      <path d="${pill(-W + 7, top + 7, W - 7, bot - 7, 32)}" fill="none" stroke="#fff" stroke-width="2.6" opacity=".55"/>`;
    const bandTxt = `<text x="0" y="${top + 42}" text-anchor="middle" font-family="Inter" font-weight="800" font-size="22" letter-spacing="2.4" fill="#fff6e8" stroke="${dk(PAL.copper, 0.6)}" stroke-width=".8" paint-order="stroke">NODOMATIC</text>`;
    const lower = `${chip}${led}${bars}${o.name ? `<text x="0" y="-62" text-anchor="middle" font-family="IBM Plex Mono" font-weight="700" font-size="14" fill="#7a5a44">${o.name}</text>` : ""}`;
    const slot = `<rect x="-24" y="${top + 11}" width="48" height="11" rx="5.5" fill="#3a2230" stroke="#1d1020" stroke-width="1.6" opacity="0"/>`;
    const clipHw = K.shape(pill(-20, top - 34, 20, top + 8, 11), "#d9dbe3", { sw: 3.4, rim: 8, inner: `<rect x="-6" y="${top - 34}" width="12" height="40" fill="${K.fade("#2d3550", 0.35, 0.05)}"/>` }) + `<path d="M-13,${top - 27} L-13,${top - 8}" stroke="#fff" stroke-width="3.4" stroke-linecap="round" opacity=".85"/>`;
    const sp = o.strap ?? 0, sd = `M-14,${top - 30} L-${34 + sp * 0.1},${top - 30 - sp} M14,${top - 30} L${34 + sp * 0.1},${top - 30 - sp}`;
    const strap = sp > 0 ? `<path d="${sd}" stroke="${ol(PAL.copper)}" stroke-width="19" stroke-linecap="round"/><path d="${sd}" stroke="${PAL.copper}" stroke-width="14" stroke-linecap="round"/><path d="${sd}" stroke="${PAL.copperL}" stroke-width="4" stroke-linecap="round" opacity=".7" transform="translate(-3,-2)"/>` : "";
    const f = face(K, { ex: 40, ey: -226, rx: 28, ry: 34, my: -160, ms: 1.12, cy: -190, lidCol: "#fdf1dc", iris: ["#2d1810", "#9a5c34"], ink: "#3a2218", browCol: "#4a2c1e", lip: "#8a3a3a", bt: 5.6 }, o);
    const armL = arm(K, -W + 6, -190, aL[0], aL[1], aL[2], aL[3], armC), armR = arm(K, W - 6, -190, aR[0], aR[1], aR[2], aR[3], armC);
    const stride = o.stride ?? null, sA = stride == null ? 0 : Math.sin(stride) * 14, l1 = stride == null ? 0 : Math.max(0, Math.sin(stride)) * 12, l2 = stride == null ? 0 : Math.max(0, -Math.sin(stride)) * 12;
    return wrapSvg(K, id, `<g class="badge" id="${id}" data-depth="0.5">${shadow(K)}<g class="squash" transform="scale(${r2(1 / Math.sqrt(sq))},${r2(sq)})">
      ${part(o, "legs", 0.44, leg(K, -36, bot - 6, -1, legC, shoe, -sA, l1) + leg(K, 36, bot - 6, 1, legC, shoe, sA, l2), ` id="${id}-legs"`)}
      ${o.lf ? "" : part(o, "armL", 0.46, armL, ` id="${id}-armL"`)}${o.rf ? "" : part(o, "armR", 0.54, armR, ` id="${id}-armR"`)}
      ${part(o, "body", 0.5, `${strap}${K.shape(body, PAL.cream, { sw: 4.6, rim: 13, inner })}${bandTxt}${lower}${slot}${clipHw}`, ` id="${id}-body"`)}
      ${faceG(o, id, f)}
      ${o.lf ? part(o, "armL", 0.62, armL, ` id="${id}-armL"`) : ""}${o.rf ? part(o, "armR", 0.62, armR, ` id="${id}-armR"`) : ""}
      ${o.sweat ? sweatDrop(K, W - 6, -318, 0.9, 12) : ""}
    </g></g>`);
  }

  // ====================================================================== IL MOUSE
  function mouse(o = {}) {
    const K = ctx(), id = uid("mouse"), sq = o.squash ?? 1;
    const base = PAL.mouse, shell = "M0,-264 C52,-264 80,-224 86,-170 C92,-110 72,-52 0,-50 C-72,-52 -92,-110 -86,-170 C-80,-224 -52,-264 0,-264Z";
    const aL = o.armL ?? [-132, -150, 0.34, "open"], aR = o.armR ?? [122, -100, -0.25, "open"], armC = mix(base, "#6f86ae", 0.4);
    const cableC = "#33415f", cd = "M0,-252 C4,-304 -50,-326 -80,-298 C-104,-274 -84,-240 -112,-232 C-134,-226 -146,-246 -140,-266";
    const cable = `<path d="${cd}" fill="none" stroke="${ol(cableC)}" stroke-width="15" stroke-linecap="round"/><path d="${cd}" fill="none" stroke="${cableC}" stroke-width="10" stroke-linecap="round"/><path d="${cd}" fill="none" stroke="#8aa0cc" stroke-width="2.6" stroke-linecap="round" opacity=".55" transform="translate(-2,-2)"/>
      <g transform="translate(-140,-268) rotate(-4)">${K.shape(pill(-11, -30, 11, 2, 4), "#cfd5e2", { sw: 3, rim: 6, inner: `<rect x="-7" y="-24" width="5" height="9" rx="1.5" fill="#35405c"/><rect x="2" y="-24" width="5" height="9" rx="1.5" fill="#35405c"/>` })}</g>`;
    const inner = `
      <path d="M-100,-290 L100,-290 L100,-212 Q0,-192 -100,-212Z" fill="${lt(base, 0.32)}" opacity=".55"/>
      <path d="M-100,-212 Q0,-192 100,-212" fill="none" stroke="${dk(base, 0.5)}" stroke-width="3.4" opacity=".75"/><path d="M-100,-209 Q0,-189 100,-209" fill="none" stroke="#fff" stroke-width="2" opacity=".5"/>
      <path d="M0,-270 L0,-203" stroke="${dk(base, 0.5)}" stroke-width="3.4" opacity=".75"/><path d="M2.4,-270 L2.4,-203" stroke="#fff" stroke-width="1.8" opacity=".45"/>
      ${K.blob(0, -190, 90, 16, "#3a2a50", 0.18)}
      <path d="M-66,-238 C-84,-206 -84,-150 -70,-100" fill="none" stroke="#fff" stroke-width="10" stroke-linecap="round" opacity=".5"/>
      <ellipse cx="-48" cy="-232" rx="22" ry="9" fill="#fff" opacity=".6" transform="rotate(-28 -48 -232)"/>
      ${K.blob(50, -90, 40, 40, "#c7a0d8", 0.22)}`;
    const wheel = K.shape(pill(-8, -250, 8, -216, 8), "#46506c", { sw: 3, rim: 5, inner: `<path d="M-8,-240 L8,-240 M-8,-232 L8,-232 M-8,-224 L8,-224" stroke="${PAL.copperL}" stroke-width="2.4"/>` });
    const f = face(K, { ex: 36, ey: -150, rx: 26, ry: 32, my: -92, ms: 1.0, cy: -118, lidCol: "#aab9d6", iris: ["#14264a", "#4c78b8"], ink: "#27345a", browCol: "#2a3558", lip: "#7a3040", bt: 5.2 }, { mouth: "smile", ...o });
    const cup = o.cup !== false ? `<g class="cup" transform="translate(${aR[0] + 4},${aR[1] - 26}) rotate(${o.cupTilt ?? 6})">
        ${K.shape("M-17,-22 L17,-22 L12,24 L-12,24Z", "#fffaf2", { sw: 3.4, rim: 7, inner: `<path d="M-20,-4 L20,-4 L17,12 L-17,12Z" fill="${PAL.copper}"/><path d="M-20,-4 L20,-4" stroke="${PAL.copperL}" stroke-width="2.4"/>` })}
        ${K.shape(pill(-20, -31, 20, -21, 4), "#3a3248", { sw: 3, rim: 5 })}
        <path d="M-3,-40 q-8,-10 1,-20 q7,-9 -1,-18 M9,-42 q-7,-8 1,-16" fill="none" stroke="#fff" stroke-width="3.6" stroke-linecap="round" opacity=".7"/></g>` : "";
    const sweat = o.sweat !== false ? sweatDrop(K, 100, -214, 0.95, 14) + sweatDrop(K, -104, -176, 0.7, -12) : "";
    const st = o.stride ?? null, sA = st == null ? 0 : Math.sin(st) * 14, l1 = st == null ? 0 : Math.max(0, Math.sin(st)) * 12, l2 = st == null ? 0 : Math.max(0, -Math.sin(st)) * 12;
    const aLs = arm(K, -80, -124, aL[0], aL[1], aL[2], aL[3], armC), aRs = arm(K, 80, -124, aR[0], aR[1], aR[2], aR[3], armC);
    return wrapSvg(K, id, `<g class="mouse" id="${id}" data-depth="0.5">${shadow(K, 100, 15)}<g class="squash" transform="scale(${r2(1 / Math.sqrt(sq))},${r2(sq)})">
      ${part(o, "cable", 0.4, cable)}
      ${part(o, "legs", 0.44, leg(K, -32, -58, -1, "#7a8fb8", "#fbfbff", -sA, l1) + leg(K, 32, -58, 1, "#7a8fb8", "#fbfbff", sA, l2), ` id="${id}-legs"`)}
      ${part(o, "armL", 0.46, aLs, ` id="${id}-armL"`)}
      ${part(o, "body", 0.5, K.shape(shell, base, { sw: 4.8, rim: 14, inner }) + wheel, ` id="${id}-body"`)}
      ${faceG(o, id, f)}
      ${part(o, "armR", 0.56, cup + aRs, ` id="${id}-armR"`)}${sweat}
    </g></g>`);
  }

  // ====================================================================== I GENI (strumenti IA con logo ufficiale)
  function medal(K, kind, x, y, r, ring) {
    const L = root.Loghi && root.Loghi[kind];
    const lg = L ? `<svg x="${r2(x - r * 0.68)}" y="${r2(y - r * 0.68)}" width="${r2(r * 1.36)}" height="${r2(r * 1.36)}" viewBox="${L.vb}" preserveAspectRatio="xMidYMid meet" overflow="visible">${L.g}</svg>` : "";
    return `<g class="medal">${K.blob(x + 3, y + r * 0.9, r * 1.05, r * 0.3, "#1d1024", 0.4)}
      ${K.shape(circ(x, y, r + 8), ring, { sw: 3.6, rim: 7 })}
      <circle cx="${x}" cy="${y}" r="${r}" fill="#ffffff" stroke="${ol(ring)}" stroke-width="2.4"/>
      <circle cx="${x}" cy="${y}" r="${r}" fill="${K.fade("#1d1024", 0, 0.14)}"/>
      ${lg}
      <path d="M${r2(x - r * 0.78)},${r2(y - r * 0.28)} A${r * 0.82},${r * 0.82} 0 0,1 ${r2(x - r * 0.2)},${r2(y - r * 0.8)}" fill="none" stroke="#fff" stroke-width="4" stroke-linecap="round" opacity=".9"/></g>`;
  }
  const star = (cx, cy, R, r, n, rot = 0) => Array.from({ length: n * 2 }, (_, i) => { const a = rot + (i * Math.PI) / n, rr = i % 2 ? r : R; return [cx + rr * Math.cos(a), cy + rr * Math.sin(a)]; }).map((p, i) => `${i ? "L" : "M"}${r2(p[0])},${r2(p[1])}`).join("") + "Z";

  function genie(kind, o = {}) {
    const K = ctx(), id = uid("genie"), ok = o.approved ?? false, sq = o.squash ?? 1;
    const c = ok ? PAL.teal : PAL.gratis, dir = ok ? 1 : -1;
    const C = ok
      ? [[56, -46, 2], [56, -58, 8], [46, -76, 17], [30, -100, 31], [14, -130, 47], [6, -164, 66], [3, -198, 80], [1, -232, 66]]
      : [[-60, -44, 2], [-58, -58, 8], [-46, -78, 17], [-30, -102, 31], [-14, -132, 47], [-6, -166, 66], [-3, -200, 80], [-1, -232, 66]];
    C.push([0, -262, 72]);
    const body = ribbon(C, { rx: 78, ry: 78 });
    const aL = o.armL ?? [-124, -196, 0.35, "open"], aR = o.armR ?? [124, -196, -0.35, "open"];
    const inner = `${K.blob(-24, -198, 60, 70, lt(c, 0.7), 0.5)}${K.blob(0, -226, 64, 18, "#102030", 0.2)}${K.blob(30, -96, 40, 50, dk(c, 0.5), 0.2)}`;
    const ring = ok ? PAL.brass : PAL.pink;
    const wing = "M-3,0 C-16,-16 -36,-18 -40,-6 C-43,6 -40,17 -34,18 C-22,18 -10,9 -3,2Z";
    const bow = ok ? `<g class="bow" transform="translate(0,-228)">${K.shape(wing, PAL.copper, { sw: 3.2, rim: 7 })}<g transform="scale(-1,1)">${K.shape(wing, PAL.copper, { sw: 3.2, rim: 7 })}</g>${K.shape(pill(-8, -9, 8, 11, 6), PAL.copperL, { sw: 3, rim: 5 })}</g>` : "";
    const hair = ok
      ? `<g transform="translate(2,-364)">${K.shape("M-9,-6 C-26,-30 8,-44 10,-66 C28,-46 26,-10 10,-4Z", PAL.tealL, { sw: 3.4, rim: 6 })}</g>`
      : `<g transform="translate(-4,-366)">${K.shape("M-26,6 L-34,-28 L-14,-10 L-8,-42 L6,-12 L22,-36 L24,2Z", PAL.gratisD, { sw: 3.4, rim: 6 })}</g>`;
    const shk = "M-9,-8 L-9,-16 Q-9,-26 0,-26 Q9,-26 9,-16 L9,-8";
    const padlock = ok ? `<g class="lock" transform="translate(40,-132)"><path d="${shk}" fill="none" stroke="${ol("#c9ccd8")}" stroke-width="8" stroke-linecap="round"/><path d="${shk}" fill="none" stroke="#c9ccd8" stroke-width="4.6" stroke-linecap="round"/>${K.shape(pill(-14, -9, 14, 14, 6), PAL.brass, { sw: 3, rim: 6 })}<circle cx="0" cy="0" r="3.4" fill="${dk(PAL.brass, 0.7)}"/><rect x="-1.4" y="1" width="2.8" height="7" rx="1.2" fill="${dk(PAL.brass, 0.7)}"/></g>`
      : `<g class="gratis" transform="translate(56,-132) rotate(14)">${K.shape(star(0, 0, 52, 41, 12, 0.1), PAL.pink, { sw: 3.6, rim: 8 })}<text x="0" y="6" text-anchor="middle" font-family="Inter" font-weight="800" font-size="19" letter-spacing=".6" fill="#fff" stroke="${ol(PAL.pink)}" stroke-width=".6">GRATIS</text></g>`;
    const mood = ok
      ? { mouth: "smile", eyes: { open: 1, px: 0, py: 0.05 }, brow: { curve: 6, worry: -0.1 } }
      : { mouth: "smirk", eyes: { open: 1, px: 0.25, py: 0, lid: 0.28 }, brow: { curve: 8, tilt: 4, raiseR: 9, worry: -0.45 } };
    const f = face(K, { ex: 33, ey: -302, rx: 25, ry: 31, my: -250, ms: 1.0, cy: -266, lidCol: mix(c, "#fff", 0.1), iris: ok ? ["#0c3430", "#2fa79a"] : ["#3a2600", "#c68a00"], ink: ok ? "#10403a" : "#5a3a00", browCol: ok ? "#14504a" : "#7a4a00", lip: "#7a2e3c", bt: 5.4 },
      { ...mood, ...o, eyes: { ...mood.eyes, ...(o.eyes ?? {}) }, brow: { ...mood.brow, ...(o.brow ?? {}) }, mouth: o.mouth ?? mood.mouth });
    const armC = ok ? mix(c, "#1d7468", 0.18) : mix(c, "#f4b400", 0.3);
    const aLs = arm(K, -66, -214, aL[0], aL[1], aL[2], aL[3], armC, 14), aRs = arm(K, 66, -214, aR[0], aR[1], aR[2], aR[3], armC, 14);
    const sparks = ok ? "" : [[-96, -330, 9], [100, -300, 7], [-84, -120, 6]].map(([x, y, s]) => `<path d="M${x},${y - s} Q${x},${y} ${x + s},${y} Q${x},${y} ${x},${y + s} Q${x},${y} ${x - s},${y} Q${x},${y} ${x},${y - s}Z" fill="#fff" stroke="${PAL.gratisD}" stroke-width="1.4"/>`).join("");
    const fl = o.float ?? 0;
    return wrapSvg(K, id, `<g class="genie genie-${kind} ${ok ? "approved" : "unauthorized"}" id="${id}" data-depth="0.5"><g class="shadow" data-depth="0.5">${K.blob(dir * 12, 3, 88 - fl * 0.2, 13, "#2a1830", 0.36)}${K.blob(dir * 12, 2, 50, 7, "#1c1024", 0.34)}</g>
      <g class="squash" transform="translate(0,${-fl}) rotate(${o.tilt ?? 0} 0 -60) scale(${r2(1 / Math.sqrt(sq))},${r2(sq)})">
      ${o.lf ? "" : part(o, "armL", 0.46, aLs, ` id="${id}-armL"`)}${o.rf ? "" : part(o, "armR", 0.54, aRs, ` id="${id}-armR"`)}
      ${part(o, "body", 0.5, K.shape(body, c, { sw: 4.8, rim: 14, inner }) + hair + bow + medal(K, kind, 0, -176, 46, ring) + padlock, ` id="${id}-body"`)}
      ${faceG(o, id, f)}
      ${o.lf ? part(o, "armL", 0.62, aLs, ` id="${id}-armL"`) : ""}${o.rf ? part(o, "armR", 0.62, aRs, ` id="${id}-armR"`) : ""}${sparks}
    </g></g>`);
  }

  // ====================================================================== L'UFFICIO a strati di profondita'
  // Coordinate 1920x1080. Strati: lontano (0), medio (0.5), davanti (1). Piu' lontano = piu' sfocato.
  // o: { cam:[cx,cy] (-1..1), camK (px di parallasse a profondita' 1, default 60), between (svg tra medio e davanti) }
  function office(o = {}) {
    const K = ctx(), id = uid("office"), W = 1920, m = 140, kk = o.camK ?? 60, cam = o.cam ?? [0, 0];
    const tr = (d) => `translate(${r2(-cam[0] * d * kk)},${r2(-cam[1] * d * kk)})`;
    const blur = (s) => { const i = K.uid("bl"); K.defs.push(`<filter id="${i}" x="-8%" y="-8%" width="116%" height="116%"><feGaussianBlur stdDeviation="${s}"/></filter>`); return i; };
    const wallD = "#d2b88e";
    // --- strato 0: parete di fondo
    let slats = ""; for (let i = 0; i < 17; i++) slats += `<rect x="150" y="${150 + i * 24}" width="400" height="14" rx="3" fill="${i % 2 ? "#fdf6e6" : "#efe3cb"}"/>`;
    const binders = [[PAL.copper, 120], [PAL.kraft, 100], [PAL.teal, 112], [PAL.navyMid, 96], [PAL.copperL, 108], [PAL.kraft, 90]].reduce((a, [c, h]) => { a.s += K.shape(pill(a.x, 420 - h, a.x + 46, 420, 7), c, { sw: 3.4, rim: 6, inner: `<rect x="${a.x + 10}" y="${420 - h + 18}" width="26" height="30" rx="3" fill="#fff" opacity=".8"/>` }); a.x += 52; return a; }, { x: 1380, s: "" }).s;
    const far = `<rect x="${-m}" y="${-m}" width="${W + 2 * m}" height="${760 + m}" fill="${K.lin("#f2e4c8", wallD)}"/>
      <rect x="${-m}" y="640" width="${W + 2 * m}" height="140" fill="${K.fade("#a07a4a", 0, 0.35)}"/>
      <rect x="${-m}" y="690" width="${W + 2 * m}" height="90" fill="#cda97a"/><rect x="${-m}" y="686" width="${W + 2 * m}" height="8" fill="#e9cfa6"/>
      ${K.shape(pill(130, 120, 570, 600, 12), "#fffaf0", { sw: 5, rim: 8, inner: `<rect x="150" y="140" width="400" height="440" fill="${K.lin("#cfe5ea", "#e9f2ee")}"/><ellipse cx="420" cy="220" rx="90" ry="30" fill="#fff" opacity=".8"/>${slats}<path d="M150,500 Q300,460 420,490 T550,470 L550,580 L150,580Z" fill="#9cc2a8" opacity=".6"/>` })}
      <polygon points="150,150 550,150 1180,760 640,760" fill="${K.fade("#fff6d8", 0.5, 0.02)}" opacity=".7"/>
      ${K.shape(circ(980, 250, 108), "#f3e7d0", { sw: 6, rim: 12, inner: `<circle cx="980" cy="250" r="86" fill="#fffaf0"/>` })}
      <g stroke="#6b4c3c" stroke-width="5" stroke-linecap="round"><path d="M980,250 L980,188"/><path d="M980,250 L1028,276" stroke-width="7"/></g><circle cx="980" cy="250" r="9" fill="${PAL.copper}"/>
      <g>${K.shape(pill(1360, 424, 1830, 444, 8), "#c79a68", { sw: 4, rim: 8 })}${binders}</g>
      ${K.shape(pill(1100, 120, 1260, 300, 10), "#e2c590", { sw: 4, rim: 8, inner: `<rect x="1116" y="136" width="52" height="40" fill="#fff6a8" transform="rotate(-4 1140 156)"/><rect x="1180" y="150" width="56" height="44" fill="#ffc7d6" transform="rotate(5 1200 170)"/><rect x="1130" y="206" width="58" height="46" fill="#bfeadf" transform="rotate(3 1160 230)"/>` })}`;
    // --- strato 0.5: scrivania, portatile, pianta, tazza, lampada
    const plant = `<g transform="translate(120,720)">${[-60, -34, -8, 20, 46].map((a, i) => `<path d="M0,0 C${a * 1.2},-80 ${a * 2},-150 ${a * 2.6},${-(170 + i * 20)}" stroke="${ol(PAL.teal)}" stroke-width="3" fill="none"/>${K.shape(`M0,0 C${a - 40},-70 ${a * 2 - 30},-150 ${a * 2.6},${-(180 + i * 22)} C${a * 2 + 36},-140 ${a + 40},-60 0,0Z`, mix(PAL.teal, "#7fb86a", i * 0.15), { sw: 3, rim: 6 })}`).join("")}${K.shape("M-62,0 L62,0 L48,96 L-48,96Z", PAL.copper, { sw: 4, rim: 9, inner: `<rect x="-70" y="0" width="140" height="18" fill="${PAL.copperL}" opacity=".7"/>` })}</g>`;
    const laptop = `<g transform="translate(560,0)">${K.shape("M-60,820 L720,820 L760,870 L-100,870Z", "#aebccd", { sw: 4, rim: 8 })}
      ${K.shape(pill(0, 470, 660, 826, 20), "#c7d2e0", { sw: 5, rim: 10, inner: `<rect x="24" y="494" width="612" height="306" rx="10" fill="${K.lin(PAL.tealL, PAL.teal)}"/><rect x="24" y="494" width="612" height="56" fill="${PAL.tealD}"/><rect x="60" y="600" width="540" height="46" rx="23" fill="#fff" opacity=".9"/><rect x="60" y="680" width="360" height="12" rx="6" fill="#fff" opacity=".55"/><rect x="60" y="708" width="260" height="12" rx="6" fill="#fff" opacity=".4"/>` })}${K.blob(330, 650, 380, 220, PAL.tealL, 0.18)}</g>`;
    const mug = `<g transform="translate(1500,830)">${K.shape("M-46,-92 L46,-92 L40,0 L-40,0Z", "#fffaf0", { sw: 4.4, rim: 10, inner: `<rect x="-50" y="-50" width="100" height="26" fill="${PAL.copper}"/>` })}<path d="M44,-70 C84,-70 84,-26 40,-24" fill="none" stroke="${ol("#fffaf0")}" stroke-width="13" stroke-linecap="round"/><path d="M44,-70 C84,-70 84,-26 40,-24" fill="none" stroke="#fffaf0" stroke-width="8" stroke-linecap="round"/></g>`;
    const lamp = `<g transform="translate(1730,820)">${K.shape(pill(-60, -16, 60, 4, 8), "#33415f", { sw: 4, rim: 7 })}<path d="M0,-12 L-60,-250 L40,-300" fill="none" stroke="${ol("#aebccd")}" stroke-width="17" stroke-linecap="round" stroke-linejoin="round"/><path d="M0,-12 L-60,-250 L40,-300" fill="none" stroke="#aebccd" stroke-width="11" stroke-linecap="round" stroke-linejoin="round"/><g transform="translate(40,-300) rotate(-24)">${K.shape("M-60,-10 L60,-10 L36,50 L-36,50Z", PAL.copper, { sw: 4.4, rim: 9 })}</g>${K.blob(60, -230, 90, 110, "#ffe9a8", 0.28)}</g>`;
    const mid = `<rect x="${-m}" y="770" width="${W + 2 * m}" height="${340 + m}" fill="${K.lin("#c88e58", "#a8683e")}"/>
      <rect x="${-m}" y="770" width="${W + 2 * m}" height="14" fill="#e9b886"/><rect x="${-m}" y="784" width="${W + 2 * m}" height="60" fill="${K.fade("#fff2d2", 0.38, 0)}"/>
      ${K.blob(960, 790, 1100, 30, "#4a2a1c", 0.25)}
      ${plant}${laptop}${mug}${lamp}`;
    // --- strato 1: bordo anteriore della scrivania, matite e carte a fuoco
    const front = `<rect x="${-m}" y="1000" width="${W + 2 * m}" height="${80 + m}" fill="${K.lin("#8f5632", "#6a3b24")}"/><rect x="${-m}" y="994" width="${W + 2 * m}" height="14" fill="#d8a06c"/><rect x="${-m}" y="1008" width="${W + 2 * m}" height="4" fill="#fff" opacity=".25"/>
      ${K.blob(960, 996, 1100, 22, "#2a1408", 0.4)}
      <g transform="translate(1700,1004)">${K.shape("M-64,0 L-72,-120 L72,-120 L64,0Z", PAL.teal, { sw: 4.6, rim: 10, inner: `<rect x="-80" y="-76" width="160" height="22" fill="${PAL.tealD}" opacity=".5"/>` })}${[[-24, -170, PAL.copper], [2, -190, PAL.brass], [26, -164, PAL.navyMid]].map(([x, h, c]) => `<rect x="${x - 7}" y="${h}" width="14" height="${-h - 100}" rx="4" fill="${c}" stroke="${ol(c)}" stroke-width="3.4"/>`).join("")}</g>
      <g transform="translate(120,1010) rotate(-2)">${K.shape(pill(-110, -26, 110, 0, 5), "#fffaf0", { sw: 3.6, rim: 8 })}${K.shape(pill(-100, -48, 100, -24, 5), "#f3e6cc", { sw: 3.6, rim: 8 })}</g>`;
    const L = (cls, d, content, fil) => `<g class="layer ${cls}" data-depth="${d}" transform="${tr(d)}"${fil ? ` filter="url(#${fil})"` : ""}>${content}</g>`;
    return wrapSvg(K, id, `<g class="office" id="${id}">${L("far", 0, far, blur(3.4))}${L("mid", 0.5, mid, blur(1.1))}${o.between ? `<g class="between" data-depth="0.6" transform="${tr(0.6)}">${o.between}</g>` : ""}${L("front", 1, front, null)}</g>`);
  }

  root.StileB = { badge, mouse, genie, office, mouth: (kind, s = 1) => { const K = ctx(); return wrapSvg(K, "m", mouth(K, kind, s)); }, palette: PAL, version: "B-morbido-1" };
})(typeof window !== "undefined" ? window : globalThis);
