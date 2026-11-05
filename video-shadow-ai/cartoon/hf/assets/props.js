// Oggetti di scena del cartone (stesso linguaggio dei personaggi: contorno d'inchiostro, ombra + luce).
(function (root) {
  const { P } = root.Chars;
  const S = 5;
  const sheet = (o = {}) => `<g opacity="${o.o ?? 1}"><rect x="-90" y="-270" width="180" height="250" rx="8" fill="${o.c ?? P.paper}" stroke="${P.ink}" stroke-width="${S}"/>
    ${[0, 1, 2, 3, 4].map((i) => `<line x1="-62" x2="${30 - (i % 2) * 30}" y1="${-220 + i * 26}" y2="${-220 + i * 26}" stroke="${P.paperLine}" stroke-width="6" stroke-linecap="round"/>`).join("")}</g>`;
  const stamp = (o = {}) => `<g><rect x="-12" y="-96" width="24" height="56" rx="12" fill="${P.navyMid}" stroke="${P.ink}" stroke-width="4"/><circle cy="-100" r="20" fill="${P.navyMid}" stroke="${P.ink}" stroke-width="4"/>
    <rect x="-44" y="-44" width="88" height="22" rx="6" fill="${P.kraftShade}" stroke="${P.ink}" stroke-width="4"/><rect x="-50" y="-24" width="100" height="24" rx="5" fill="${o.c ?? P.copper}" stroke="${P.ink}" stroke-width="4"/></g>`;
  // cartellino a terra con timbro (stamped 0..1)
  const tagCard = (text, o = {}) => { const w = o.w ?? 420, st = o.stamped ?? 0;
    return `<g transform="rotate(${o.rot ?? 0})"><rect x="${-w / 2}" y="-34" width="${w}" height="68" rx="5" fill="#f1ebde" stroke="${P.ink}" stroke-width="4"/>
      <text x="${-w / 2 + 22}" y="10" font-family="Inter" font-weight="800" font-size="24" letter-spacing="2" fill="${P.ink}">${text}</text>
      ${st > 0 ? `<g transform="translate(${w / 2 - 46},0) rotate(-10) scale(${0.6 + 0.4 * Math.min(1, st)})" opacity="${Math.min(1, st)}"><rect x="-28" y="-24" width="56" height="48" rx="9" fill="none" stroke="${P.copper}" stroke-width="5"/><path d="M-15,0 L-4,12 L16,-12" fill="none" stroke="${P.copper}" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/></g>` : ""}</g>`; };
  // cupola di carta da lucido (l'azienda che protegge)
  const dome = (o = {}) => `<g opacity="${o.o ?? 1}"><defs><radialGradient id="domeG" cx="50%" cy="100%" r="70%"><stop offset=".68" stop-color="#fffaf0" stop-opacity=".04"/><stop offset=".97" stop-color="#fffaf0" stop-opacity=".3"/><stop offset="1" stop-color="#fffaf0" stop-opacity=".5"/></radialGradient></defs>
    <path d="M-760,0 C-760,-570 760,-570 760,0Z" fill="url(#domeG)"/><path d="M-760,0 C-760,-570 760,-570 760,0" fill="none" stroke="#fffaf0" stroke-width="6" opacity=".7"/>
    <path d="M-630,-160 C-580,-360 -360,-440 -120,-458" fill="none" stroke="#fff" stroke-width="12" stroke-linecap="round" opacity="${0.3 * (o.shine ?? 1)}"/>
    ${o.glow ? `<path d="M-760,0 C-760,-570 760,-570 760,0" fill="none" stroke="${P.copperLight}" stroke-width="14" opacity="${o.glow * 0.5}"/>` : ""}</g>`;
  const clientCard = (o = {}) => `<g transform="rotate(${o.r ?? 0})"><rect x="-60" y="-40" width="120" height="80" rx="8" fill="#f8f4ec" stroke="${P.ink}" stroke-width="4"/>
    <rect x="-46" y="-26" width="34" height="34" rx="6" fill="${o.c ?? P.teal}"/><text x="-29" y="-2" text-anchor="middle" font-family="Inter" font-weight="800" font-size="16" fill="#fff">${o.l ?? "C"}</text>
    <line x1="0" x2="44" y1="-18" y2="-18" stroke="${P.paperLine}" stroke-width="6" stroke-linecap="round"/><line x1="0" x2="32" y1="-2" y2="-2" stroke="${P.paperLine}" stroke-width="6" stroke-linecap="round"/>
    <line x1="-46" x2="44" y1="22" y2="22" stroke="${P.paperLine}" stroke-width="5" stroke-linecap="round"/></g>`;
  const folder = (o = {}) => { const a = o.open ?? 0;
    return `<g><path d="M-120,-150 L-50,-150 L-34,-130 L120,-130 L120,0 L-120,0Z" fill="${P.kraftShade}" stroke="${P.ink}" stroke-width="${S}" stroke-linejoin="round"/>
      <g transform="translate(-120,0) skewX(${-a * 28}) scale(1,${1 - a * 0.45})"><path d="M0,-120 L240,-120 L240,0 L0,0Z" fill="${P.kraft}" stroke="${P.ink}" stroke-width="${S}" stroke-linejoin="round"/>
      <g transform="translate(120,-62) rotate(-8)"><rect x="-74" y="-20" width="148" height="40" rx="4" fill="none" stroke="${P.copper}" stroke-width="5"/><text y="9" text-anchor="middle" font-family="Inter" font-weight="800" font-size="22" letter-spacing="2" fill="${P.copper}">RISERVATO</text></g></g></g>`; };
  const padlock = (o = {}) => { const op = o.open ?? 0;
    return `<g><path d="M-26,-40 L-26,${-66 - op * 26} Q-26,${-96 - op * 26} 0,${-96 - op * 26} Q26,${-96 - op * 26} 26,${-66 - op * 26} L26,${-40 - op * 30}" fill="none" stroke="${P.ink}" stroke-width="9" transform="rotate(${op * 18} -26 -40)"/>
      <rect x="-40" y="-46" width="80" height="62" rx="10" fill="${P.brass}" stroke="${P.ink}" stroke-width="${S}"/><circle cy="-18" r="8" fill="${P.ink}"/><rect x="-3" y="-18" width="6" height="18" fill="${P.ink}"/></g>`; };
  const pen = () => `<g><rect x="-8" y="-120" width="16" height="110" rx="6" fill="${P.navyMid}" stroke="${P.ink}" stroke-width="4"/><path d="M-8,-10 L8,-10 L0,12Z" fill="${P.brass}" stroke="${P.ink}" stroke-width="4" stroke-linejoin="round"/><rect x="-10" y="-124" width="20" height="18" rx="5" fill="${P.copper}" stroke="${P.ink}" stroke-width="4"/></g>`;
  const rulebook = (o = {}) => { const tear = o.tear ?? 0;
    return `<g><g transform="translate(${-tear * 14},0) rotate(${-tear * 8})"><path d="M-110,-150 L0,-140 L0,0 L-110,-10Z" fill="${P.teal}" stroke="${P.ink}" stroke-width="${S}" stroke-linejoin="round"/><text x="-56" y="-66" text-anchor="middle" font-family="Inter" font-weight="800" font-size="20" fill="#fff" transform="rotate(5 -56 -66)">REGOLE</text></g>
      <g transform="translate(${tear * 14},0) rotate(${tear * 8})"><path d="M0,-140 L110,-150 L110,-10 L0,0Z" fill="${P.tealShade}" stroke="${P.ink}" stroke-width="${S}" stroke-linejoin="round"/>${[0, 1, 2].map((i) => `<line x1="20" x2="88" y1="${-110 + i * 26}" y2="${-114 + i * 26}" stroke="#bfe1da" stroke-width="5" stroke-linecap="round"/>`).join("")}</g></g>`; };
  const cross = (s = 1) => `<g transform="scale(${s})"><path d="M-14,-14 L14,14 M14,-14 L-14,14" stroke="#d6402e" stroke-width="9" stroke-linecap="round"/></g>`;
  const mortarboard = () => `<g><path d="M-60,-8 L0,-34 L60,-8 L0,18Z" fill="${P.ink}"/><rect x="-30" y="0" width="60" height="22" rx="4" fill="${P.ink}"/><path d="M44,-6 L48,36" stroke="${P.brass}" stroke-width="5"/><circle cx="48" cy="40" r="7" fill="${P.brass}"/></g>`;
  const steam = (t) => [0, 1, 2].map((i) => { const ph = ((t * 0.5 + i / 3) % 1); return `<path d="M${-14 + i * 14},${-10 - ph * 60} q-16,-22 4,-44 q16,-22 -4,-46" fill="none" stroke="#fff" stroke-width="7" stroke-linecap="round" opacity="${0.5 * Math.sin(ph * Math.PI)}"/>`; }).join("");
  const secondHand = (t, r = 78) => `<g transform="rotate(${(Math.floor(t) % 60) * 6 + 90})"><line x1="0" y1="10" x2="0" y2="${-r}" stroke="${P.copper}" stroke-width="3.5" stroke-linecap="round"/></g>`;
  const speedLines = (n = 3, len = 110, o = 0.7) => Array.from({ length: n }, (_, i) => `<path d="M0,${-60 - i * 50} l${len - i * 20},0" stroke="#fff" stroke-width="8" stroke-linecap="round" opacity="${o}"/>`).join("");
  const sparkle = (r = 14, c = "#fff") => `<path d="M0,${-r} L${r * .28},${-r * .28} L${r},0 L${r * .28},${r * .28} L0,${r} L${-r * .28},${r * .28} L${-r},0 L${-r * .28},${-r * .28}Z" fill="${c}" stroke="${P.ink}" stroke-width="3" stroke-linejoin="round"/>`;
  // lavagnetta di carta appesa con tre caselle da spuntare (checks: [0..1, 0..1, 0..1])
  const checklist = (o = {}) => { const ch = o.checks ?? [0, 0, 0], rows = o.rows ?? ["Strumento approvato", "Regole chiare", "Persone formate"], W = o.w ?? 780, H = 330, fs = o.fs ?? 44;
    return `<g><line x1="${-W / 2 + 90}" y1="${-400}" x2="${-W / 2 + 90}" y2="0" stroke="#8e6640" stroke-width="4"/><line x1="${W / 2 - 90}" y1="${-400}" x2="${W / 2 - 90}" y2="0" stroke="#8e6640" stroke-width="4"/>
      <rect x="${-W / 2}" y="6" width="${W}" height="${H}" rx="8" fill="#000" opacity=".16"/>
      <rect x="${-W / 2}" y="0" width="${W}" height="${H}" rx="8" fill="#f6f1e6" stroke="${P.ink}" stroke-width="${S}"/>
      <text x="0" y="58" text-anchor="middle" font-family="Inter" font-weight="800" font-size="26" letter-spacing="4" fill="${P.copper}">TRE COSE IN PIÙ</text>
      ${rows.map((r, i) => { const y = 120 + i * 72, u = Math.min(1, ch[i] ?? 0);
        return `<rect x="${-W / 2 + 56}" y="${y - 26}" width="52" height="52" rx="8" fill="#fff" stroke="${P.ink}" stroke-width="4"/>
          ${u > 0 ? `<g transform="translate(${-W / 2 + 82},${y}) rotate(-8) scale(${0.6 + 0.4 * u})" opacity="${u}"><path d="M-18,0 L-5,14 L22,-18" fill="none" stroke="${P.copper}" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/></g>` : ""}
          <text x="${-W / 2 + 132}" y="${y + 15}" font-family="EB Garamond" font-weight="700" font-size="${fs}" fill="${P.ink}" opacity="${0.45 + 0.55 * Math.max(u, o.lit?.[i] ?? 0)}">${r}</text>`; }).join("")}</g>`; };
  // campana di vetro (l'azienda che "tiene il documento sotto una campana di vetro"): vetro con riflessi, pomello,
  // base in ottone con targhetta. shine: passaggio di luce sul vetro (0..1), glow: calore interno (0..1)
  const belljar = (o = {}) => { const Wd = o.w ?? 760, Hh = o.h ?? 560, sh = o.shine ?? -1, gl = o.glow ?? 0, u = o.uid ?? (root.Anim?._sid ?? "bj");
    const body = `M${-Wd},0 C${-Wd - 8},${-Hh * 0.55} ${-Wd * 0.86},${-Hh * 0.93} ${-Wd * 0.5},${-Hh * 0.99} Q0,${-Hh * 1.06} ${Wd * 0.5},${-Hh * 0.99} C${Wd * 0.86},${-Hh * 0.93} ${Wd + 8},${-Hh * 0.55} ${Wd},0Z`;
    return `<g opacity="${o.o ?? 1}"><defs>
        <linearGradient id="bjFill-${u}" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#e6f4f1" stop-opacity=".22"/><stop offset=".3" stop-color="#ffffff" stop-opacity=".05"/><stop offset=".75" stop-color="#ffffff" stop-opacity=".03"/><stop offset="1" stop-color="#cfe8e3" stop-opacity=".2"/></linearGradient>
        <radialGradient id="bjGlow-${u}" cx="50%" cy="85%" r="60%"><stop offset="0" stop-color="#ffcf9a" stop-opacity="${0.42 * gl}"/><stop offset="1" stop-color="#ffcf9a" stop-opacity="0"/></radialGradient>
        <clipPath id="bjClip-${u}"><path d="${body}"/></clipPath>
        <linearGradient id="bjShine-${u}" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset=".5" stop-color="#fff" stop-opacity=".55"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient></defs>
      <ellipse cx="0" cy="22" rx="${Wd + 40}" ry="20" fill="#3a2312" opacity=".25"/>
      <path d="${body}" fill="url(#bjFill-${u})"/><path d="${body}" fill="url(#bjGlow-${u})"/>
      <g clip-path="url(#bjClip-${u})">
        <path d="M${-Wd * 0.8},-30 C${-Wd * 0.84},${-Hh * 0.5} ${-Wd * 0.7},${-Hh * 0.82} ${-Wd * 0.42},${-Hh * 0.9}" fill="none" stroke="#fff" stroke-width="26" stroke-linecap="round" opacity=".42"/>
        <path d="M${-Wd * 0.64},-60 C${-Wd * 0.67},${-Hh * 0.42} ${-Wd * 0.58},${-Hh * 0.66} ${-Wd * 0.4},${-Hh * 0.74}" fill="none" stroke="#fff" stroke-width="9" stroke-linecap="round" opacity=".35"/>
        <path d="M${Wd * 0.3},${-Hh * 0.95} Q${Wd * 0.62},${-Hh * 0.9} ${Wd * 0.78},${-Hh * 0.62}" fill="none" stroke="#fff" stroke-width="12" stroke-linecap="round" opacity=".3"/>
        <rect x="${-Wd * 0.5}" y="${-Hh * 0.7}" width="70" height="${Hh * 0.42}" rx="30" fill="#fff" opacity=".08" transform="skewX(-12)"/>
        ${sh >= 0 && sh <= 1 ? `<rect x="${-Wd - 300 + sh * (2 * Wd + 600)}" y="${-Hh * 1.1}" width="220" height="${Hh * 1.2}" fill="url(#bjShine-${u})" transform="skewX(-24)"/>` : ""}
      </g>
      <path d="${body}" fill="none" stroke="#fffaf0" stroke-width="6" opacity=".85"/>
      <path d="${body}" fill="none" stroke="#9fd0c8" stroke-width="2.5" opacity=".6" transform="translate(0,-6) scale(.985)"/>
      <g transform="translate(0,${-Hh * 1.03})"><rect x="-14" y="-24" width="28" height="30" rx="6" fill="#e9f5f2" opacity=".55" stroke="#fffaf0" stroke-width="4"/>
        <circle cy="-42" r="26" fill="#e9f5f2" opacity=".6" stroke="#fffaf0" stroke-width="5"/><circle cx="-8" cy="-50" r="7" fill="#fff" opacity=".9"/></g>
      <rect x="${-Wd - 26}" y="-14" width="${2 * Wd + 52}" height="32" rx="12" fill="${P.brass}" stroke="${P.ink}" stroke-width="4"/>
      <rect x="${-Wd - 16}" y="-10" width="${2 * Wd + 32}" height="8" rx="4" fill="#f3cf86" opacity=".8"/>
      <g transform="translate(0,2)"><rect x="-74" y="-12" width="148" height="26" rx="5" fill="#e8bf66" stroke="${P.ink}" stroke-width="3"/>
        <text y="8" text-anchor="middle" font-family="Inter" font-weight="800" font-size="15" letter-spacing="3" fill="${P.ink}">AZIENDA</text></g></g>`; };
  // ---- protezioni finali (tre alternative alla campana)
  // CASA POP-UP DI CARTA: pareti e tetto si alzano dalla scrivania come in un libro pop-up (u: 0..1)
  const house = (o = {}) => { const u = o.u ?? 1, uw = Math.min(1, u / 0.55), ur = Math.max(0, (u - 0.45) / 0.55), L = o.l ?? 160, R = o.r ?? 1380, F = o.floor ?? 900, top = o.top ?? 400, ax = (L + R) / 2, ay = o.apex ?? 110, gl = o.glow ?? 0;
    const wall = (x, dir) => `<g transform="translate(${x},${F}) scale(1,${uw})"><rect x="${dir < 0 ? -44 : 0}" y="${top - F}" width="44" height="${F - top}" fill="#c99d6c" stroke="${P.ink}" stroke-width="5"/>
      <rect x="${dir < 0 ? -44 : 0}" y="${top - F}" width="14" height="${F - top}" fill="#a67b4d" opacity=".6"/>${[0.25, 0.5, 0.75].map((k) => `<line x1="${dir < 0 ? -44 : 0}" x2="${dir < 0 ? 0 : 44}" y1="${(top - F) * k}" y2="${(top - F) * k}" stroke="${P.ink}" stroke-width="2.5" opacity=".35"/>`).join("")}</g>`;
    const roofSide = (sx, sy) => { const mx = ax + (sx - ax) * ur, my = ay + (sy - ay) * ur, th = 74;
      const lines = Array.from({ length: 6 }, (_, i) => { const k = (i + 1) / 7; return `<line x1="${ax + (mx - ax) * k}" y1="${ay + (my - ay) * k + 8}" x2="${ax + (mx - ax) * k}" y2="${ay + (my - ay) * k + th - 6}" stroke="#8f3f1b" stroke-width="4" opacity=".55"/>`; }).join("");
      return `<path d="M${ax},${ay + th} L${mx},${my + th} L${mx},${my + th + 18} L${ax},${ay + th + 18}Z" fill="#000" opacity=".12"/>
        <path d="M${ax},${ay} L${mx},${my} L${mx},${my + th} L${ax},${ay + th}Z" fill="${P.copper}" stroke="${P.ink}" stroke-width="5" stroke-linejoin="round"/>${lines}
        <path d="M${ax},${ay + 6} L${mx},${my + 6}" stroke="${P.copperLight}" stroke-width="8" stroke-linecap="round" opacity=".7"/>`; };
    const shingles = ur > 0 ? Array.from({ length: 5 }, (_, i) => { const k = (i + 1) / 6; return `<path d="M${ax + (L - 50 - ax) * k * ur},${ay + (top + 30 - ay) * k * ur} L${ax + (R + 50 - ax) * k * ur},${ay + (top + 30 - ay) * k * ur}" stroke="#8f3f1b" stroke-width="4" opacity=".0"/>`; }).join("") : "";
    const chim = ur > 0.6 ? `<g transform="translate(${ax + (R - ax) * 0.55},${ay + (top - ay) * 0.55 - 70})"><rect x="-30" y="0" width="60" height="90" fill="#a67b4d" stroke="${P.ink}" stroke-width="5"/><rect x="-38" y="-14" width="76" height="20" rx="4" fill="#8e6640" stroke="${P.ink}" stroke-width="4"/>
      ${[0, 1, 2].map((i) => `<circle cx="${10 + i * 14}" cy="${-36 - i * 34}" r="${14 + i * 6}" fill="#fff" opacity="${0.5 - i * 0.12}"/>`).join("")}</g>` : "";
    const sign = ur > 0.85 ? `<g transform="translate(${ax},${ay + 120})"><rect x="-90" y="-22" width="180" height="44" rx="6" fill="${P.brass}" stroke="${P.ink}" stroke-width="4"/><text y="9" text-anchor="middle" font-family="Inter" font-weight="800" font-size="22" letter-spacing="3" fill="${P.ink}">AZIENDA</text></g>` : "";
    return `<g><defs><radialGradient id="hsGlow" cx="50%" cy="75%" r="60%"><stop offset="0" stop-color="#ffcf9a" stop-opacity="${0.35 * gl}"/><stop offset="1" stop-color="#ffcf9a" stop-opacity="0"/></radialGradient></defs>
      <rect x="${L}" y="${top}" width="${R - L}" height="${F - top}" fill="url(#hsGlow)"/>${wall(L, -1)}${wall(R, 1)}${chim}${ur > 0 ? roofSide(L - 10, top) + roofSide(R + 10, top) : ""}${shingles}${sign}</g>`; };
  // SCUDO: uno scudo verde petrolio con il lucchetto si apre attorno all'azienda (u: 0..1)
  const shield = (o = {}) => { const u = o.u ?? 1, gl = o.glow ?? 0, w = o.w ?? 560, h = o.h ?? 700;
    const d = `M0,${-h * 0.52} C${w * 0.35},${-h * 0.4} ${w * 0.7},${-h * 0.46} ${w},${-h * 0.5} C${w * 1.02},${h * 0.05} ${w * 0.7},${h * 0.36} 0,${h * 0.5} C${-w * 0.7},${h * 0.36} ${-w * 1.02},${h * 0.05} ${-w},${-h * 0.5} C${-w * 0.7},${-h * 0.46} ${-w * 0.35},${-h * 0.4} 0,${-h * 0.52}Z`;
    const hex = Array.from({ length: 70 }, (_, i) => { const c = i % 10, r = Math.floor(i / 10), x = -w + 60 + c * 128 + (r % 2) * 64, y = -h * 0.45 + r * 112;
      return `<path transform="translate(${x},${y})" d="M0,-40 L35,-20 L35,20 L0,40 L-35,20 L-35,-20Z" fill="none" stroke="${P.tealLight}" stroke-width="3" opacity=".22"/>`; }).join("");
    return `<g transform="scale(${0.2 + 0.8 * u})" opacity="${Math.min(1, u * 1.5)}"><defs><clipPath id="shClip"><path d="${d}"/></clipPath></defs>
      <path d="${d}" fill="${P.teal}" opacity="${0.1 + 0.08 * gl}"/><g clip-path="url(#shClip)">${hex}</g>
      <path d="${d}" fill="none" stroke="${P.tealLight}" stroke-width="16" opacity=".45"/><path d="${d}" fill="none" stroke="#e9fffb" stroke-width="5" opacity=".9"/>
      <g transform="translate(0,${-h * 0.52})"><circle r="46" fill="${P.teal}" stroke="${P.ink}" stroke-width="5"/><g transform="translate(0,14) scale(1.2)"><path d="M-12,-8 L-12,-18 Q-12,-30 0,-30 Q12,-30 12,-18 L12,-8" fill="none" stroke="${P.ink}" stroke-width="5"/><rect x="-19" y="-10" width="38" height="30" rx="6" fill="${P.brass}" stroke="${P.ink}" stroke-width="4"/><circle cy="3" r="4" fill="${P.ink}"/></g></g></g>`; };
  // BOLLA DI SAPONE: sfera trasparente con bordo iridescente (wob: ondeggiamento)
  const bubble = (o = {}) => { const u = o.u ?? 1, wob = o.wob ?? 0, rx = (o.rx ?? 720) * (1 + 0.015 * wob), ry = (o.ry ?? 430) * (1 - 0.015 * wob);
    return `<g transform="scale(${u})" opacity="${Math.min(1, u * 1.3)}"><defs>
      <linearGradient id="bbRim" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ff9ad1"/><stop offset=".3" stop-color="#ffe27a"/><stop offset=".55" stop-color="#8ff0d8"/><stop offset=".8" stop-color="#9bb8ff"/><stop offset="1" stop-color="#ff9ad1"/></linearGradient>
      <radialGradient id="bbFill" cx="50%" cy="50%" r="50%"><stop offset=".75" stop-color="#ffffff" stop-opacity=".02"/><stop offset=".97" stop-color="#e8f6ff" stop-opacity=".28"/><stop offset="1" stop-color="#ffffff" stop-opacity=".45"/></radialGradient></defs>
      <ellipse rx="${rx}" ry="${ry}" fill="url(#bbFill)"/><ellipse rx="${rx}" ry="${ry}" fill="none" stroke="url(#bbRim)" stroke-width="10" opacity=".75"/>
      <ellipse rx="${rx - 8}" ry="${ry - 8}" fill="none" stroke="#fff" stroke-width="3" opacity=".6"/>
      <path d="M${-rx * 0.72},${-ry * 0.32} Q${-rx * 0.6},${-ry * 0.78} ${-rx * 0.2},${-ry * 0.9}" fill="none" stroke="#fff" stroke-width="22" stroke-linecap="round" opacity=".6"/>
      <circle cx="${-rx * 0.5}" cy="${-ry * 0.78}" r="12" fill="#fff" opacity=".85"/><path d="M${rx * 0.55},${ry * 0.55} Q${rx * 0.72},${ry * 0.3} ${rx * 0.78},${ry * 0.05}" fill="none" stroke="#fff" stroke-width="10" stroke-linecap="round" opacity=".35"/></g>`; };
  // CUPOLA-BOLLA: mezza bolla di sapone appoggiata sulla scrivania (base piatta), bordo iridescente, riflessi.
  // u: gonfiaggio 0..1, wob: ondeggiamento (-1..1), glow: calore interno, uid: id univoci
  const bubbleDome = (o = {}) => { const u = o.u ?? 1, wob = o.wob ?? 0, rx = (o.rx ?? 680) * u * (1 + 0.03 * wob), ry = (o.ry ?? 780) * u * (1 - 0.03 * wob), gl = o.glow ?? 0, k = o.uid ?? (root.Anim?._sid ?? "bd");
    if (u <= 0.01) return "";
    const arc = `M${-rx},0 C${-rx},${-ry * 0.56} ${-rx * 0.56},${-ry} 0,${-ry} C${rx * 0.56},${-ry} ${rx},${-ry * 0.56} ${rx},0`;
    return `<g opacity="${Math.min(1, u * 1.4)}"><defs>
        <linearGradient id="bdRim-${k}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ff9ad1"/><stop offset=".28" stop-color="#ffe27a"/><stop offset=".52" stop-color="#8ff0d8"/><stop offset=".78" stop-color="#9bb8ff"/><stop offset="1" stop-color="#ff9ad1"/></linearGradient>
        <radialGradient id="bdFill-${k}" cx="50%" cy="100%" r="100%"><stop offset=".6" stop-color="#ffffff" stop-opacity=".02"/><stop offset=".93" stop-color="#eaf7ff" stop-opacity=".22"/><stop offset="1" stop-color="#ffffff" stop-opacity=".38"/></radialGradient>
        <radialGradient id="bdGlow-${k}" cx="50%" cy="90%" r="70%"><stop offset="0" stop-color="#ffcf9a" stop-opacity="${0.4 * gl}"/><stop offset="1" stop-color="#ffcf9a" stop-opacity="0"/></radialGradient></defs>
      <path d="${arc}Z" fill="url(#bdFill-${k})"/><path d="${arc}Z" fill="url(#bdGlow-${k})"/>
      <path d="${arc}" fill="none" stroke="url(#bdRim-${k})" stroke-width="11" stroke-linecap="round" opacity=".8"/>
      <path d="${arc}" fill="none" stroke="#fff" stroke-width="3" opacity=".7" transform="translate(0,4) scale(.985)"/>
      <path d="M${-rx * 0.8},${-ry * 0.28} C${-rx * 0.76},${-ry * 0.66} ${-rx * 0.5},${-ry * 0.9} ${-rx * 0.16},${-ry * 0.96}" fill="none" stroke="#fff" stroke-width="24" stroke-linecap="round" opacity=".55"/>
      <circle cx="${-rx * 0.44}" cy="${-ry * 0.84}" r="13" fill="#fff" opacity=".9"/><circle cx="${-rx * 0.34}" cy="${-ry * 0.89}" r="6" fill="#fff" opacity=".8"/>
      <path d="M${rx * 0.6},${-ry * 0.62} C${rx * 0.74},${-ry * 0.46} ${rx * 0.8},${-ry * 0.3} ${rx * 0.82},${-ry * 0.12}" fill="none" stroke="#fff" stroke-width="10" stroke-linecap="round" opacity=".32"/>
      <ellipse cx="0" cy="2" rx="${rx}" ry="${10 + 6 * u}" fill="none" stroke="#fff" stroke-width="3" opacity=".45"/></g>`; };
  root.Props = { bubbleDome, house, shield, bubble, belljar, checklist, sheet, stamp, tagCard, dome, clientCard, folder, padlock, pen, rulebook, cross, mortarboard, steam, secondHand, speedLines, sparkle };
})(typeof window !== "undefined" ? window : globalThis);
