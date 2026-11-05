// Personaggi del cartone "Sono un contratto" — oggetti antropomorfi, disegnati come SVG parametrici.
// Ogni personaggio ha l'origine ai piedi (0,0) e gruppi con id stabili per l'animazione (brow, eyes, mouth, armL...).
// Linguaggio: contorno d'inchiostro 5px a giunti arrotondati sui personaggi, un tono d'ombra e uno di luce per colore.
(function (root) {
  const P = {
    ink: "#1c2230", paper: "#f6f1e6", paperShade: "#e2d6bf", paperLine: "#b9ab92",
    copper: "#b5562a", copperLight: "#de8a57", navy: "#141f2e", navyMid: "#24354f",
    kraft: "#c99d6c", kraftShade: "#a67b4d", mouth: "#5b2331", tongue: "#d9675e",
    mouse: "#8e9db4", mouseShade: "#6f7f98", mouseLight: "#b4c0d2",
    gratis: "#ffd23f", gratisShade: "#f0b400", gratisPink: "#e0457b",
    teal: "#2f8f83", tealShade: "#226c63", tealLight: "#5fb3a7", brass: "#d8a24a",
    server: "#1d2a44", serverShade: "#131c2f", led: "#ff4d3d", ledAmber: "#ffb43d",
  };
  let uid = 0; const id = (p) => `${p}${++uid}`;
  const S = 5; // spessore contorno

  // ---------------------------------------------------------------- parti condivise
  function eye(cx, cy, o = {}) {
    const rx = o.rx ?? 27, ry = (o.ry ?? 33) * (o.open ?? 1), px = (o.px ?? 0) * rx * 0.38, py = (o.py ?? 0) * ry * 0.35;
    const sx = o.sx ?? 1, lid = o.lid ?? 0;           // lid: palpebra (0 aperta .. 1 chiusa) dall'alto
    if ((o.open ?? 1) < 0.12) return `<path d="M${cx - rx},${cy} Q${cx},${cy + (o.happy ? -16 : 8)} ${cx + rx},${cy}" fill="none" stroke="${P.ink}" stroke-width="${S}" stroke-linecap="round"/>`;
    const cid = id("eye");
    let s = `<g transform="translate(${cx},${cy}) scale(${sx},1)"><clipPath id="${cid}"><ellipse rx="${rx}" ry="${ry}"/></clipPath>
      <ellipse rx="${rx}" ry="${ry}" fill="#fff"/>
      <g clip-path="url(#${cid})"><ellipse cx="${px}" cy="${py}" rx="${rx * 0.5}" ry="${Math.min(ry * 0.56, rx * 0.66)}" fill="${P.ink}"/>
        <circle cx="${px + rx * 0.2}" cy="${py - ry * 0.24}" r="${rx * 0.2}" fill="#fff"/>
        <circle cx="${px - rx * 0.14}" cy="${py + ry * 0.2}" r="${rx * 0.08}" fill="#fff" opacity=".8"/>`;
    if (lid > 0) s += `<rect x="${-rx - 2}" y="${-ry - 2}" width="${rx * 2 + 4}" height="${(ry * 2 + 4) * lid}" fill="${o.lidColor ?? P.paper}"/>
        <line x1="${-rx}" x2="${rx}" y1="${-ry + ry * 2 * lid}" y2="${-ry + ry * 2 * lid}" stroke="${P.ink}" stroke-width="${S}"/>`;
    return s + `</g><ellipse rx="${rx}" ry="${ry}" fill="none" stroke="${P.ink}" stroke-width="${S}"/></g>`;
  }

  // forme della bocca (visemi italiani + espressioni), centrate in (0,0), larghezza ~44
  function mouth(kind, sc = 1) {
    const D = (d, fill = P.mouth) => `<path d="${d}" fill="${fill}" stroke="${P.ink}" stroke-width="${S}" stroke-linejoin="round"/>`;
    const L = (d) => `<path d="${d}" fill="none" stroke="${P.ink}" stroke-width="${S}" stroke-linecap="round"/>`;
    const tongue = (y, w) => `<path d="M${-w},${y} Q0,${y - 9} ${w},${y} Q0,${y + 5} ${-w},${y}Z" fill="${P.tongue}"/>`;
    const m = {
      rest: L("M-14,-2 Q0,7 14,-2"),
      smile: L("M-22,-6 Q0,14 22,-6"),
      A: D("M-22,-8 Q0,-11 22,-8 Q18,20 0,22 Q-18,20 -22,-8Z") + tongue(14, 11),
      E: D("M-21,-5 Q0,-8 21,-5 Q16,9 0,10 Q-16,9 -21,-5Z"),
      I: D("M-19,-4 Q0,-6 19,-4 Q14,5 0,6 Q-14,5 -19,-4Z") + `<path d="M-14,-3 L14,-3" stroke="#fff" stroke-width="4"/>`,
      O: D("M0,-14 Q13,-14 13,2 Q13,17 0,17 Q-13,17 -13,2 Q-13,-14 0,-14Z") + tongue(10, 6),
      U: D("M0,-8 Q8,-8 8,1 Q8,10 0,10 Q-8,10 -8,1 Q-8,-8 0,-8Z"),
      MBP: L("M-15,0 Q0,3 15,0"),
      FV: D("M-17,-3 Q0,-5 17,-3 Q12,7 0,8 Q-12,7 -17,-3Z") + `<rect x="-10" y="-5" width="20" height="6" rx="2" fill="#fff" stroke="${P.ink}" stroke-width="2.5"/>`,
      frown: L("M-18,6 Q0,-8 18,6"),
      flat: L("M-14,1 L14,1"),
      smirk: L("M-14,2 Q4,4 18,-6"),
      worried: L("M-18,4 Q-9,-4 0,3 Q9,9 18,1"),
      grin: D("M-30,-8 Q0,-12 30,-8 Q24,22 0,24 Q-24,22 -30,-8Z", "#fff") + `<path d="M-28,-1 L28,-1 M-15,-9 L-15,16 M0,-10 L0,24 M15,-9 L15,16" stroke="${P.ink}" stroke-width="3"/>`,
      shout: D("M-18,-12 Q0,-16 18,-12 Q16,22 0,26 Q-16,22 -18,-12Z") + tongue(17, 10),
    };
    return `<g transform="scale(${sc})">${m[kind] ?? m.rest}</g>`;
  }

  // braccio "rubber hose": dalla spalla (sx,sy) alla mano (hx,hy) con curvatura bend
  function arm(sx, sy, hx, hy, bend = 0.25, hand = "open") {
    const mx = (sx + hx) / 2, my = (sy + hy) / 2, dx = hx - sx, dy = hy - sy;
    const cx = mx - dy * bend, cy = my + dx * bend;
    let h;
    if (hand === "point") h = `<g transform="translate(${hx},${hy}) rotate(${Math.atan2(dy, dx) * 57.3})"><rect x="-4" y="-10" width="30" height="13" rx="6.5" fill="${P.paper}" stroke="${P.ink}" stroke-width="4"/><circle r="12" fill="${P.paper}" stroke="${P.ink}" stroke-width="4"/></g>`;
    else if (hand === "thumb") h = `<g transform="translate(${hx},${hy})"><rect x="-5" y="-30" width="12" height="26" rx="6" fill="${P.paper}" stroke="${P.ink}" stroke-width="4"/><circle r="13" fill="${P.paper}" stroke="${P.ink}" stroke-width="4"/></g>`;
    else h = `<g transform="translate(${hx},${hy})"><circle r="13" fill="${P.paper}" stroke="${P.ink}" stroke-width="4"/><circle cx="${dx > 0 ? -9 : 9}" cy="-9" r="6.5" fill="${P.paper}" stroke="${P.ink}" stroke-width="3.5"/></g>`;
    return `<path d="M${sx},${sy} Q${cx},${cy} ${hx},${hy}" fill="none" stroke="${P.ink}" stroke-width="6.5" stroke-linecap="round"/>${h}`;
  }
  const leg = (x0, y0, x1, y1, shoe = P.copper, dir = 1) =>
    `<path d="M${x0},${y0} Q${(x0 + x1) / 2 + 6 * dir},${(y0 + y1) / 2} ${x1},${y1 - 6}" fill="none" stroke="${P.ink}" stroke-width="6.5" stroke-linecap="round"/>
     <path d="M${x1 - 12 * dir},${y1 - 10} Q${x1 + 20 * dir},${y1 - 14} ${x1 + 26 * dir},${y1 - 2} Q${x1 + 26 * dir},${y1 + 3} ${x1 + 14 * dir},${y1 + 3} L${x1 - 14 * dir},${y1 + 3} Q${x1 - 18 * dir},${y1 - 6} ${x1 - 12 * dir},${y1 - 10}Z"
       fill="${shoe}" stroke="${P.ink}" stroke-width="4" stroke-linejoin="round"/>
     <path d="M${x1 + 2 * dir},${y1 - 9} Q${x1 + 14 * dir},${y1 - 10} ${x1 + 19 * dir},${y1 - 4}" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" opacity=".55"/>`;

  // ---------------------------------------------------------------- IL CONTRATTO (protagonista)
  // o: { brow:{tilt,curve,y}, eyes:{open,px,py,lid,happy}, mouth, armL:[hx,hy,bend,hand], armR:[...], turn (-1..1), view:'front'|'profile', squash }
  function contratto(o = {}) {
    if (o.view === "profile") return contrattoProfilo(o);
    const t = o.turn ?? 0, W = 100 * (1 - 0.18 * Math.abs(t)), top = -330, bot = -62, dog = 40;
    const sq = o.squash ?? 1, fx = t * 22;                        // il volto scivola verso il lato in cui si gira
    const br = o.brow ?? {}, ey = o.eyes ?? {};
    const cid = id("ct");
    const body = `M${-W + 10},${top} L${W - dog},${top} L${W},${top + dog} L${W},${bot - 12} Q${W},${bot} ${W - 12},${bot} L${-W + 12},${bot} Q${-W},${bot} ${-W},${bot - 12} L${-W},${top + 12} Q${-W},${top} ${-W + 10},${top}Z`;
    const bodyCurved = body.replace(`L${W},${bot - 12}`, `Q${W + 7},${(top + dog + bot) / 2} ${W},${bot - 12}`).replace(`L${-W},${top + 12}`, `Q${-W - 7},${(top + bot) / 2} ${-W},${top + 12}`);
    const edge = t !== 0 ? `<path d="M${t > 0 ? -W : W},${top + 14} L${t > 0 ? -W - 9 : W + 9},${top + 20} L${t > 0 ? -W - 9 : W + 9},${bot - 4} L${t > 0 ? -W : W},${bot - 12}" fill="${P.paperShade}" stroke="${P.ink}" stroke-width="${S}" stroke-linejoin="round"/>` : "";
    const lines = [130, 118, 136, 96].map((w, i) => `<line x1="${-W + 30 + fx * .3}" x2="${-W + 30 + w * (W / 100) + fx * .3}" y1="${-168 + i * 17}" y2="${-168 + i * 17}" stroke="${P.paperLine}" stroke-width="6" stroke-linecap="round"/>`).join("");
    const sig = `<path d="M${W - 82},${-94} q8,-16 14,-2 q6,14 12,-4 q6,-16 12,0 q5,12 14,-6" fill="none" stroke="${P.copper}" stroke-width="4.5" stroke-linecap="round"/><line x1="${W - 88}" x2="${W - 22}" y1="-80" y2="-80" stroke="${P.paperLine}" stroke-width="3"/>`;
    const by = (br.y ?? -296), tl = br.tilt ?? 0, cv = br.curve ?? 6;
    const brow = `<path id="brow" d="M${-58 + fx},${by + tl} Q${fx},${by - cv - 6} ${58 + fx},${by - tl}" fill="none" stroke="${P.ink}" stroke-width="12" stroke-linecap="round"/>`;
    const eyeSx = (side) => (t === 0 ? 1 : (side * t > 0 ? 1 : 0.78));
    const eyes = `<g id="eyes">${eye(-40 + fx, -246, { ...ey, sx: eyeSx(-1) })}${eye(40 + fx, -246, { ...ey, sx: eyeSx(1) })}</g>`;
    const cheeks = `<circle cx="${-60 + fx}" cy="-206" r="12" fill="${P.copperLight}" opacity=".32"/><circle cx="${60 + fx}" cy="-206" r="12" fill="${P.copperLight}" opacity=".32"/>`;
    const aL = o.armL ?? [-150, -120, 0.3], aR = o.armR ?? [150, -120, -0.3];
    return `<g class="contratto"><g transform="scale(${1 / Math.sqrt(sq)},${sq})">
      ${leg(-36, bot, -44, 0, P.copper, -1)}${leg(36, bot, 44, 0, P.copper, 1)}
      ${o.lf ? "" : arm(-W + 4, -165, aL[0], aL[1], aL[2], aL[3])}
      ${o.rf ? "" : arm(W - 4, -165, aR[0], aR[1], aR[2], aR[3])}
      ${edge}
      <clipPath id="${cid}"><path d="${bodyCurved}"/></clipPath>
      <path d="${bodyCurved}" fill="${P.paper}"/>
      <g clip-path="url(#${cid})">
        <path d="M${W - 26},${top} L${W + 20},${top} L${W + 20},${bot} L${W - 18},${bot} Q${W - 10},${(top + bot) / 2} ${W - 26},${top}Z" fill="${P.paperShade}" opacity=".75"/>
        <path d="M${-W + 12},${top + 30} L${-W + 12},${bot - 30}" stroke="#fff" stroke-width="7" stroke-linecap="round" opacity=".7"/>
        ${lines}${sig}${cheeks}
      </g>
      <path d="${bodyCurved}" fill="none" stroke="${P.ink}" stroke-width="${S}" stroke-linejoin="round"/>
      <path d="M${W - dog},${top} L${W - dog + 4},${top + dog - 4} L${W},${top + dog}Z" fill="${P.paperShade}" stroke="${P.ink}" stroke-width="${S}" stroke-linejoin="round"/>
      ${brow}${eyes}
      <g id="mouth" transform="translate(${fx},-196)">${mouth(o.mouth ?? "rest")}</g>
      ${o.lf ? arm(-W + 4, -165, aL[0], aL[1], aL[2], aL[3]) : ""}${o.rf ? arm(W - 4, -165, aR[0], aR[1], aR[2], aR[3]) : ""}
      ${o.sweat ? `<path d="M${W - 6},-318 q12,18 0,26 q-12,-8 0,-26Z" fill="#9fd3ff" stroke="${P.ink}" stroke-width="3.5"/>` : ""}
    </g></g>`;
  }
  // di profilo il foglio e' sottilissimo: la gag classica
  function contrattoProfilo(o = {}) {
    const top = -330, bot = -62, dir = o.dir ?? 1;
    return `<g class="contratto" transform="scale(${dir},1)">
      ${leg(0, bot, 6, 0, P.copper, 1)}${leg(0, bot, -4, 0, P.copper, 1)}
      ${arm(0, -165, 46, -110, -0.3, "open")}
      <rect x="-6" y="${top}" width="12" height="${bot - top}" rx="5" fill="${P.paperShade}" stroke="${P.ink}" stroke-width="${S}"/>
      <path d="M2,-296 L28,-300" stroke="${P.ink}" stroke-width="11" stroke-linecap="round"/>
      <g transform="translate(10,-246)"><path d="M0,-30 Q26,-30 26,0 Q26,30 0,30Z" fill="#fff" stroke="${P.ink}" stroke-width="${S}"/><ellipse cx="13" cy="2" rx="7" ry="12" fill="${P.ink}"/><circle cx="16" cy="-4" r="3.5" fill="#fff"/></g>
      <path d="M6,-196 Q16,-192 18,-200" fill="none" stroke="${P.ink}" stroke-width="${S}" stroke-linecap="round"/>
    </g>`;
  }

  // ---------------------------------------------------------------- IL MOUSE FRETTOLOSO (chi copia e incolla)
  function mouse(o = {}) {
    const ey = o.eyes ?? { open: 1.1, px: 0.3, py: -0.1 };
    const cid = id("ms");
    const body = "M0,-232 C62,-232 82,-180 82,-118 C82,-58 50,-34 0,-34 C-50,-34 -82,-58 -82,-118 C-82,-180 -62,-232 0,-232Z";
    const aL = o.armL ?? [-130, -150, 0.35], aR = o.armR ?? [118, -96, -0.2];
    const cup = o.cup !== false ? `<g transform="translate(${aR[0] + 6},${aR[1] - 22}) rotate(${o.cupTilt ?? 8})"><path d="M-14,-24 L14,-24 L10,16 L-10,16Z" fill="#fff" stroke="${P.ink}" stroke-width="4" stroke-linejoin="round"/><rect x="-16" y="-31" width="32" height="8" rx="3" fill="${P.copper}" stroke="${P.ink}" stroke-width="4"/><path d="M-9,-4 L9,-4" stroke="${P.copper}" stroke-width="5"/><path d="M-2,-38 q-6,-10 2,-18 M8,-40 q-6,-8 2,-14" fill="none" stroke="${P.mouseLight}" stroke-width="3" stroke-linecap="round"/></g>` : "";
    const sweat = o.sweat !== false ? `<path d="M88,-214 q10,16 0,22 q-10,-6 0,-22Z" fill="#9fd3ff" stroke="${P.ink}" stroke-width="3"/><path d="M-96,-182 q8,13 0,18 q-8,-5 0,-18Z" fill="#9fd3ff" stroke="${P.ink}" stroke-width="3"/>` : "";
    return `<g class="mouse">
      <path d="M0,-232 C6,-262 -30,-286 -52,-268 C-74,-250 -58,-216 -86,-206 C-104,-200 -118,-214 -122,-226" fill="none" stroke="${P.ink}" stroke-width="7" stroke-linecap="round"/>
      <rect x="-136" y="-246" width="24" height="16" rx="3" transform="rotate(-30 -124 -238)" fill="${P.mouseLight}" stroke="${P.ink}" stroke-width="4"/>
      ${(() => { const st = o.stride; if (st == null) return leg(-30, -40, -38, 0, "#f4f4f4", -1) + leg(30, -40, 40, 0, "#f4f4f4", 1);
        const a = Math.sin(st) * 22, l1 = Math.max(0, Math.sin(st)) * 16, l2 = Math.max(0, -Math.sin(st)) * 16;   // ciclo di corsa
        return leg(-30, -40, -38 - a, -l1, "#f4f4f4", -1) + leg(30, -40, 40 + a, -l2, "#f4f4f4", 1); })()}
      ${arm(-76, -128, aL[0], aL[1], aL[2], aL[3])}${arm(76, -128, aR[0], aR[1], aR[2], aR[3])}${cup}
      <clipPath id="${cid}"><path d="${body}"/></clipPath>
      <path d="${body}" fill="${P.mouse}"/>
      <g clip-path="url(#${cid})"><path d="M40,-240 C96,-200 96,-70 30,-30 L110,-30 L110,-240Z" fill="${P.mouseShade}" opacity=".8"/>
        <path d="M-58,-196 C-70,-160 -70,-110 -56,-80" stroke="${P.mouseLight}" stroke-width="9" stroke-linecap="round" fill="none" opacity=".8"/></g>
      <path d="${body}" fill="none" stroke="${P.ink}" stroke-width="${S}"/>
      <path d="M0,-232 L0,-150 M-80,-140 Q0,-160 80,-140" fill="none" stroke="${P.ink}" stroke-width="4"/>
      ${eye(-34, -190, { rx: 22, ry: 26, ...ey })}${eye(34, -190, { rx: 22, ry: 26, ...ey })}
      <rect x="-8" y="-216" width="16" height="30" rx="8" fill="#3a4456" stroke="${P.ink}" stroke-width="4"/>
      <path d="M-56,-230 Q-34,${-244 + (o.browUp ?? 0)} -12,-232 M12,-232 Q34,${-244 + (o.browUp ?? 0)} 56,-230" fill="none" stroke="${P.ink}" stroke-width="7" stroke-linecap="round"/>
      <g transform="translate(0,-110)">${mouth(o.mouth ?? "grin", 0.9)}</g>
      ${o.cap ? `<g transform="translate(0,-238) rotate(-8)"><path d="M-64,-8 L0,-36 L64,-8 L0,20Z" fill="${P.ink}"/><rect x="-32" y="0" width="64" height="22" rx="4" fill="${P.ink}"/><path d="M46,-6 L52,40" stroke="${P.brass}" stroke-width="5"/><circle cx="52" cy="44" r="7" fill="${P.brass}"/></g>` : ""}
      ${sweat}
    </g>`;
  }

  // ---------------------------------------------------------------- CHATGRATIS (il chatbot trovato online)
  function chatgratis(o = {}) {
    const ey = o.eyes ?? { lid: 0.42, px: 0.6, lidColor: P.gratis };
    const cid = id("cg");
    const body = "M-120,-300 L120,-300 Q140,-300 140,-280 L140,-130 Q140,-110 120,-110 L-40,-110 L-78,-62 L-70,-110 L-120,-110 Q-140,-110 -140,-130 L-140,-280 Q-140,-300 -120,-300Z";
    const star = (x, y, r, c = P.gratisPink) => `<path transform="translate(${x},${y})" d="M0,${-r} L${r * .28},${-r * .28} L${r},0 L${r * .28},${r * .28} L0,${r} L${-r * .28},${r * .28} L${-r},0 L${-r * .28},${-r * .28}Z" fill="${c}" stroke="${P.ink}" stroke-width="3" stroke-linejoin="round"/>`;
    const aL = o.armL ?? [-210, -250, 0.25], aR = o.armR ?? [205, -255, -0.25, "thumb"];
    return `<g class="chatgratis">
      ${arm(-136, -190, aL[0], aL[1], aL[2], aL[3])}${arm(136, -190, aR[0], aR[1], aR[2], aR[3])}
      <clipPath id="${cid}"><path d="${body}"/></clipPath>
      <path d="${body}" fill="${P.gratis}"/>
      <g clip-path="url(#${cid})"><path d="M90,-310 L150,-310 L150,-100 L110,-100 Q126,-200 90,-310Z" fill="${P.gratisShade}"/>
        <path d="M-120,-282 L60,-282" stroke="#fff6c9" stroke-width="9" stroke-linecap="round"/></g>
      <path d="${body}" fill="none" stroke="${P.ink}" stroke-width="${S}" stroke-linejoin="round"/>
      <path d="M-86,-262 Q-56,-282 -24,-264 M24,-266 Q56,-286 88,-268" fill="none" stroke="${P.ink}" stroke-width="8" stroke-linecap="round"/>
      ${eye(-54, -232, { rx: 24, ry: 26, ...ey })}${eye(54, -232, { rx: 24, ry: 26, ...ey })}
      <g transform="translate(0,-168)">${mouth(o.mouth ?? "grin", 1.35)}</g>
      <g transform="translate(116,-306) rotate(14)"><path d="M0,-34 L9,-14 L32,-18 L18,0 L32,18 L9,14 L0,34 L-9,14 L-32,18 L-18,0 L-32,-18 L-9,-14Z" fill="${P.gratisPink}" stroke="${P.ink}" stroke-width="4" stroke-linejoin="round"/>
        <text y="6" text-anchor="middle" font-family="Inter" font-weight="800" font-size="13" fill="#fff">GRATIS</text></g>
      ${star(-168, -300, 14)}${star(176, -150, 11, "#fff")}${star(-176, -150, 9, "#fff")}
    </g>`;
  }

  // ---------------------------------------------------------------- L'ASSISTENTE AZIENDALE (lo strumento approvato)
  function assistente(o = {}) {
    const ey = o.eyes ?? { open: 0.9, px: 0, py: 0.1 };
    const cid = id("as");
    const body = "M-112,-296 L112,-296 Q132,-296 132,-276 L132,-136 Q132,-116 112,-116 L74,-116 L84,-70 L42,-116 L-112,-116 Q-132,-116 -132,-136 L-132,-276 Q-132,-296 -112,-296Z";
    const aL = o.armL ?? [-178, -128, 0.3], aR = o.armR ?? [184, -238, -0.2, "thumb"];
    return `<g class="assistente">
      ${arm(-128, -186, aL[0], aL[1], aL[2], aL[3])}${arm(128, -186, aR[0], aR[1], aR[2], aR[3])}
      <clipPath id="${cid}"><path d="${body}"/></clipPath>
      <path d="${body}" fill="${P.teal}"/>
      <g clip-path="url(#${cid})"><path d="M84,-306 L140,-306 L140,-100 L100,-100 Q118,-200 84,-306Z" fill="${P.tealShade}"/>
        <path d="M-112,-278 L40,-278" stroke="${P.tealLight}" stroke-width="9" stroke-linecap="round"/></g>
      <path d="${body}" fill="none" stroke="${P.ink}" stroke-width="${S}" stroke-linejoin="round"/>
      <path d="M-76,-258 Q-50,-266 -24,-258 M24,-258 Q50,-266 76,-258" fill="none" stroke="${P.ink}" stroke-width="8" stroke-linecap="round"/>
      ${eye(-48, -226, { rx: 22, ry: 25, lidColor: P.teal, ...ey })}${eye(48, -226, { rx: 22, ry: 25, lidColor: P.teal, ...ey })}
      <g transform="translate(0,-170)">${mouth(o.mouth ?? "smile", 1.1)}</g>
      <g transform="translate(-92,-160)"><path d="M-12,-8 L-12,-18 Q-12,-30 0,-30 Q12,-30 12,-18 L12,-8" fill="none" stroke="${P.ink}" stroke-width="5"/>
        <rect x="-19" y="-10" width="38" height="30" rx="6" fill="${P.brass}" stroke="${P.ink}" stroke-width="4"/><circle cy="3" r="4" fill="${P.ink}"/></g>
    </g>`;
  }

  // ---------------------------------------------------------------- IL SERVER SCONOSCIUTO (fuori dall'azienda)
  function server(o = {}) {
    const cid = id("sv"), glow = id("gl");
    const rows = Array.from({ length: 9 }, (_, i) => `<rect x="-58" y="${-392 + i * 38}" width="116" height="24" rx="4" fill="${P.serverShade}"/>
      <circle cx="${-44 + (i * 37) % 70}" cy="${-380 + i * 38}" r="3.5" fill="${i % 3 ? P.ledAmber : P.led}" opacity="${0.5 + (i % 2) * .5}"/>`).join("");
    return `<g class="server"><defs><filter id="${glow}" x="-1" y="-1" width="3" height="3"><feGaussianBlur stdDeviation="6"/></filter></defs>
      <rect x="-74" y="-420" width="148" height="420" rx="10" fill="${P.server}" stroke="${P.ink}" stroke-width="${S}"/>
      <clipPath id="${cid}"><rect x="-74" y="-420" width="148" height="420" rx="10"/></clipPath>
      <g clip-path="url(#${cid})"><rect x="40" y="-430" width="40" height="440" fill="${P.serverShade}"/></g>
      ${rows}
      <g transform="translate(0,${o.eyeY ?? -300})">
        <ellipse cx="-28" rx="16" ry="${7 * (o.open ?? 1)}" fill="${P.led}" filter="url(#${glow})"/><ellipse cx="28" rx="16" ry="${7 * (o.open ?? 1)}" fill="${P.led}" filter="url(#${glow})"/>
        <ellipse cx="-28" rx="12" ry="${5 * (o.open ?? 1)}" fill="#ffd2c8"/><ellipse cx="28" rx="12" ry="${5 * (o.open ?? 1)}" fill="#ffd2c8"/></g>
      <path d="M-34,-252 Q0,-228 34,-252" fill="none" stroke="${P.led}" stroke-width="5" stroke-linecap="round" opacity=".9"/>
    </g>`;
  }

  root.Chars = { P, contratto, mouse, chatgratis, assistente, server, mouth, eye, arm };
})(typeof window !== "undefined" ? window : globalThis);
