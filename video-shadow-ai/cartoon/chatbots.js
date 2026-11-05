// Tre direzioni di design per i chatbot (ChatGratis e Assistente aziendale), senza la forma squadrata.
// Stesso linguaggio dei personaggi: contorno d'inchiostro, un tono d'ombra e uno di luce. Origine (0,0) in basso.
(function (root) {
  const { P, eye, mouth, arm } = root.Chars;
  const S = 5; let n = 0; const u = (p) => `${p}${++n}`;
  const star = (x, y, r, c) => `<path transform="translate(${x},${y})" d="M0,${-r} L${r * .28},${-r * .28} L${r},0 L${r * .28},${r * .28} L0,${r} L${-r * .28},${r * .28} L${-r},0 L${-r * .28},${-r * .28}Z" fill="${c}" stroke="${P.ink}" stroke-width="3" stroke-linejoin="round"/>`;
  const badge = (x, y, rot = 14) => `<g transform="translate(${x},${y}) rotate(${rot})"><path d="M0,-34 L9,-14 L32,-18 L18,0 L32,18 L9,14 L0,34 L-9,14 L-32,18 L-18,0 L-32,-18 L-9,-14Z" fill="${P.gratisPink}" stroke="${P.ink}" stroke-width="4" stroke-linejoin="round"/><text y="5" text-anchor="middle" font-family="Inter" font-weight="800" font-size="12.5" fill="#fff">GRATIS</text></g>`;
  const padlock = (x, y, s = 1) => `<g transform="translate(${x},${y}) scale(${s})"><path d="M-12,-8 L-12,-18 Q-12,-30 0,-30 Q12,-30 12,-18 L12,-8" fill="none" stroke="${P.ink}" stroke-width="5"/><rect x="-19" y="-10" width="38" height="30" rx="6" fill="${P.brass}" stroke="${P.ink}" stroke-width="4"/><circle cy="3" r="4" fill="${P.ink}"/></g>`;
  // corpo con ombra e luce ritagliate sulla sagoma
  const body = (d, fill, shade, light, shadeD, lightD) => { const c = u("cb");
    return `<clipPath id="${c}"><path d="${d}"/></clipPath><path d="${d}" fill="${fill}"/><g clip-path="url(#${c})"><path d="${shadeD}" fill="${shade}"/><path d="${lightD}" fill="none" stroke="${light}" stroke-width="10" stroke-linecap="round"/></g><path d="${d}" fill="none" stroke="${P.ink}" stroke-width="${S}" stroke-linejoin="round"/>`; };

  // ---------------------------------------------------------------- 1) NUVOLETTA: balloon da fumetto morbido
  function cloudPath(cx, cy, rx, ry, k = 9, bulge = 0.22, tail = null) {
    let d = "";
    for (let i = 0; i <= k; i++) {
      const a = (i / k) * Math.PI * 2 - Math.PI / 2, x = cx + rx * Math.cos(a), y = cy + ry * Math.sin(a);
      if (i === 0) { d += `M${x.toFixed(1)},${y.toFixed(1)}`; continue; }
      const am = ((i - 0.5) / k) * Math.PI * 2 - Math.PI / 2, f = 1 + bulge * (i % 2 ? 1 : 0.7);
      d += ` Q${(cx + rx * f * Math.cos(am)).toFixed(1)},${(cy + ry * f * Math.sin(am)).toFixed(1)} ${x.toFixed(1)},${y.toFixed(1)}`;
    }
    return d + "Z";
  }
  function cloudGratis(o = {}) {
    const d = cloudPath(0, -215, 150, 98, 9, 0.24);
    const tail = `M-70,-138 Q-96,-92 -118,-72 Q-70,-80 -40,-128Z`;
    return `<g>${arm(-150, -200, ...(o.armL ?? [-232, -270, 0.25]))}${arm(150, -200, ...(o.armR ?? [228, -276, -0.25, "thumb"]))}
      <path d="${tail}" fill="${P.gratis}" stroke="${P.ink}" stroke-width="${S}" stroke-linejoin="round"/>
      ${body(d, P.gratis, P.gratisShade, "#fff6c9", "M70,-330 Q200,-240 120,-100 L220,-100 L220,-330Z", "M-110,-260 Q-70,-300 -10,-306")}
      <path d="M-86,-262 Q-60,-282 -26,-266 M26,-268 Q58,-288 90,-268" fill="none" stroke="${P.ink}" stroke-width="8" stroke-linecap="round"/>
      ${eye(-46, -232, { rx: 24, ry: 26, lid: 0.42, px: 0.6, lidColor: P.gratis, ...(o.eyes ?? {}) })}${eye(46, -232, { rx: 24, ry: 26, lid: 0.42, px: 0.6, lidColor: P.gratis, ...(o.eyes ?? {}) })}
      <g transform="translate(0,-176)">${mouth(o.mouth ?? "grin", 1.25)}</g>${badge(132, -300)}${star(-176, -310, 13, P.gratisPink)}${star(184, -170, 10, "#fff")}</g>`;
  }
  function cloudAssist(o = {}) {
    const d = "M0,-312 C86,-312 140,-270 140,-212 C140,-156 92,-118 30,-112 L70,-70 L-2,-112 C-80,-116 -140,-150 -140,-212 C-140,-270 -86,-312 0,-312Z";
    return `<g>${arm(-136, -205, ...(o.armL ?? [-196, -132, 0.3]))}${arm(136, -205, ...(o.armR ?? [204, -262, -0.2, "thumb"]))}
      ${body(d, P.teal, P.tealShade, P.tealLight, "M70,-330 Q170,-230 90,-100 L200,-100 L200,-330Z", "M-104,-250 Q-64,-292 -6,-298")}
      <path d="M-80,-262 Q-52,-272 -26,-262 M26,-262 Q52,-272 80,-262" fill="none" stroke="${P.ink}" stroke-width="8" stroke-linecap="round"/>
      ${eye(-46, -228, { rx: 22, ry: 25, lidColor: P.teal, open: 0.9, ...(o.eyes ?? {}) })}${eye(46, -228, { rx: 22, ry: 25, lidColor: P.teal, open: 0.9, ...(o.eyes ?? {}) })}
      <g transform="translate(0,-172)">${mouth(o.mouth ?? "smile", 1.1)}</g>${padlock(-96, -150)}</g>`;
  }

  // ---------------------------------------------------------------- 2) GENIO DIGITALE: fumo che esce dal computer
  // pezzi separati (testa, busto, coda di fumo, sbuffi) con contorno unico: prima tutti i contorni spessi, poi i riempimenti
  function genie(top, col, shade, light, extra = {}) {
    const head = `M0,${top} C46,${top} 78,${top + 30} 78,${top + 72} C78,${top + 114} 46,${top + 142} 0,${top + 142} C-46,${top + 142} -78,${top + 114} -78,${top + 72} C-78,${top + 30} -46,${top} 0,${top}Z`;
    const torso = `M-96,${top + 160} C-124,${top + 214} -82,${top + 262} -44,${top + 284} L44,${top + 284} C82,${top + 262} 124,${top + 214} 96,${top + 160} C70,${top + 124} -70,${top + 124} -96,${top + 160}Z`;
    const tail = extra.tidy
      ? `M-46,${top + 270} C-64,${top + 330} -6,${top + 350} -20,-120 C-34,-70 -36,-26 0,-14 C40,-2 52,-48 20,-56 C0,-62 -4,-40 12,-36 C26,-70 40,-110 26,-160 C20,${top + 330} 64,${top + 320} 46,${top + 270}Z`
      : `M-46,${top + 270} C-66,${top + 332} -6,${top + 352} -26,-96 C-40,-46 6,-32 0,0 C34,-24 12,-70 30,-112 C52,-166 40,${top + 332} 46,${top + 270}Z`;
    const puffs = extra.tidy ? [[-40, top + 304, 22], [38, top + 322, 18]] : [[-46, top + 300, 26], [38, top + 330, 22], [-26, -136, 17], [24, -66, 13]];
    const circ = (x, y, r) => `M${x - r},${y} a${r},${r} 0 1,0 ${2 * r},0 a${r},${r} 0 1,0 ${-2 * r},0Z`;
    const pieces = [tail, ...puffs.map(([x, y, r]) => circ(x, y, r)), torso, head];
    const outline = pieces.map((d) => `<path d="${d}" fill="none" stroke="${P.ink}" stroke-width="${2 * S}" stroke-linejoin="round"/>`).join("");
    const c = u("gn");
    const fills = pieces.map((d) => `<path d="${d}" fill="${col}"/>`).join("");
    const shadeAll = `<clipPath id="${c}">${pieces.map((d) => `<path d="${d}"/>`).join("")}</clipPath>
      <g clip-path="url(#${c})"><path d="M30,${top - 10} C110,${top + 60} 90,${top + 150} 60,${top + 170} C130,${top + 210} 60,${top + 300} 30,${top + 320} C60,-200 40,-40 10,10 L200,10 L200,${top - 10}Z" fill="${shade}"/>
        <path d="M-58,${top + 40} Q-40,${top + 14} -8,${top + 10}" fill="none" stroke="${light}" stroke-width="10" stroke-linecap="round"/>
        <path d="M-84,${top + 180} Q-74,${top + 160} -50,${top + 152}" fill="none" stroke="${light}" stroke-width="8" stroke-linecap="round"/></g>`;
    // linee interne: separazione testa/busto
    const inner = `<path d="M-60,${top + 136} Q0,${top + 156} 60,${top + 136}" fill="none" stroke="${P.ink}" stroke-width="4" opacity=".55"/>`;
    return outline + fills + shadeAll + inner;
  }
  const pixels = (c, top, seed = 1) => Array.from({ length: 6 }, (_, i) => { const x = (i % 2 ? 1 : -1) * (54 + (i * 13) % 30), y = -40 - i * 34, s = 10 + (i % 3) * 4;
    return `<rect x="${x}" y="${y}" width="${s}" height="${s}" rx="2" fill="${c}" stroke="${P.ink}" stroke-width="2.5" opacity="${0.6 + 0.4 * ((i + seed) % 2)}" transform="rotate(${i * 17} ${x} ${y})"/>`; }).join("");
  function genieGratis(o = {}) {
    const top = -430;
    return `<g>${pixels(P.gratis, top, 1)}
      ${arm(-100, top + 172, ...(o.armL ?? [-206, top + 80, 0.3]))}${arm(100, top + 172, ...(o.armR ?? [210, top + 70, -0.3, "thumb"]))}
      ${genie(top, P.gratis, P.gratisShade, "#fff6c9")}
      <path d="M-62,${top + 46} Q-40,${top + 26} -12,${top + 42} M12,${top + 40} Q42,${top + 22} 66,${top + 44}" fill="none" stroke="${P.ink}" stroke-width="8" stroke-linecap="round"/>
      ${eye(-34, top + 72, { rx: 22, ry: 25, lid: 0.4, px: 0.7, lidColor: P.gratis, ...(o.eyes ?? {}) })}${eye(34, top + 72, { rx: 22, ry: 25, lid: 0.4, px: 0.7, lidColor: P.gratis, ...(o.eyes ?? {}) })}
      <g transform="translate(0,${top + 114})">${mouth(o.mouth ?? "grin", 1.05)}</g>${badge(52, top + 216, -10)}
      ${star(-150, top + 20, 13, P.gratisPink)}${star(160, top + 4, 10, "#fff")}</g>`;
  }
  function genieAssist(o = {}) {
    const top = -420;
    return `<g>${arm(-100, top + 172, ...(o.armL ?? [-150, top + 236, 0.45]))}${arm(100, top + 172, ...(o.armR ?? [192, top + 76, -0.25, "thumb"]))}
      ${genie(top, P.teal, P.tealShade, P.tealLight, { tidy: true })}
      <path d="M-58,${top + 50} Q-36,${top + 40} -12,${top + 50} M12,${top + 50} Q36,${top + 40} 58,${top + 50}" fill="none" stroke="${P.ink}" stroke-width="8" stroke-linecap="round"/>
      ${eye(-34, top + 76, { rx: 20, ry: 23, lidColor: P.teal, open: 0.9, ...(o.eyes ?? {}) })}${eye(34, top + 76, { rx: 20, ry: 23, lidColor: P.teal, open: 0.9, ...(o.eyes ?? {}) })}
      <g transform="translate(0,${top + 116})">${mouth(o.mouth ?? "smile", 1.0)}</g>
      <path d="M-28,${top + 140} L0,${top + 152} L28,${top + 140} L28,${top + 168} L0,${top + 156} L-28,${top + 168}Z" fill="${P.brass}" stroke="${P.ink}" stroke-width="4" stroke-linejoin="round"/>
      ${padlock(0, top + 236, 0.95)}</g>`;
  }

  // ---------------------------------------------------------------- 3) CAPSULA CON ANTENNA: robottino a pillola
  function capsuleGratis(o = {}) {
    const d = "M-150,-200 C-150,-262 -110,-292 -60,-292 L60,-292 C110,-292 150,-262 150,-200 C150,-138 110,-108 60,-108 L-40,-108 L-86,-66 L-74,-114 C-122,-126 -150,-160 -150,-200Z";
    return `<g><path d="M0,-292 L0,-344" stroke="${P.ink}" stroke-width="6"/>${star(0, -356, 18, P.gratisPink)}
      ${arm(-146, -196, ...(o.armL ?? [-232, -272, 0.25]))}${arm(146, -196, ...(o.armR ?? [228, -276, -0.25, "thumb"]))}
      ${body(d, P.gratis, P.gratisShade, "#fff6c9", "M90,-300 Q170,-200 96,-100 L200,-100 L200,-300Z", "M-112,-246 Q-90,-276 -40,-280")}
      ${[-26, 0, 26].map((x, i) => `<circle cx="${x}" cy="-268" r="6" fill="${P.ink}" opacity="${0.4 + 0.3 * i}"/>`).join("")}
      ${eye(-48, -214, { rx: 24, ry: 26, lid: 0.42, px: 0.6, lidColor: P.gratis, ...(o.eyes ?? {}) })}${eye(48, -214, { rx: 24, ry: 26, lid: 0.42, px: 0.6, lidColor: P.gratis, ...(o.eyes ?? {}) })}
      <g transform="translate(0,-160)">${mouth(o.mouth ?? "grin", 1.15)}</g>${badge(140, -278, 16)}${star(-188, -290, 12, "#fff")}</g>`;
  }
  function capsuleAssist(o = {}) {
    const d = "M-120,-220 C-120,-286 -72,-316 0,-316 C72,-316 120,-286 120,-220 L120,-168 C120,-130 90,-110 50,-110 L84,-70 L20,-110 L-50,-110 C-90,-110 -120,-130 -120,-168Z";
    return `<g><path d="M0,-316 L0,-360" stroke="${P.ink}" stroke-width="6"/>${padlock(0, -362, 0.62)}
      ${arm(-116, -206, ...(o.armL ?? [-176, -132, 0.3]))}${arm(116, -206, ...(o.armR ?? [184, -262, -0.2, "thumb"]))}
      ${body(d, P.teal, P.tealShade, P.tealLight, "M66,-330 Q150,-220 76,-100 L200,-100 L200,-330Z", "M-90,-262 Q-60,-296 -8,-300")}
      <path d="M-74,-262 Q-48,-272 -24,-262 M24,-262 Q48,-272 74,-262" fill="none" stroke="${P.ink}" stroke-width="8" stroke-linecap="round"/>
      ${eye(-42, -226, { rx: 22, ry: 25, lidColor: P.teal, open: 0.9, ...(o.eyes ?? {}) })}${eye(42, -226, { rx: 22, ry: 25, lidColor: P.teal, open: 0.9, ...(o.eyes ?? {}) })}
      <g transform="translate(0,-170)">${mouth(o.mouth ?? "smile", 1.1)}</g></g>`;
  }
  root.Chatbots = { cloudGratis, cloudAssist, genieGratis, genieAssist, capsuleGratis, capsuleAssist };
})(typeof window !== "undefined" ? window : globalThis);
