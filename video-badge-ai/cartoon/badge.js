// Episodio "Sono un badge": il protagonista (Chars.badge) e i geni con il logo ufficiale (Chatbots.genieLogo).
// Stesso linguaggio del primo episodio: contorno 5 px, occhi Chars.eye, bocche Chars.mouth, braccia Chars.arm.
// I loghi non vengono mai deformati né ricolorati: stanno interi su un medaglione bianco sul petto del genio.
(function (root) {
  const { P, eye, mouth, arm } = root.Chars;
  const S = 5; let n = 0; const u = (p) => `${p}${++n}`;
  const AZIENDA = root.AZIENDA ?? "Nodomatic";   // azienda di fantasia: mai il nome di un'azienda vera

  const leg = (x0, y0, x1, y1, dir = 1, shoe = P.navyMid) =>
    `<path d="M${x0},${y0} Q${(x0 + x1) / 2 + 6 * dir},${(y0 + y1) / 2} ${x1},${y1 - 6}" fill="none" stroke="${P.ink}" stroke-width="6.5" stroke-linecap="round"/>
     <path d="M${x1 - 12 * dir},${y1 - 10} Q${x1 + 20 * dir},${y1 - 14} ${x1 + 26 * dir},${y1 - 2} Q${x1 + 26 * dir},${y1 + 3} ${x1 + 14 * dir},${y1 + 3} L${x1 - 14 * dir},${y1 + 3} Q${x1 - 18 * dir},${y1 - 6} ${x1 - 12 * dir},${y1 - 10}Z" fill="${shoe}" stroke="${P.ink}" stroke-width="4" stroke-linejoin="round"/>`;

  // logo ufficiale intero dentro un riquadro size×size centrato in (x,y)
  function logo(kind, x, y, size) {
    const L = root.Loghi[kind]; if (!L) return "";
    return `<svg x="${x - size / 2}" y="${y - size / 2}" width="${size}" height="${size}" viewBox="${L.vb}" preserveAspectRatio="xMidYMid meet" overflow="visible">${L.g}</svg>`;
  }
  // medaglione: disco bianco con bordo d'inchiostro e il logo dentro (r = raggio)
  const medal = (kind, x, y, r = 46, ring = P.ink) =>
    `<circle cx="${x}" cy="${y + 4}" r="${r}" fill="#000" opacity=".18"/><circle cx="${x}" cy="${y}" r="${r}" fill="#fff" stroke="${ring}" stroke-width="${S}"/>${logo(kind, x, y, r * 1.22)}`;

  // ---------------------------------------------------------------- IL BADGE (protagonista)
  // tessera verticale con fascia aziendale, foro e clip, chip dorato e codice a barre. Origine ai piedi.
  // o: { brow:{y,curve,tilt}, eyes, mouth, armL:[hx,hy,bend,hand], armR, squash, strap (lunghezza cordino, 0 = niente),
  //      led: "ok"|"no"|null (luce sul chip), name (testo sotto la fascia) }
  function badge(o = {}) {
    const W = 92, top = -372, bot = -66, sq = o.squash ?? 1, br = o.brow ?? {}, ey = o.eyes ?? {};
    const c = u("bd"), body = `M${-W + 20},${top} L${W - 20},${top} Q${W},${top} ${W},${top + 20} L${W},${bot - 20} Q${W},${bot} ${W - 20},${bot} L${-W + 20},${bot} Q${-W},${bot} ${-W},${bot - 20} L${-W},${top + 20} Q${-W},${top} ${-W + 20},${top}Z`;
    const aL = o.armL ?? [-150, -150, 0.3], aR = o.armR ?? [150, -150, -0.3];
    const by = br.y ?? -268, tl = br.tilt ?? 0, cv = br.curve ?? 6;
    const strap = o.strap ?? 0;
    const lanyard = strap > 0 ? `<path d="M-14,${top - 30} L-30,${top - 30 - strap} M14,${top - 30} L30,${top - 30 - strap}" stroke="${P.teal}" stroke-width="14" stroke-linecap="round"/><path d="M-14,${top - 30} L-30,${top - 30 - strap} M14,${top - 30} L30,${top - 30 - strap}" stroke="${P.ink}" stroke-width="3" opacity=".3"/>` : "";
    const clip = `<rect x="-20" y="${top - 38}" width="40" height="30" rx="7" fill="${P.brass}" stroke="${P.ink}" stroke-width="4"/><rect x="-7" y="${top - 14}" width="14" height="26" rx="4" fill="${P.brass}" stroke="${P.ink}" stroke-width="4"/>`;
    const bars = [4, 2, 6, 3, 2, 5, 2, 4, 3, 6, 2, 3, 5].reduce((a, w, i) => { a.s += i % 2 ? "" : `<rect x="${a.x}" y="${bot - 54}" width="${w}" height="28" fill="${P.ink}"/>`; a.x += w + 3; return a; }, { x: -10, s: "" }).s;
    const led = o.led ? `<circle cx="-56" cy="${bot - 40}" r="9" fill="${o.led === "ok" ? P.tealLight : P.led}" stroke="${P.ink}" stroke-width="3"/><circle cx="-56" cy="${bot - 40}" r="20" fill="${o.led === "ok" ? P.tealLight : P.led}" opacity=".25"/>` : "";
    return `<g class="badge"><g transform="scale(${1 / Math.sqrt(sq)},${sq})">
      ${leg(-34, bot, -42, 0, -1)}${leg(34, bot, 42, 0, 1)}
      ${o.lf ? "" : arm(-W + 4, -190, aL[0], aL[1], aL[2], aL[3])}${o.rf ? "" : arm(W - 4, -190, aR[0], aR[1], aR[2], aR[3])}
      ${lanyard}
      <clipPath id="${c}"><path d="${body}"/></clipPath>
      <path d="${body}" fill="${P.paper}"/>
      <g clip-path="url(#${c})">
        <rect x="${-W}" y="${top}" width="${W * 2}" height="70" fill="${P.copper}"/><rect x="${-W}" y="${top + 58}" width="${W * 2}" height="12" fill="#8f3f1b" opacity=".45"/>
        <path d="M${W - 24},${top + 70} L${W + 10},${top + 70} L${W + 10},${bot} L${W - 16},${bot} Q${W - 6},${(top + bot) / 2} ${W - 24},${top + 70}Z" fill="${P.paperShade}" opacity=".8"/>
        <path d="M${-W + 14},${top + 92} L${-W + 14},${bot - 40}" stroke="#fff" stroke-width="7" stroke-linecap="round" opacity=".7"/>
        <circle cx="-56" cy="-182" r="12" fill="${P.copperLight}" opacity=".35"/><circle cx="56" cy="-182" r="12" fill="${P.copperLight}" opacity=".35"/>
        <rect x="-72" y="${bot - 58}" width="34" height="30" rx="5" fill="${P.brass}" stroke="${P.ink}" stroke-width="3"/><path d="M-72,${bot - 43} L-38,${bot - 43} M-55,${bot - 58} L-55,${bot - 28}" stroke="${P.ink}" stroke-width="2.5"/>
        ${bars}${led}
      </g>
      <rect x="-26" y="${top + 12}" width="52" height="12" rx="6" fill="${P.ink}"/>
      <text x="0" y="${top + 52}" text-anchor="middle" font-family="Inter" font-weight="800" font-size="17" letter-spacing="1" fill="#fbf4ea" textLength="${Math.min(150, AZIENDA.length * 14)}" lengthAdjust="spacingAndGlyphs">${AZIENDA.toUpperCase()}</text>
      ${o.name ? `<text x="0" y="${bot - 70}" text-anchor="middle" font-family="IBM Plex Mono" font-weight="700" font-size="17" fill="${P.ink}">${o.name}</text>` : ""}
      <path d="${body}" fill="none" stroke="${P.ink}" stroke-width="${S}" stroke-linejoin="round"/>
      ${clip}
      <path d="M-56,${by + tl} Q0,${by - cv - 6} 56,${by - tl}" fill="none" stroke="${P.ink}" stroke-width="12" stroke-linecap="round"/>
      <g>${eye(-36, -222, { rx: 24, ry: 29, ...ey })}${eye(36, -222, { rx: 24, ry: 29, ...ey })}</g>
      <g transform="translate(0,-160)">${mouth(o.mouth ?? "rest")}</g>
      ${o.lf ? arm(-W + 4, -190, aL[0], aL[1], aL[2], aL[3]) : ""}${o.rf ? arm(W - 4, -190, aR[0], aR[1], aR[2], aR[3]) : ""}
      ${o.sweat ? `<path d="M${W - 4},-330 q12,18 0,26 q-12,-8 0,-26Z" fill="#9fd3ff" stroke="${P.ink}" stroke-width="3.5"/>` : ""}
    </g></g>`;
  }

  // ---------------------------------------------------------------- GENI CON LOGO
  // approved: true = verde acqua con papillon (in sede), false = giallo "gratis". Il logo va sul medaglione al petto.
  // o: { approved, mouth, eyes, armL, armR }
  function genieLogo(kind, o = {}) {
    const ok = o.approved ?? false;
    const base = ok ? root.Chatbots.genieAssist(o) : root.Chatbots.genieGratis(o);
    const top = ok ? -420 : -430;
    return `<g class="genie-${kind}">${base}${medal(kind, ok ? 0 : -18, top + 226, 44)}</g>`;
  }

  // ---------------------------------------------------------------- PORTA CON LETTORE DI BADGE
  // origine in basso a sinistra; o: { open (0..1), led: "ok"|"no"|null }
  function door(o = {}) {
    const op = o.open ?? 0, w = 330, h = 640;
    const ledC = o.led === "ok" ? P.tealLight : o.led === "no" ? P.led : "#6f7f98";
    return `<g class="door"><rect x="-24" y="${-h - 24}" width="${w + 48}" height="${h + 24}" rx="6" fill="${P.kraftShade}" stroke="${P.ink}" stroke-width="${S}"/>
      <rect x="0" y="${-h}" width="${w}" height="${h}" fill="#3a2a1a"/>
      ${op > 0 ? `<path d="M0,${-h} L${w * (1 - 0.7 * op)},${-h + 30 * op} L${w * (1 - 0.7 * op)},${-30 * op} L0,0Z" fill="#fff4d8" opacity="${0.5 * op}"/>` : ""}
      <g transform="scale(${1 - 0.72 * op},1)"><rect x="0" y="${-h}" width="${w}" height="${h}" fill="${P.kraft}" stroke="${P.ink}" stroke-width="${S}"/>
        <rect x="40" y="${-h + 50}" width="${w - 80}" height="230" rx="6" fill="none" stroke="${P.kraftShade}" stroke-width="8"/><rect x="40" y="${-h + 330}" width="${w - 80}" height="240" rx="6" fill="none" stroke="${P.kraftShade}" stroke-width="8"/>
        <circle cx="${w - 44}" cy="${-h / 2 + 10}" r="16" fill="${P.brass}" stroke="${P.ink}" stroke-width="4"/></g>
      <g transform="translate(${w + 70},${-h / 2 - 40})"><rect x="-32" y="-60" width="64" height="120" rx="12" fill="${P.navyMid}" stroke="${P.ink}" stroke-width="${S}"/>
        <rect x="-20" y="-40" width="40" height="40" rx="6" fill="${P.server}"/><path d="M-10,-28 q10,-8 20,0 M-6,-20 q6,-5 12,0" fill="none" stroke="${P.mouseLight}" stroke-width="3" stroke-linecap="round"/>
        <circle cy="30" r="10" fill="${ledC}" stroke="${P.ink}" stroke-width="3"/>${o.led ? `<circle cy="30" r="24" fill="${ledC}" opacity=".3"/>` : ""}</g></g>`;
  }

  // finestra a parete (per i geni che "entrano dalla finestra"); origine in alto a sinistra; o: { open 0..1 }
  function windowFrame(o = {}) {
    const w = 380, h = 330, op = o.open ?? 0;
    return `<g class="finestra"><rect x="-14" y="-14" width="${w + 28}" height="${h + 28}" rx="6" fill="#f1ebde" stroke="${P.ink}" stroke-width="${S}"/>
      <rect x="0" y="0" width="${w}" height="${h}" fill="#bcd6dd"/><ellipse cx="${w * 0.7}" cy="${h * 0.3}" rx="70" ry="26" fill="#fff" opacity=".7"/>
      <path d="M0,${h * 0.78} Q${w * 0.3},${h * 0.62} ${w * 0.6},${h * 0.74} T${w},${h * 0.7} L${w},${h} L0,${h}Z" fill="#8fb39a"/>
      <g transform="translate(${w / 2},0) scale(${1 - 0.8 * op},1)"><rect x="0" y="0" width="${w / 2}" height="${h}" fill="#fff" opacity=".14" stroke="${P.ink}" stroke-width="${S}"/></g>
      <rect x="0" y="0" width="${w / 2}" height="${h}" fill="#fff" opacity=".1" stroke="${P.ink}" stroke-width="${S}"/>
      <rect x="-30" y="${h}" width="${w + 60}" height="22" rx="4" fill="${P.paperShade}" stroke="${P.ink}" stroke-width="4"/></g>`;
  }

  Object.assign(root.Chars, { badge, logo, medal });
  Object.assign(root.Chatbots, { genieLogo });
  root.Props = Object.assign(root.Props ?? {}, { door, windowFrame });
})(typeof window !== "undefined" ? window : globalThis);
