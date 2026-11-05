// Sfondi e composizioni delle scene (1920x1080). Sfondi SENZA contorni (dipinti a forme e luce),
// personaggi CON contorno: separazione classica tra fondale e animazione. Grana e vignettatura su tutto.
(function (root) {
  const { P, contratto, mouse, chatgratis, assistente, server } = root.Chars;
  const W = 1920, H = 1080;
  let n = 0; const u = (p) => `${p}${++n}`;

  const grain = (op = 0.09) => { const f = u("gr"); return `<filter id="${f}"><feTurbulence type="fractalNoise" baseFrequency=".85" numOctaves="2" seed="4"/><feColorMatrix values="0 0 0 0 .2  0 0 0 0 .16  0 0 0 0 .1  0 0 0 1.4 -.35"/></filter><rect width="${W}" height="${H}" filter="url(#${f})" opacity="${op}" style="mix-blend-mode:multiply"/>`; };
  const vignette = (c = "#2a1a0c", op = 0.45) => { const g = u("vg"); return `<radialGradient id="${g}" cx="50%" cy="46%" r="75%"><stop offset="55%" stop-color="${c}" stop-opacity="0"/><stop offset="100%" stop-color="${c}" stop-opacity="${op}"/></radialGradient><rect width="${W}" height="${H}" fill="url(#${g})"/>`; };
  const lg = (id, stops, x2 = 0, y2 = 1) => `<linearGradient id="${id}" x1="0" y1="0" x2="${x2}" y2="${y2}">${stops.map(([o, c, a = 1]) => `<stop offset="${o}" stop-color="${c}" stop-opacity="${a}"/>`).join("")}</linearGradient>`;
  const shadow = (x, y, rx, ry = rx * 0.16, c = "#6b4a2a", op = .35) => `<ellipse cx="${x}" cy="${y}" rx="${rx}" ry="${ry}" fill="${c}" opacity="${op}"/>`;
  const place = (g, x, y, s = 1, extra = "") => `<g transform="translate(${x},${y}) scale(${s}) ${extra}">${g}</g>`;
  const tag = (x, y, text, o = {}) => `<g transform="translate(${x},${y}) rotate(${o.rot ?? -2})"><rect x="0" y="0" width="${o.w ?? 420}" height="${o.h ?? 64}" rx="4" fill="${o.bg ?? "#f1ebde"}"/>
      <rect x="0" y="${(o.h ?? 64)}" width="${o.w ?? 420}" height="4" fill="#000" opacity=".18"/>
      <text x="${(o.w ?? 420) / 2}" y="${(o.h ?? 64) / 2 + (o.fs ?? 26) * .36}" text-anchor="middle" font-family="${o.font ?? "Inter"}" font-weight="${o.fw ?? 800}" font-size="${o.fs ?? 26}" letter-spacing="${o.ls ?? 3}" fill="${o.fg ?? P.ink}">${text}</text></g>`;
  const caption = (text) => `<g transform="translate(${W / 2},${H - 92})"><rect x="${-text.length * 10.5 - 26}" y="-40" width="${text.length * 21 + 52}" height="58" rx="10" fill="#101828" opacity=".62"/><text y="0" text-anchor="middle" font-family="Inter" font-weight="600" font-size="38" fill="#fbf6ec">${text}</text></g>`;
  const hud = `<g transform="translate(84,70) rotate(-1.5)"><rect width="560" height="52" rx="3" fill="#16233a"/><text x="280" y="34" text-anchor="middle" font-family="Inter" font-weight="800" font-size="22" letter-spacing="3" fill="#f1ebde">USARE L'IA IN SICUREZZA · LEZIONE 3</text></g>`;

  // schermo del portatile con uno strumento vero (episodio "Sono un badge"): logo ufficiale + nome.
  // Copilot e Claude hanno la fascia verde acqua (autorizzati), gli altri quella gialla "gratis".
  const APPS = { copilot: ["Microsoft Copilot", 1], claude: ["Claude", 1], chatgpt: ["ChatGPT", 0], perplexity: ["Perplexity", 0] };
  function appScreen(kind) {
    const a = APPS[kind]; if (!a || !root.Loghi) return "";
    const L = root.Loghi[kind], ok = a[1];
    return `<rect x="14" y="18" width="402" height="44" fill="${ok ? P.teal : P.gratis}"/><text x="215" y="48" text-anchor="middle" font-family="Inter" font-weight="800" font-size="20" fill="${ok ? "#fff" : P.ink}">${a[0]}</text>
      <rect x="14" y="62" width="402" height="184" fill="${ok ? "#e8f1ee" : "#fff8dc"}"/>
      <svg x="175" y="80" width="80" height="80" viewBox="${L.vb}" preserveAspectRatio="xMidYMid meet">${L.g}</svg>
      <rect x="60" y="182" width="310" height="40" rx="20" fill="#fff" stroke="${ok ? P.tealLight : P.gratisShade}" stroke-width="3"/>
      <text x="84" y="208" font-family="Inter" font-size="17" fill="#9b9384">Chiedi qualcosa…</text>`;
  }

  // ---------------------------------------------------------------- ufficio (casa): luce del mattino
  function office(o = {}) {
    const warm = o.warm ?? 1;
    let s = `<defs>${lg("wall", [[0, "#efd6ae"], [1, "#d9b27f"]])}${lg("sky", [[0, "#fbefd5"], [1, "#cfe1e6"]])}${lg("desk", [[0, "#d4a875"], [1, "#b98a57"]])}${lg("deskF", [[0, "#9c7044"], [1, "#7d5733"]])}
      ${lg("beam", [[0, "#fff4d8", .55], [1, "#fff4d8", 0]], 1, 1)}${lg("scr", [[0, "#fff8cf"], [1, "#ffe27a"]])}</defs>
      <rect width="${W}" height="${H}" fill="url(#wall)"/>
      <!-- finestra con veneziane -->
      <rect x="110" y="70" width="560" height="560" rx="6" fill="#a87c4f"/><rect x="132" y="92" width="516" height="516" fill="url(#sky)"/>
      <path d="M132,520 L200,470 L260,500 L330,430 L400,480 L470,440 L540,490 L648,450 L648,608 L132,608Z" fill="#d9cdb4" opacity=".8"/>
      <path d="M132,560 L240,530 L340,550 L460,520 L560,545 L648,530 L648,608 L132,608Z" fill="#c9bb9e" opacity=".9"/>
      ${Array.from({ length: 11 }, (_, i) => `<rect x="132" y="${100 + i * 46}" width="516" height="10" fill="#e9d3ad" opacity=".9"/>`).join("")}
      <rect x="382" y="92" width="16" height="516" fill="#a87c4f"/><rect x="100" y="622" width="580" height="22" rx="4" fill="#8e6640"/>
      <!-- fasci di luce dalla finestra sul muro e sulla scrivania -->
      <path d="M648,92 L1180,560 L1180,760 L648,608Z" fill="url(#beam)" opacity="${.55 * warm}"/>
      <path d="M300,740 L760,740 L1380,900 L560,900Z" fill="#fff1cf" opacity="${.22 * warm}"/>
      <!-- orologio a muro: 9:15 -->
      <g transform="translate(1470,230)"><circle r="104" fill="#8e6640"/><circle r="90" fill="#f7ecd6"/>
        ${Array.from({ length: 12 }, (_, i) => `<rect x="-3" y="-82" width="6" height="${i % 3 ? 10 : 18}" rx="2" fill="#5d4630" transform="rotate(${i * 30})"/>`).join("")}
        <rect x="-5" y="-48" width="10" height="56" rx="5" fill="${P.navy}" transform="rotate(${(9.25 / 12) * 360})"/><rect x="-3.5" y="-72" width="7" height="80" rx="3.5" fill="${P.navy}" transform="rotate(90)"/>
        <circle r="9" fill="${P.copper}"/></g>
      <!-- mensola con faldoni -->
      <rect x="1560" y="430" width="330" height="18" rx="4" fill="#8e6640"/>
      ${[["#7c8fa6", 64], ["#b5562a", 56], ["#c9a06a", 70], ["#5d7f78", 52], ["#9aa7b8", 62]].map(([c, w], i, a) => { const x = 1576 + a.slice(0, i).reduce((p, q) => p + q[1] + 4, 0); return `<rect x="${x}" y="${296 + (i % 2) * 10}" width="${w}" height="${134 - (i % 2) * 10}" rx="4" fill="${c}"/><rect x="${x + w / 2 - 14}" y="${330}" width="28" height="40" rx="3" fill="#f3ead8" opacity=".85"/>`; }).join("")}
      <!-- scrivania in prospettiva -->
      <path d="M0,900 L1920,900 L1920,1080 L0,1080Z" fill="url(#deskF)"/>
      <path d="M-40,740 L1960,740 L1920,900 L0,900Z" fill="url(#desk)"/>
      <path d="M0,900 L1920,900 L1920,922 L0,922Z" fill="#6e4b2b"/>
      ${Array.from({ length: 6 }, (_, i) => `<path d="M${-40 + i * 8},${758 + i * 24} Q960,${750 + i * 24} ${1960 - i * 8},${760 + i * 24}" fill="none" stroke="#a97c4c" stroke-width="3" opacity=".3"/>`).join("")}
      <!-- portatile: lo schermo mostra la pubblicita' di ChatGratis -->
      <g transform="translate(250,470)">
        <path d="M0,0 L430,0 Q446,0 446,16 L446,262 L-16,262 L-16,16 Q-16,0 0,0Z" fill="#3b4658"/>
        <rect x="14" y="18" width="402" height="228" rx="6" fill="${o.screen === "azienda" ? "#e8f1ee" : "url(#scr)"}"/>
        ${appScreen(o.screen) || (o.screen === "azienda"
          ? `<rect x="14" y="18" width="402" height="44" fill="${P.teal}"/><text x="34" y="48" font-family="Inter" font-weight="800" font-size="20" fill="#fff">Assistente aziendale</text><rect x="300" y="28" width="104" height="24" rx="12" fill="${P.brass}"/><text x="352" y="45" text-anchor="middle" font-family="Inter" font-weight="800" font-size="13" fill="${P.ink}">IN SEDE</text>
             <rect x="40" y="86" width="230" height="34" rx="14" fill="#d3e5e0"/><rect x="150" y="134" width="240" height="70" rx="14" fill="#cfe0f0"/>`
          : `<text x="215" y="96" text-anchor="middle" font-family="Inter" font-weight="800" font-size="44" fill="${P.ink}">ChatGratis</text><text x="215" y="132" text-anchor="middle" font-family="Inter" font-weight="800" font-size="22" fill="${P.gratisPink}">✦ 100% GRATIS ✦</text>
             <rect x="70" y="160" width="290" height="44" rx="22" fill="#fff" opacity=".9"/><text x="96" y="189" font-family="Inter" font-size="18" fill="#9b9384">Incolla qui il tuo testo…</text>`)}
        <path d="M-60,262 L506,262 L546,300 L-100,300Z" fill="#56637a"/><path d="M-100,300 L546,300 L546,312 L-100,312Z" fill="#2e3747"/>
        <path d="M120,276 L326,276 L330,290 L116,290Z" fill="#46526a"/>
      </g>
      <ellipse cx="473" cy="790" rx="420" ry="34" fill="${o.screen === "azienda" ? P.teal : "#ffd23f"}" opacity=".12"/>
      <!-- tazza e vapore, pianta, portapenne -->
      <g transform="translate(1600,642)"><path d="M-46,0 L46,0 L38,110 Q36,124 22,124 L-22,124 Q-36,124 -38,110Z" fill="${P.copper}"/><path d="M46,24 Q84,26 80,62 Q76,92 40,92" fill="none" stroke="${P.copper}" stroke-width="16"/>
        <path d="M14,0 L46,0 L38,110 Q36,124 22,124 L10,124Z" fill="#8f3f1b" opacity=".5"/><ellipse cy="2" rx="46" ry="10" fill="#5b2a14"/>
        <path d="M-14,-20 q-18,-30 4,-56 q18,-24 -2,-52 M14,-24 q-14,-26 4,-48" fill="none" stroke="#fff" stroke-width="7" stroke-linecap="round" opacity=".45"/></g>
      <g transform="translate(90,604)"><path d="M-40,60 L40,60 L32,140 L-32,140Z" fill="#b5562a"/><path d="M0,60 C-10,10 -60,0 -70,-40 C-30,-30 -10,0 0,60 C6,0 20,-60 60,-70 C50,-20 20,10 0,60" fill="#5d7f68"/><path d="M0,60 C10,20 40,10 70,0 C50,40 20,50 0,60" fill="#6f957a"/></g>
      <g transform="translate(1800,684)"><rect x="-34" y="0" width="68" height="100" rx="8" fill="#5d7f78"/><rect x="-24" y="-60" width="8" height="70" rx="3" fill="${P.copper}" transform="rotate(-8)"/><rect x="2" y="-74" width="8" height="84" rx="3" fill="#f1ebde" transform="rotate(6)"/><rect x="16" y="-50" width="8" height="60" rx="3" fill="${P.navyMid}" transform="rotate(12)"/></g>
      ${o.fx === false ? "" : grain(0.1) + vignette("#3a2312", .42)}`;
    return s;
  }

  // ---------------------------------------------------------------- fuori: sala server sconosciuta
  // file di armadi server in prospettiva; opens[i] = apertura degli occhi del server i (0 chiusi .. 1 aperti)
  function racks(opens = [], lean = 0) {
    const vx = 960, vy = 470; let out = "";
    const depths = [0.3, 0.42, 0.58, 0.8, 1.1, 1.5]; let k = 0;
    for (const side of [-1, 1]) depths.forEach((d, i) => {
      const x = vx + side * (120 + d * 640), y = vy + 40 + d * 420, fog = Math.max(0, 1 - d), op = opens[k++] ?? 0.7;
      out += `<g opacity="${0.35 + d * 0.65}">${place(server({ open: op }), x, y, d * 1.25, `rotate(${-side * lean * d})`)}</g>`;
      out += `<rect x="${x - 100 * d * 1.25}" y="${y - 430 * d * 1.25}" width="${200 * d * 1.25}" height="${440 * d * 1.25}" fill="#0b1526" opacity="${fog * 0.55}"/>`;
    });
    return out;
  }
  function serverRoom(o = {}) {
    const vx = 960, vy = 470;
    const racksSvg = o.servers === false ? "" : racks();
    return `<defs>${lg("nbg", [[0, "#0a1322"], [1, "#16243b"]])}${lg("floor", [[0, "#13203a"], [1, "#0a1222"]])}
        <radialGradient id="redglow" cx="50%" cy="50%" r="50%"><stop offset="0" stop-color="#ff4d3d" stop-opacity=".55"/><stop offset="1" stop-color="#ff4d3d" stop-opacity="0"/></radialGradient>
        ${lg("cone", [[0, "#cfe0ff", .35], [1, "#cfe0ff", 0]])}</defs>
      <rect width="${W}" height="${H}" fill="url(#nbg)"/>
      <ellipse cx="${vx}" cy="${vy - 40}" rx="420" ry="300" fill="url(#redglow)"/>
      <path d="M0,${vy + 60} L${W},${vy + 60} L${W},${H} L0,${H}Z" fill="url(#floor)"/>
      ${Array.from({ length: 9 }, (_, i) => { const x = -400 + i * 340; return `<path d="M${vx + (i - 4) * 14},${vy + 60} L${x},${H}" stroke="#2a3b5c" stroke-width="${3 + Math.abs(i - 4)}" opacity=".7"/>`; }).join("")}
      ${Array.from({ length: 6 }, (_, i) => `<path d="M0,${vy + 80 + i * i * 16} L${W},${vy + 80 + i * i * 16}" stroke="#1d2b45" stroke-width="2" opacity=".8"/>`).join("")}
      ${racksSvg}
      ${o.cone === false ? "" : '<path d="M800,0 L1120,0 L1260,1000 L660,1000Z" fill="url(#cone)"/>'}
      <ellipse cx="960" cy="890" rx="260" ry="44" fill="#cfe0ff" opacity=".16"/>
      ${o.fx === false ? "" : grain(0.14) + vignette("#02060d", .7)}`;
  }

  // ---------------------------------------------------------------- composizioni delle 3 immagini chiave
  const frames = {
    // 1) 09:15 — il contratto saluta, il Mouse frettoloso arriva col caffe'
    buongiorno: () => `${office()}
      ${shadow(1000, 866, 140, 20)}${place(contratto({ armL: [-150, -120, 0.3], armR: [150, -270, -0.3, "open"], mouth: "smile", brow: { y: -304, curve: 12 }, eyes: { px: 0.15 } }), 1000, 864, 1.22)}
      ${shadow(1430, 872, 110, 16)}${place(mouse({ armL: [-140, -190, 0.4], armR: [124, -110, -0.2], mouth: "grin", eyes: { open: 1.15, px: -0.6, py: -0.1 }, browUp: -8 }), 1430, 870, 0.95, "rotate(-6)")}
      <path d="M1580,690 l90,-6 M1590,740 l110,-4 M1600,790 l80,-2" stroke="#fff" stroke-width="7" stroke-linecap="round" opacity=".6"/>
      ${hud}${tag(1600, 70, "09:15", { w: 250, h: 92, fs: 64, font: "IBM Plex Mono", ls: 0, rot: 2 })}${caption("Buongiorno. Sono un contratto.")}`,
    // 2) fuori — sala server sconosciuta: Shadow AI
    fuori: () => `${serverRoom()}
      ${shadow(960, 892, 110, 18, "#000", .5)}${place(contratto({ mouth: "worried", brow: { curve: 22, y: -302 }, eyes: { px: 0.7, py: -0.6, open: 1.1 }, sweat: true, armL: [-70, -130, 0.7], armR: [70, -130, -0.7] }), 960, 890, 0.95)}
      <line x1="1250" y1="300" x2="1250" y2="560" stroke="#8796b3" stroke-width="3"/>${tag(1110, 556, "PAESE: ?", { w: 280, h: 80, fs: 40, rot: 6, fg: P.ink })}
      ${["S", "H", "A", "D", "O", "W", " ", "A", "I"].map((c, i) => c === " " ? "" : `<g transform="translate(${470 + i * 118},${220}) rotate(${[-6, 4, -3, 7, -5, 3, 0, -4, 6][i]})"><rect x="-52" y="-78" width="104" height="132" rx="4" fill="${i > 6 ? P.copper : "#f1ebde"}"/><rect x="-52" y="54" width="104" height="5" fill="#000" opacity=".25"/><text y="34" text-anchor="middle" font-family="EB Garamond" font-weight="700" font-size="120" fill="${i > 6 ? "#fbf4ea" : P.ink}">${c}</text></g>`).join("")}
      ${hud}${caption("Questa si chiama Shadow AI.")}`,
    // 3) la versione giusta — cupola aziendale, assistente approvato, tre timbri
    giusta: () => `${office({ screen: "azienda", warm: 1.2 })}
      <defs><radialGradient id="dome" cx="50%" cy="100%" r="70%"><stop offset=".7" stop-color="#fffaf0" stop-opacity=".05"/><stop offset=".97" stop-color="#fffaf0" stop-opacity=".32"/><stop offset="1" stop-color="#fffaf0" stop-opacity=".55"/></radialGradient></defs>
      <path d="M200,900 C200,330 1720,330 1720,900Z" fill="url(#dome)"/><path d="M200,900 C200,330 1720,330 1720,900" fill="none" stroke="#fffaf0" stroke-width="5" opacity=".65"/>
      <path d="M330,700 C380,500 600,420 840,402" fill="none" stroke="#fff" stroke-width="12" stroke-linecap="round" opacity=".3"/>
      ${shadow(720, 872, 160, 20)}${place(assistente({ mouth: "smile", armR: [190, -250, -0.2, "thumb"] }), 720, 868, 0.95)}
      ${shadow(1130, 872, 110, 16)}${place(contratto({ mouth: "A", brow: { y: -306, curve: 16 }, eyes: { open: 0, happy: true }, armL: [-160, -270, 0.2], armR: [160, -270, -0.2] }), 1130, 836, 1.12)}
      ${shadow(1560, 846, 80, 12, "#6b4a2a", .25)}${place(mouse({ mouth: "smile", eyes: { lid: 0.35, px: -0.4 }, sweat: false, armR: [100, -130, -0.3], cupTilt: -10 }), 1560, 844, 0.62)}
      ${[["STRUMENTO APPROVATO", 250, -3], ["REGOLE CHIARE", 790, 2], ["PERSONE FORMATE", 1250, -2]].map(([t, x, r]) => `<g transform="translate(${x},950) rotate(${r})"><rect width="430" height="64" rx="4" fill="#f1ebde"/><rect y="64" width="430" height="4" fill="#000" opacity=".2"/><text x="22" y="42" font-family="Inter" font-weight="800" font-size="22" letter-spacing="2" fill="${P.ink}">${t}</text><g transform="translate(386,32) rotate(-10)"><rect x="-26" y="-22" width="52" height="44" rx="8" fill="none" stroke="${P.copper}" stroke-width="5"/><path d="M-14,0 L-3,11 L15,-11" fill="none" stroke="${P.copper}" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/></g></g>`).join("")}
      ${hud}<g transform="translate(1560,64) rotate(-2)"><rect width="290" height="118" rx="4" fill="#f1ebde"/><text x="24" y="38" font-family="Inter" font-weight="800" font-size="20" letter-spacing="3" fill="#9a3f17">STESSO TEMPO</text><text x="24" y="98" font-family="IBM Plex Mono" font-weight="700" font-size="58" fill="${P.ink}">00:30</text><path d="M214,74 L232,92 L262,56" fill="none" stroke="#9a3f17" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/></g>`,
  };
  const fx = (kind = "warm") => kind === "night" ? grain(0.14) + vignette("#02060d", .7) : kind === "cool" ? grain(0.12) + vignette("#0b1526", .5) : grain(0.1) + vignette("#3a2312", .42);
  // la rete: cavi in tre piani di parallasse, pacchetti di dati, nuvole e bandierine "?" (disegnata a ogni fotogramma)
  function network(t, o = {}) {
    const sp = o.speed ?? 900, off = (k) => -((t * sp * k) % 3840);
    const cable = (y, amp, w, c, k, ph) => { let d = ""; for (let x = -200; x <= 4200; x += 120) { const yy = y + amp * Math.sin((x / 520) + ph); d += (x === -200 ? "M" : "L") + x + "," + yy.toFixed(1); }
      return `<g transform="translate(${off(k)},0)"><path d="${d}" fill="none" stroke="#0a1120" stroke-width="${w + 10}" stroke-linecap="round" opacity=".5"/><path d="${d}" fill="none" stroke="${c}" stroke-width="${w}" stroke-linecap="round"/><path d="${d}" fill="none" stroke="#fff" stroke-width="${w * 0.18}" stroke-linecap="round" opacity=".18" transform="translate(0,${-w * 0.25})"/></g>`; };
    const packets = (y, k, c) => Array.from({ length: 10 }, (_, i) => { const x = ((i * 410 + t * sp * k * 1.6) % 2400) - 200; return `<g transform="translate(${x},${y + 30 * Math.sin(i + t * 3)})"><rect x="-26" y="-17" width="52" height="34" rx="5" fill="${c}" stroke="${P.ink}" stroke-width="3.5"/><path d="M-26,-17 L0,4 L26,-17" fill="none" stroke="${P.ink}" stroke-width="3"/></g>`; }).join("");
    const clouds = Array.from({ length: 5 }, (_, i) => { const x = ((i * 620 - t * sp * 0.15) % 3100 + 3100) % 3100 - 400; return `<g transform="translate(${x},${150 + (i % 3) * 80})" opacity=".35"><ellipse rx="120" ry="44" fill="#2b3e63"/><ellipse cx="-60" cy="-20" rx="70" ry="44" fill="#2b3e63"/><ellipse cx="50" cy="-30" rx="80" ry="52" fill="#2b3e63"/></g>`; }).join("");
    const flags = Array.from({ length: 4 }, (_, i) => { const x = ((i * 700 - t * sp * 0.4) % 2800 + 2800) % 2800 - 300; return `<g transform="translate(${x},${330 + (i % 2) * 40})"><line y1="0" y2="-120" stroke="#8796b3" stroke-width="5"/><path d="M0,-120 L80,-104 L0,-86Z" fill="${["#b5562a", "#5fb3a7", "#d8a24a", "#e0457b"][i]}"/><text x="26" y="-96" font-family="Inter" font-weight="800" font-size="20" fill="#fff">?</text></g>`; }).join("");
    return `<rect width="${W}" height="${H}" fill="#0d1729"/><radialGradient id="netglow" cx="50%" cy="40%" r="60%"><stop offset="0" stop-color="#22345a"/><stop offset="1" stop-color="#0d1729"/></radialGradient><rect width="${W}" height="${H}" fill="url(#netglow)"/>
      ${clouds}${flags}${cable(470, 60, 26, "#34507f", 0.35, 0)}${packets(430, 0.35, "#5d7fb0")}${cable(640, 40, 54, P.copper, 0.7, 1.3)}${cable(880, 70, 90, "#2f8f83", 1.2, 2.1)}${packets(830, 1.2, "#f1ebde")}`;
  }
  // studio della spiegazione (scena del rischio): carta chiara a sinistra, blu a destra, divisi da uno strappo
  function studio(split = 0) {
    const x = 960 + (1 - split) * 960;
    const tear = Array.from({ length: 18 }, (_, i) => `L${x + (i % 2 ? 14 : -10)},${i * 64}`).join(" ");
    return `<rect width="${W}" height="${H}" fill="#efe3cc"/><rect x="${x}" y="0" width="${W}" height="${H}" fill="#1d2b45"/>
      <path d="M${x},0 ${tear} L${x},${H}" fill="none" stroke="#f6f1e6" stroke-width="10" opacity="${split}"/>`;
  }
  root.Scenes = { office, serverRoom, racks, network, studio, frames, fx, tag, W, H };
})(typeof window !== "undefined" ? window : globalThis);
