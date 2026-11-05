// Pannelli p07-p12 dello storyboard (episodio "Sono un badge"): nome, rischio, riavvolgi, giusta, finale, cartello.
const SB = require("./lib");
const { Chars, Chatbots, Props, Scenes, P, place, shadow } = SB;
const W = 1920, H = 1080;

// ruota un personaggio attorno al suo centro (origine ai piedi): ang gradi, cy = altezza del perno
const spin = (g, ang, cy = -200) => `<g transform="translate(0,${cy}) rotate(${ang}) translate(0,${-cy})">${g}</g>`;
const dim = (g, op) => `<g opacity="${op}">${g}</g>`;
const wave = [150, -330, -0.3, "open"];   // braccio destro alzato che saluta

// ------------------------------------------------------------------ p07 NOME
{
  const letter = (c, x, y, rot, fill, fg) => `<g transform="translate(${x},${y}) rotate(${rot})"><rect x="-52" y="-78" width="104" height="132" rx="4" fill="${fill}"/><rect x="-52" y="54" width="104" height="5" fill="#000" opacity=".25"/><text y="34" text-anchor="middle" font-family="EB Garamond" font-weight="700" font-size="120" fill="${fg}">${c}</text></g>`;
  const row = (word, y, fill, fg, rots) => [...word].map((c, i) => letter(c, 960 + (i - (word.length - 1) / 2) * 118, y, rots[i % rots.length], fill, fg)).join("");
  const inner = `${Scenes.serverRoom()}
    ${row("NON", 215, P.copper, "#fbf4ea", [-5, 3, -3])}
    ${row("AUTORIZZATO", 430, "#f1ebde", P.ink, [-4, 4, -3, 6, -5, 3, 0, -4, 5, -3, 4])}
    ${shadow(790, 892, 105, 16, "#000", 0.5)}${place(Chatbots.genieLogo("chatgpt", { mouth: "grin", armR: wave }), 790, 890, 0.68)}
    ${shadow(1130, 892, 105, 16, "#000", 0.5)}${place(Chatbots.genieLogo("perplexity", { mouth: "grin", armL: [-150, -330, 0.3, "open"], armR: [190, -190, -0.3, "thumb"] }), 1130, 890, 0.68)}`;
  SB.save("p07", inner, { scena: "nome", caption: "Questo si chiama: strumento non autorizzato." });
}

// ------------------------------------------------------------------ p08 RISCHIO
{
  const card = (x, y, r, s, c, l) => place(Props.clientCard({ c, l, r }), x, y, s);
  const ln = (x, y, len, o = 0.8) => `<path d="M${x},${y} l${len},0" stroke="#fff" stroke-width="8" stroke-linecap="round" opacity="${o}"/>`;
  const inner = `${Scenes.studio(1)}
    <rect x="0" y="880" width="960" height="200" fill="#e0d2b6"/><rect x="960" y="880" width="960" height="200" fill="#16233a"/><path d="M960,880 L950,944 L974,1008 L950,1080" fill="none" stroke="#f6f1e6" stroke-width="10"/>
    ${Scenes.tag(250, 120, "PER L'AZIENDA", { w: 440, h: 74, fs: 34, rot: -2 })}
    ${Scenes.tag(1330, 120, "PER TE", { w: 330, h: 74, fs: 34, rot: 2, bg: "#f1ebde" })}
    <!-- sinistra: la cartella si apre e le schede dei clienti volano via -->
    ${shadow(440, 884, 190, 22)}${place(Props.folder({ open: 0.9 }), 440, 880, 1.5)}
    ${ln(300, 640, 130, 0.35)}${ln(540, 600, 150, 0.35)}${ln(470, 470, 140, 0.35)}${ln(250, 400, 120, 0.35)}${ln(700, 380, 130, 0.35)}
    ${card(330, 620, -14, 1.2, P.teal, "C")}${card(580, 560, 12, 1.0, P.copper, "M")}${card(760, 410, -20, 1.3, "#5d7f78", "R")}
    ${card(480, 410, 18, 1.1, P.brass, "L")}${card(250, 360, -8, 0.95, P.teal, "S")}${card(670, 280, 26, 0.8, P.copper, "D")}
    ${dim(card(880, 250, -15, 0.7, P.teal, "C"), 0.7)}${dim(card(130, 560, 10, 0.8, P.brass, "A"), 0.7)}
    <!-- destra: il Badge preoccupato, il regolamento strappato e il cartellino col suo nome -->
    ${shadow(1450, 884, 120, 18, "#000", 0.55)}${place(Chars.badge({ mouth: "worried", brow: { curve: 22, y: -262 }, eyes: { px: 0.7, py: -0.5, open: 1.1 }, sweat: true, led: "no", name: "IL TUO NOME", armL: [-150, -190, 0.5], armR: [150, -190, -0.5] }), 1450, 880, 1.12)}
    ${shadow(1130, 886, 130, 14, "#000", 0.5)}${place(Props.rulebook({ tear: 1 }), 1130, 880, 1.15)}${place(Props.cross(2.4), 1130, 690)}
    ${place(Props.tagCard("IL TUO NOME", { w: 360, rot: 7 }), 1680, 330)}`;
  SB.save("p08", inner, { scena: "rischio", caption: "Per l'azienda: dati dei clienti fuori controllo. Per te: regole violate. E sul badge c'è il tuo nome." });
}

// ------------------------------------------------------------------ p09 RIAVVOLGI
{
  const lines = [[1700, 120, 420], [1500, 250, 360], [1850, 330, 500], [1400, 560, 460], [1750, 610, 300], [1200, 700, 380], [1820, 760, 520], [1550, 830, 340], [1000, 160, 300], [1300, 440, 260], [1650, 470, 400], [700, 640, 300], [450, 780, 360], [1900, 900, 300]]
    .map(([x, y, len], i) => `<path d="M${x},${y} l${-len},0" stroke="#fff" stroke-width="${i % 3 ? 7 : 11}" stroke-linecap="round" opacity="${0.35 + (i % 3) * 0.15}"/>`).join("");
  const tri = (x) => `<path d="M${x + 130},-110 L${x},0 L${x + 130},110Z" fill="#f6f1e6" stroke="${P.ink}" stroke-width="9" stroke-linejoin="round"/>`;
  const bd = (o) => Chars.badge(o);
  const inner = `<defs><filter id="desat"><feColorMatrix type="saturate" values="0.25"/></filter></defs>
    <g filter="url(#desat)">${Scenes.office({ screen: "chatgpt", warm: 0.4 })}</g>
    <rect width="${W}" height="${H}" fill="#1d2b45" opacity=".42"/>
    <!-- l'orologio gira all'indietro -->
    <g transform="translate(1470,230)"><path d="M-128,-20 A128,128 0 1 1 20,128" fill="none" stroke="#f6f1e6" stroke-width="12" stroke-linecap="round" opacity=".9"/><path d="M-128,-20 l-24,-28 M-128,-20 l30,-18" stroke="#f6f1e6" stroke-width="12" stroke-linecap="round"/></g>
    ${lines}
    <!-- simbolo di riavvolgimento -->
    <g transform="translate(960,300)"><circle r="190" fill="#141f2e" opacity=".5"/><circle r="190" fill="none" stroke="#f6f1e6" stroke-width="8"/>
      <g transform="translate(-130,0)">${tri(-130 + 0)}${tri(0)}</g></g>
    <!-- il Badge viene riavvolto: gira all'indietro, schiacciato dalla velocita' -->
    ${shadow(960, 904, 120, 16, "#0b1526", 0.5)}
    ${place(dim(spin(bd({ mouth: "O", eyes: { open: 1.15 }, armL: [-150, -110, 0.3], armR: [150, -110, -0.3] }), 20), 0.28), 960, 900, 1.0)}
    ${place(spin(bd({ mouth: "O", brow: { curve: 18, y: -270 }, eyes: { open: 1.15, px: -0.6 }, squash: 0.94, armL: [-190, -120, 0.4], armR: [190, -260, -0.4, "open"] }), -10), 960, 900, 1.0)}
    ${[[760, 600, 16], [1170, 560, 12], [1220, 760, 14], [720, 780, 11]].map(([x, y, r]) => `<g transform="translate(${x},${y})">${Props.sparkle(r)}</g>`).join("")}`;
  SB.save("p09", inner, { scena: "riavvolgi", caption: "Riavvolgiamo." });
}

// ------------------------------------------------------------------ p10 GIUSTA
{
  const rows = ["Copilot con l'account aziendale", "Claude solo con licenza a tuo nome", "Tutto il resto: prima chiedi all'IT"];
  const bw = 900, lx = bw / 2 - 70;   // x locale dei loghi nella lavagna
  const ry = (i) => 120 + i * 72;
  const board = Props.checklist({ checks: [1, 1, 1], rows, w: bw, fs: 36 }).replace("TRE COSE IN PIÙ", "TRE REGOLE") +
    Chars.logo("copilot", lx, ry(0), 56) + Chars.logo("claude", lx, ry(1), 56) +
    `<g transform="translate(${lx},${ry(2)})"><rect x="-34" y="-24" width="68" height="48" rx="10" fill="${P.brass}" stroke="${P.ink}" stroke-width="4"/><text y="10" text-anchor="middle" font-family="IBM Plex Mono" font-weight="700" font-size="28" fill="${P.ink}">IT</text></g>`;
  const inner = `${Scenes.office({ screen: "copilot", warm: 1.2 })}
    ${place(board, 1250, 130)}
    ${shadow(770, 904, 120, 17)}${place(Chars.badge({ mouth: "smile", brow: { curve: 12, y: -270 }, eyes: { px: 0.5, py: -0.3 }, led: "ok", armL: [-150, -120, 0.3], armR: [200, -400, -0.2, "point"] }), 770, 900, 1.1)}
    ${shadow(1150, 902, 90, 14)}${place(Chatbots.genieLogo("copilot", { approved: true, mouth: "smile", armR: [180, -300, -0.25, "open"] }), 1150, 898, 0.62)}
    ${shadow(1520, 902, 90, 14)}${place(Chatbots.genieLogo("claude", { approved: true, mouth: "smile", armL: [-190, -300, 0.25, "open"], armR: [190, -190, -0.3, "thumb"] }), 1520, 898, 0.62)}`;
  SB.save("p10", inner, { scena: "giusta", caption: "Tre regole: Copilot aziendale, Claude con licenza tua, e per tutto il resto prima si chiede all'IT." });
}

// ------------------------------------------------------------------ p11 FINALE
{
  const glass = `<clipPath id="winc"><rect x="132" y="92" width="516" height="378"/></clipPath>`;
  const outside = `${glass}<g clip-path="url(#winc)">
      ${place(Chatbots.genieLogo("chatgpt", { mouth: "frown", armL: [-150, -230, 0.3, "open"], armR: [190, -330, -0.3, "open"] }), 262, 545, 0.52)}
      ${place(Chatbots.genieLogo("perplexity", { mouth: "O", armR: [200, -340, -0.3, "open"] }), 522, 545, 0.52)}
      <path d="M150,110 L330,110 L150,330Z M520,470 L648,470 L648,300Z" fill="#fff" opacity=".22"/>
    </g>
    <rect x="382" y="92" width="16" height="516" fill="#a87c4f"/><rect x="132" y="92" width="516" height="516" fill="none" stroke="#a87c4f" stroke-width="8"/>`;
  const inner = `${Scenes.office({ screen: "azienda", warm: 1.2 })}
    ${outside}
    ${place(Props.door({ open: 0, led: "ok" }), 1190, 900, 0.84)}
    ${shadow(700, 904, 95, 14)}${place(Chars.mouse({ mouth: "smile", eyes: { lid: 0.35, px: -0.4 }, sweat: false, armR: [100, -130, -0.3], cupTilt: -10 }), 700, 900, 0.98)}
    ${shadow(1030, 904, 120, 17)}${place(Chars.badge({ mouth: "smile", brow: { curve: 14, y: -270 }, eyes: { open: 0, happy: true }, led: "ok", armL: [-140, -110, 0.3], armR: [140, -110, -0.3] }), 1030, 900, 1.12)}`;
  SB.save("p11", inner, { scena: "finale", caption: "Stesso lavoro, stessa velocità. E io… non devo cacciare nessuno." });
}

// ------------------------------------------------------------------ p12 CARTELLO
{
  const wink = Chars.badge({ mouth: "grin", brow: { curve: 14, y: -270 }, eyes: { px: 0.3, py: -0.2 }, led: "ok", armL: [-140, -110, 0.3], armR: [170, -240, -0.3, "open"] }) +
    `<ellipse cx="36" cy="-222" rx="29" ry="34" fill="${P.paper}"/><path d="M12,-222 Q36,-240 60,-222" fill="none" stroke="${P.ink}" stroke-width="5" stroke-linecap="round"/>`;
  const med = (kind, x, y) => `<circle cx="${x}" cy="${y}" r="74" fill="${P.teal}"/><circle cx="${x}" cy="${y}" r="74" fill="none" stroke="${P.tealLight}" stroke-width="5"/>${Chars.medal(kind, x, y, 50, P.ink)}`;
  const inner = `<defs><radialGradient id="tg" cx="50%" cy="48%" r="55%"><stop offset="0" stop-color="${P.navyMid}"/><stop offset="1" stop-color="${P.ink}"/></radialGradient></defs>
    <rect width="${W}" height="${H}" fill="url(#tg)"/>
    <ellipse cx="960" cy="600" rx="760" ry="150" fill="${P.teal}" opacity=".09"/>
    <text x="960" y="352" text-anchor="middle" font-family="Inter" font-weight="800" font-size="32" letter-spacing="8" fill="${P.brass}">USARE L'IA IN SICUREZZA · LEZIONE 4</text>
    <path d="M760,390 L1160,390" stroke="${P.copper}" stroke-width="6" stroke-linecap="round"/>
    <text x="960" y="560" text-anchor="middle" font-family="EB Garamond" font-weight="700" font-size="140" fill="#fbf4ea">Gli strumenti autorizzati</text>
    ${med("copilot", 880, 740)}${med("claude", 1040, 740)}
    ${shadow(1640, 1044, 150, 20, "#000", 0.5)}${place(wink, 1640, 1040, 0.98)}
    ${[[330, 800, 14], [1500, 220, 12], [260, 260, 10]].map(([x, y, r]) => `<g transform="translate(${x},${y})">${Props.sparkle(r, P.brass)}</g>`).join("")}
    ${Scenes.fx("night")}`;
  SB.save("p12", inner, { scena: "cartello" });
}
console.log("ok");
