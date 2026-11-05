// Pannelli p01..p06 dello storyboard "Sono un badge" (buongiorno, porta, fretta, porta-sbagliata, nessuno, viaggio).
const SB = require("./lib");
const { Chars, Chatbots, Props, Scenes, P, place, shadow } = SB;
const tag = Scenes.tag;
const mono = { font: "IBM Plex Mono", fw: 700, ls: 2 };
const wash = (op) => `<rect width="1920" height="1080" fill="${P.paper}" opacity="${op}"/>`;   // schiarisce lo sfondo per far leggere i cartellini
const genie = (kind, ok, x, y, s, o = {}, extra = "") => place(Chatbots.genieLogo(kind, { approved: ok, ...o }), x, y, s, extra);
const speed = (x, y, n, len, c = P.navy, op = 0.45) => Array.from({ length: n }, (_, i) => `<path d="M${x},${y - i * 38} l${len - (i % 2) * 40},0" stroke="${c}" stroke-width="8" stroke-linecap="round" opacity="${op}"/>`).join("");

// p01 - buongiorno
SB.save("p01", `${Scenes.office({ screen: "copilot" })}${wash(0.12)}
  ${tag(1500, 44, "09:00", { ...mono, w: 300, h: 76, fs: 50, rot: 2 })}
  ${shadow(960, 896, 150, 22)}
  ${place(Chars.badge({ mouth: "A", brow: { curve: 14 }, eyes: { px: 0.1 }, armR: [190, -410, -0.3, "open"], armL: [-150, -120, 0.3, "open"], led: "ok" }), 960, 890, 1.2)}`,
  { scena: "buongiorno", caption: "Buongiorno. Sono il badge di Nodomatic. Qui dentro entra solo chi è autorizzato." });

// p02 - porta
SB.save("p02", `${Scenes.office({ screen: "claude" })}${wash(0.1)}
  ${place(Props.door({ open: 0.6, led: "ok" }), 1330, 900, 0.85)}
  ${shadow(1450, 904, 110, 15)}${genie("copilot", true, 1450, 880, 0.68, { mouth: "grin" }, "rotate(-5)")}
  ${place(Props.sparkle(16, "#fff"), 1345, 600)}${place(Props.sparkle(10, P.brass), 1310, 660)}
  ${shadow(540, 900, 110, 16)}${genie("claude", true, 540, 890, 0.8)}
  ${tag(300, 430, "LICENZA: A TUO NOME", { w: 480, h: 64, fs: 25, rot: -3 })}
  ${shadow(1020, 900, 150, 22)}${place(Chars.badge({ mouth: "I", brow: { curve: 10 }, eyes: { px: -0.3 }, armL: [-170, -430, -0.2, "point"], armR: [210, -230, -0.25, "open"], led: "ok" }), 1010, 892, 1.1)}`,
  { scena: "porta", caption: "Claude: solo con chi ha la licenza. Il nome sul badge deve essere il tuo." });

// p03 - fretta
SB.save("p03", `${Scenes.office({ screen: "chatgpt" })}${wash(0.1)}
  ${tag(1500, 44, "09:15", { ...mono, w: 300, h: 76, fs: 50, rot: 2 })}
  ${place(Props.windowFrame({ open: 1 }), 1020, 310)}
  ${genie("perplexity", false, 1330, 700, 0.5, { mouth: "grin" }, "rotate(8)")}
  ${shadow(1190, 740, 90, 12)}${genie("chatgpt", false, 1190, 810, 0.68, { mouth: "grin" }, "rotate(-8)")}
  ${speed(960, 860, 3, 130)}
  ${shadow(770, 900, 120, 18)}${place(Chars.mouse({ mouth: "shout", sweat: true, stride: 1.1, eyes: { open: 1.2, px: -0.5 } }), 770, 892, 1.15, "scale(-1,1)")}`,
  { scena: "fretta", caption: "E dalla finestra entra un amico famoso: ChatGPT. Poi Perplexity." });

// p04 - porta sbagliata
SB.save("p04", `${Scenes.office({ screen: "claude" })}${wash(0.1)}
  ${place(Props.door({ open: 0, led: "no" }), 1400, 900, 0.7)}
  ${tag(1385, 520, "PORTA SBAGLIATA", { w: 330, h: 60, fs: 24, rot: -5, bg: "#ffd8d0" })}
  ${shadow(1130, 900, 110, 16)}${genie("claude", false, 1130, 890, 0.8, { mouth: "grin" })}
  ${tag(930, 410, "ACCOUNT PERSONALE", { w: 400, h: 60, fs: 24, rot: 3, bg: "#fff0a8" })}
  ${shadow(640, 900, 150, 22)}${place(Chars.badge({ mouth: "smirk", brow: { tilt: 16, curve: 4 }, eyes: { lid: 0.4, px: 0.7 }, armL: [-150, -130, 0.3, "open"], armR: [150, -150, -0.3, "open"], led: "no" }), 640, 892, 1.15)}
  ${place(Props.cross(1.5), 1752, 668)}`,
  { scena: "porta-sbagliata", caption: "Stesso nome, porta sbagliata." });

// p05 - nessuno
SB.save("p05", `${Scenes.office({ screen: "azienda" })}${wash(0.28)}
  ${tag(170, 150, "È SOLO UN RIASSUNTO", { w: 560, h: 72, fs: 30, rot: -4 })}
  ${tag(1250, 420, "LO USANO TUTTI", { w: 470, h: 72, fs: 30, rot: 3 })}
  ${tag(1270, 700, "CHI CONTROLLA?", { w: 470, h: 72, fs: 30, rot: -3 })}
  ${shadow(960, 896, 160, 24)}
  ${place(Chars.badge({ mouth: "flat", brow: { tilt: 10, curve: 2 }, eyes: { lid: 0.5 }, armL: [-270, -230, 0.15, "open"], armR: [270, -230, -0.15, "open"], led: "ok" }), 960, 890, 1.25)}`,
  { scena: "nessuno", caption: "Tanto è solo un riassunto. Tanto lo usano tutti. Tanto… chi controlla?" });

// p06 - viaggio (la mano del genio è in (210,-360)*scala; il file pende da lì)
const gs = 0.85, gx = 1000, gy = 830, hx = gx + 210 * gs, hy = gy - 360 * gs;
SB.save("p06", `${Scenes.network(1.2)}
  ${speed(gx - 330, gy - 230, 4, 220, "#8fb3e8", 0.5)}
  ${genie("chatgpt", false, gx, gy, gs, { mouth: "grin", armR: [210, -360, -0.3, "open"] }, "rotate(-6)")}
  ${place(Chars.contratto({ mouth: "worried", brow: { tilt: -6 }, eyes: { px: -0.5 }, armL: [-140, -140, 0.3, "open"], armR: [140, -150, -0.3, "open"] }), hx + 10, hy + 330 * 0.6 + 14, 0.6, "rotate(10)")}
  ${tag(1380, 160, "NESSUN BADGE", { w: 400, h: 64, fs: 26, rot: 3 })}
  ${tag(1420, 270, "NESSUN REGISTRO", { w: 430, h: 64, fs: 26, rot: -2 })}
  ${place(Props.cross(2.2), 1340, 192)}${place(Props.cross(2.2), 1385, 302)}`,
  { scena: "viaggio", caption: "Il file esce. Senza badge, senza registro. E non torna indietro." });
