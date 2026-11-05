// Pannello di prova: Badge + porta + geni con logo, per controllare disegno e conversione con VectorCraft.
const SB = require("./lib");
const { Chars, Chatbots, Props, Scenes, place, shadow } = SB;
const inner = `${Scenes.office({ screen: "copilot" })}
  ${place(Props.door({ open: 0.6, led: "ok" }), 1180, 900)}
  ${shadow(820, 904, 120, 18)}${place(Chars.badge({ mouth: "smile", brow: { curve: 12 }, armR: [170, -300, -0.3, "open"], led: "ok" }), 820, 900, 1.15)}
  ${place(Chatbots.genieLogo("copilot", { approved: true }), 1320, 880, 0.75)}
  ${place(Chatbots.genieLogo("claude", { approved: true }), 420, 880, 0.75)}
  ${place(Chatbots.genieLogo("chatgpt", {}), 1700, 700, 0.55)}${place(Chatbots.genieLogo("perplexity", {}), 160, 640, 0.5)}`;
console.log(SB.save("prova", inner, { scena: "prova", caption: "Copilot, con l'account aziendale: prego, entri pure." }));
