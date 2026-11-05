// Base comune dei pannelli dello storyboard: carica i moduli del cartone in Node e scrive un pannello 1920×1080.
// Uso in uno script di pannelli:  const SB = require("./lib"); SB.save("p01", svgInterno, { scena, caption });
const fs = require("fs"), path = require("path");
const C = path.join(__dirname, "..", "cartoon");
for (const f of ["characters", "chatbots", "props", "scenes", "loghi", "badge"]) require(path.join(C, f + ".js"));
const { Chars, Chatbots, Props, Scenes, Loghi } = globalThis;
const P = Chars.P, W = 1920, H = 1080;

const place = (g, x, y, s = 1, extra = "") => `<g transform="translate(${x},${y}) scale(${s}) ${extra}">${g}</g>`;
const shadow = (x, y, rx, ry = rx * 0.16, c = "#6b4a2a", op = 0.35) => `<ellipse cx="${x}" cy="${y}" rx="${rx}" ry="${ry}" fill="${c}" opacity="${op}"/>`;
const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;");
// sottotitolo come nel cartone: Inter 600, crema su pillola scura in basso al centro (va a capo oltre ~58 caratteri)
function caption(text) {
  const words = esc(text).split(" "), lines = [""];
  for (const w of words) (lines[lines.length - 1] + " " + w).trim().length > 58 ? lines.push(w) : (lines[lines.length - 1] = (lines[lines.length - 1] + " " + w).trim());
  const lw = Math.max(...lines.map((l) => l.length)) * 20 + 60, lh = 50, h = lines.length * lh + 22, y0 = H - 50 - h;
  return `<rect x="${(W - lw) / 2}" y="${y0}" width="${lw}" height="${h}" rx="12" fill="#101828" opacity=".66"/>` +
    lines.map((l, i) => `<text x="${W / 2}" y="${y0 + 48 + i * lh}" text-anchor="middle" font-family="Inter" font-weight="600" font-size="38" fill="#fbf6ec">${l}</text>`).join("");
}
// numero del pannello in alto a sinistra (solo storyboard, non va nel video)
const panelNo = (n, scena) => `<g transform="translate(40,36)"><rect width="${120 + scena.length * 15}" height="50" rx="6" fill="${P.ink}" opacity=".85"/><text x="18" y="34" font-family="IBM Plex Mono" font-weight="700" font-size="26" fill="#fbf4ea">${n} · ${esc(scena.toUpperCase())}</text></g>`;

function save(id, inner, o = {}) {
  const dir = path.join(__dirname, "panels"); fs.mkdirSync(dir, { recursive: true });
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">` +
    `<rect width="${W}" height="${H}" fill="${P.paper}"/>${inner}${o.caption ? caption(o.caption) : ""}${o.scena ? panelNo(id.replace(/^p/, ""), o.scena) : ""}</svg>`;
  fs.writeFileSync(path.join(dir, id + ".svg"), svg);
  return path.join(dir, id + ".svg");
}
module.exports = { Chars, Chatbots, Props, Scenes, Loghi, P, W, H, place, shadow, caption, save, esc };
