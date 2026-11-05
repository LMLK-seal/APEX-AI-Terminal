// Genera cartoon/loghi.js dai loghi ufficiali in loghi/*.svg (usabile sia nel browser sia in Node).
// Ogni logo diventa { vb: "viewBox", g: "contenuto SVG" }, con le classi CSS trasformate in attributi fill
// (VectorCraft e DeckCraft non leggono i <style> interni) e gli id resi unici per logo.
const fs = require("fs"), path = require("path");
const dir = path.join(__dirname, "..", "loghi");
// colore ufficiale per i loghi monocromatici di Simple Icons (path senza fill)
const mono = { claude: "#D97757", anthropic: "#191919", perplexity: "#1FB8CD", googlegemini: "#8E75B2", deepseek: "#4D6BFE" };
const pick = { copilot: "copilot.svg", claude: "claude.svg", anthropic: "anthropic.svg", chatgpt: "chatgpt.svg",
  perplexity: "perplexity.svg", gemini: "googlegemini.svg", deepseek: "deepseek.svg" };
const out = {};
for (const [k, f] of Object.entries(pick)) {
  let s = fs.readFileSync(path.join(dir, f), "utf8").replace(/<\?xml[^>]*>|<!--[\s\S]*?-->|<title>[\s\S]*?<\/title>/g, "");
  const vb = /viewBox=["']([^"']+)["']/.exec(s)[1];
  const styles = {};
  s.replace(/\.([\w-]+)\s*\{\s*fill:\s*([^;]+);?\s*\}/g, (_, c, v) => (styles[c] = v.trim()));
  s = s.replace(/<style>[\s\S]*?<\/style>/g, "").replace(/class="([\w-]+)"/g, (m, c) => (styles[c] ? `fill="${styles[c]}"` : ""));
  s = s.replace(/\bid="([^"]+)"/g, `id="${k}-$1"`).replace(/url\(#([^)]+)\)/g, `url(#${k}-$1)`).replace(/(xlink:)?href="#([^"]+)"/g, `$1href="#${k}-$2"`);
  let g = /<svg[^>]*>([\s\S]*)<\/svg>/.exec(s)[1].trim().replace(/\s+/g, " ");
  const m = mono[f.replace(".svg", "")];
  if (m) g = `<g fill="${m}">${g}</g>`;
  out[k] = { vb, g };
}
const js = `// Generato da tools/make_loghi.js: loghi ufficiali (Simple Icons, Wikimedia Commons). Non modificare a mano.\n` +
  `(function (root) { root.Loghi = ${JSON.stringify(out)}; })(typeof window !== "undefined" ? window : globalThis);\n`;
fs.writeFileSync(path.join(__dirname, "..", "cartoon", "loghi.js"), js);
console.log(Object.entries(out).map(([k, v]) => `${k} ${v.vb} ${v.g.length}b`).join("\n"));
