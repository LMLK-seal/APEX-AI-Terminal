// Client MCP che usa Chrome DevTools MCP per "guardare" una pagina/frame prima dell'export.
// Uso: node devtools_check.mjs <url> <out.png> [js-expression]
import { Client } from "/opt/tools/node_modules/@modelcontextprotocol/sdk/dist/esm/client/index.js";
import { StdioClientTransport } from "/opt/tools/node_modules/@modelcontextprotocol/sdk/dist/esm/client/stdio.js";
const [url, out, expr] = process.argv.slice(2);
const transport = new StdioClientTransport({
  command: "node",
  args: ["/opt/tools/node_modules/chrome-devtools-mcp/build/src/bin/chrome-devtools-mcp.js", "--headless", "--isolated", "--no-usage-statistics", "--no-performance-crux",
         "--executablePath", "/opt/pw-browsers/chromium", "--viewport", process.env.VIEWPORT || "1920x1080",
         "--chromeArg=--no-sandbox", "--chromeArg=--use-angle=swiftshader", "--chromeArg=--enable-unsafe-swiftshader"],
});
const c = new Client({ name: "shadow-ai-check", version: "1.0.0" });
await c.connect(transport);
const tools = (await c.listTools()).tools.map(t => t.name);
console.log("tools:", tools.length, tools.slice(0, 30).join(", "));
const call = async (name, args) => { const r = await c.callTool({ name, arguments: args }); return r.content?.map(x => x.text ?? `[${x.type}]`).join("\n"); };
console.log(await call("new_page", { url }));
const pages = await call("list_pages", {}); console.log(pages);
const ids = [...pages.matchAll(/(\d+):/g)].map(m => +m[1]); const pageId = ids[ids.length - 1];
if (expr) console.log(await call("evaluate_script", { pageId, function: `() => (${expr})` }));
console.log((await call("list_console_messages", { pageId }))?.slice(0, 800));
console.log(await call("take_screenshot", { pageId, filePath: out, format: "png", fullPage: process.env.FULLPAGE === "1" }));
await c.close();
