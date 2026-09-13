// Serve only the public UI, plus fixed loopback NCA API operations.
"use strict";
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const ROOT = path.resolve(__dirname, '..'), WEB = path.join(ROOT, 'web');
const PORT = Number(process.env.PORT) || 8776;
const lab = require('./native-lab.cjs').createLab({root: ROOT, port: PORT});
const types = {'.html':'text/html; charset=utf-8','.mjs':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.gif':'image/gif','.png':'image/png','.json':'application/json'};
const server = http.createServer(async (req,res) => {
  let url, pathname;
  try { url = new URL(req.url, 'http://localhost'); pathname = decodeURIComponent(url.pathname); }
  catch { res.writeHead(400).end('bad request'); return; }
  if (await lab.handle(req,res,url)) return;
  if (!['GET','HEAD'].includes(req.method)) { res.writeHead(405).end(); return; }
  if (pathname.endsWith('/')) pathname += 'index.html';
  const file = path.resolve(WEB, '.' + pathname);
  if (!file.startsWith(WEB + path.sep)) { res.writeHead(403).end(); return; }
  fs.realpath(file, (error, resolved) => {
    if (error || !resolved.startsWith(WEB + path.sep)) { res.writeHead(404).end('not found'); return; }
    fs.readFile(resolved, (error, body) => {
      if (error) { res.writeHead(404).end('not found'); return; }
      res.writeHead(200, {'content-type':types[path.extname(file)] || 'application/octet-stream','cache-control':'no-store'});
      res.end(req.method === 'HEAD' ? undefined : body);
    });
  });
});
for (const signal of ['SIGINT','SIGTERM']) process.on(signal,()=>{lab.close();server.close();process.exit(0);});
process.on('exit',()=>lab.close());
server.on('error',error=>{console.error(error.message);process.exitCode=1;});
server.on('listening',()=>console.log(`native GPU lab: http://127.0.0.1:${PORT}/`));
server.listen(PORT,'127.0.0.1');
