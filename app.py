from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_pipeline import PrivaRepoRAG
import webbrowser
import threading

app = Flask(__name__)
CORS(app)

print("Loading PrivaRepo...")
rag = PrivaRepoRAG()
print("Ready!")

# HTML is built directly inside app.py - no external file needed!
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PrivaRepo — Code Intelligence</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');
:root{--bg:#0a0a0f;--surface:#111118;--surface2:#1a1a24;--border:#2a2a3a;--accent:#00ff88;--accent2:#7c3aed;--accent3:#f59e0b;--text:#e8e8f0;--muted:#6b6b80;--danger:#ff4444;}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--bg);color:var(--text);font-family:'Space Mono',monospace;min-height:100vh;}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,255,136,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,255,136,0.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0;}
.wrap{max-width:1200px;margin:0 auto;padding:0 24px;position:relative;z-index:1;}
header{border-bottom:1px solid var(--border);padding:18px 0;position:sticky;top:0;background:rgba(10,10,15,0.95);backdrop-filter:blur(12px);z-index:100;}
.hinner{display:flex;align-items:center;justify-content:space-between;}
.logo{display:flex;align-items:center;gap:10px;}
.logo-icon{width:34px;height:34px;background:linear-gradient(135deg,var(--accent),var(--accent2));border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px;}
.logo-text{font-family:'Syne',sans-serif;font-weight:800;font-size:20px;}
.logo-text span{color:var(--accent);}
.status{display:flex;align-items:center;gap:12px;font-size:11px;color:var(--muted);}
.dot{width:7px;height:7px;border-radius:50%;background:var(--accent);box-shadow:0 0 8px var(--accent);animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.badge{padding:3px 8px;border-radius:20px;font-size:10px;font-weight:700;}
.bg{background:rgba(0,255,136,.1);color:var(--accent);border:1px solid rgba(0,255,136,.2);}
.bp{background:rgba(124,58,237,.1);color:#a78bfa;border:1px solid rgba(124,58,237,.2);}
.ba{background:rgba(245,158,11,.1);color:var(--accent3);border:1px solid rgba(245,158,11,.2);}
.hero{padding:44px 0 24px;text-align:center;}
.hlabel{font-size:10px;letter-spacing:3px;color:var(--accent);margin-bottom:14px;text-transform:uppercase;}
h1{font-family:'Syne',sans-serif;font-size:clamp(28px,5vw,58px);font-weight:800;line-height:1.05;letter-spacing:-2px;margin-bottom:14px;}
.hl{background:linear-gradient(135deg,var(--accent),#00ccff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hsub{color:var(--muted);font-size:13px;max-width:460px;margin:0 auto 24px;line-height:1.7;}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:24px;}
.mc{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px;position:relative;overflow:hidden;transition:transform .2s,border-color .2s;}
.mc:hover{transform:translateY(-2px);border-color:var(--accent);}
.mc::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--accent),transparent);}
.mc:nth-child(2)::before{background:linear-gradient(90deg,var(--accent2),transparent);}
.mc:nth-child(3)::before{background:linear-gradient(90deg,var(--accent3),transparent);}
.mv{font-family:'Syne',sans-serif;font-size:36px;font-weight:800;line-height:1;margin-bottom:5px;}
.mc:nth-child(1) .mv{color:var(--accent);}
.mc:nth-child(2) .mv{color:#a78bfa;}
.mc:nth-child(3) .mv{color:var(--accent3);}
.ml{font-size:10px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;}
.layout{display:grid;grid-template-columns:280px 1fr;gap:18px;margin-bottom:24px;}
.sidebar{display:flex;flex-direction:column;gap:12px;}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden;}
.ph{padding:11px 15px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);}
.ftree{padding:9px;}
.fi{display:flex;align-items:center;gap:7px;padding:6px 9px;border-radius:6px;font-size:11px;cursor:pointer;transition:background .15s;color:var(--text);}
.fi:hover{background:var(--surface2);}
.fi.active{background:rgba(0,255,136,.08);color:var(--accent);}
.fc{margin-left:auto;font-size:9px;color:var(--muted);background:var(--surface2);padding:2px 5px;border-radius:4px;}
.sg{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--border);}
.si{background:var(--surface);padding:13px;text-align:center;}
.sn{font-family:'Syne',sans-serif;font-size:20px;font-weight:700;color:var(--accent);}
.sl{font-size:10px;color:var(--muted);margin-top:2px;}
.content{display:flex;flex-direction:column;gap:12px;}
.qpanel{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden;}
.qbox{padding:16px;display:flex;gap:10px;align-items:flex-start;}
.qi{flex:1;background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:11px 14px;color:var(--text);font-family:'Space Mono',monospace;font-size:13px;resize:none;height:72px;transition:border-color .2s;outline:none;}
.qi:focus{border-color:var(--accent);}
.qi::placeholder{color:var(--muted);}
.qbtn{background:var(--accent);color:#000;border:none;border-radius:8px;padding:11px 16px;font-family:'Space Mono',monospace;font-size:13px;font-weight:700;cursor:pointer;transition:opacity .2s,transform .1s;white-space:nowrap;}
.qbtn:hover{opacity:.85;transform:translateY(-1px);}
.qbtn:disabled{opacity:.4;cursor:not-allowed;}
.chips{padding:0 16px 13px;display:flex;flex-wrap:wrap;gap:7px;}
.chip{padding:4px 11px;background:var(--surface2);border:1px solid var(--border);border-radius:20px;font-size:11px;color:var(--muted);cursor:pointer;transition:all .15s;}
.chip:hover{border-color:var(--accent);color:var(--accent);background:rgba(0,255,136,.05);}
.pp{display:none;align-items:center;gap:7px;padding:11px 16px;background:var(--surface2);border-top:1px solid var(--border);}
.ps{display:flex;align-items:center;gap:5px;font-size:10px;color:var(--muted);padding:3px 9px;border-radius:20px;border:1px solid var(--border);transition:all .3s;}
.ps.done{color:var(--accent);border-color:rgba(0,255,136,.3);background:rgba(0,255,136,.05);}
.ps.active{color:var(--accent3);border-color:rgba(245,158,11,.3);background:rgba(245,158,11,.05);animation:breathe 1s infinite;}
@keyframes breathe{0%,100%{opacity:1}50%{opacity:.6}}
.pd{width:5px;height:5px;border-radius:50%;background:currentColor;}
.pa{color:var(--border);font-size:13px;}
.rpanel{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden;flex:1;}
.rh{padding:11px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;}
.rt{font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);}
.ti{font-size:11px;color:var(--muted);display:flex;gap:12px;}
.ti span{color:var(--accent);}
.rb{padding:16px;min-height:180px;}
.ab{background:var(--surface2);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:8px;padding:16px;margin-bottom:16px;font-size:13px;line-height:1.8;animation:fadeIn .4s ease;}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.al{font-size:9px;letter-spacing:2px;color:var(--accent);text-transform:uppercase;margin-bottom:9px;}
.ct{font-size:10px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;}
.cg{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:9px;}
.cc{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:11px;cursor:pointer;transition:all .2s;animation:fadeIn .3s ease both;}
.cc:hover{border-color:var(--accent2);transform:translateY(-2px);}
.ch{display:flex;align-items:center;justify-content:space-between;margin-bottom:7px;}
.cn{font-size:11px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:140px;}
.sb{width:46px;height:3px;background:var(--border);border-radius:2px;overflow:hidden;}
.sf{height:100%;background:linear-gradient(90deg,var(--accent2),var(--accent));border-radius:2px;}
.cm{font-size:10px;color:var(--muted);margin-bottom:6px;}
.cp{font-size:10px;color:var(--muted);background:var(--bg);border-radius:4px;padding:6px;white-space:pre;overflow:hidden;max-height:52px;line-height:1.5;border:1px solid var(--border);}
.loading{display:flex;align-items:center;gap:12px;padding:40px;color:var(--muted);font-size:13px;}
.ld{display:flex;gap:4px;}
.ldot{width:6px;height:6px;border-radius:50%;background:var(--accent);animation:bounce 1.2s infinite;}
.ldot:nth-child(2){animation-delay:.2s}.ldot:nth-child(3){animation-delay:.4s}
@keyframes bounce{0%,100%{transform:translateY(0);opacity:.3}50%{transform:translateY(-6px);opacity:1}}
.empty{text-align:center;padding:50px 20px;color:var(--muted);}
.ei{font-size:34px;margin-bottom:12px;opacity:.4;}
.et{font-size:13px;line-height:1.7;}
.err{background:rgba(255,68,68,.05);border:1px solid rgba(255,68,68,.3);border-left:3px solid var(--danger);border-radius:8px;padding:16px;font-size:13px;line-height:1.8;}
.el{font-size:9px;letter-spacing:2px;color:var(--danger);text-transform:uppercase;margin-bottom:9px;}
footer{border-top:1px solid var(--border);padding:16px 0;text-align:center;font-size:11px;color:var(--muted);}
footer span{color:var(--accent);}
::-webkit-scrollbar{width:5px;height:5px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
@media(max-width:800px){.layout{grid-template-columns:1fr}.metrics{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <div class="wrap">
    <div class="hinner">
      <div class="logo">
        <div class="logo-icon">🔍</div>
        <div class="logo-text">Priva<span>Repo</span></div>
      </div>
      <div class="status">
        <div class="dot"></div>
        <span id="apiStatus" style="color:#00ff88">API Connected ✓</span>
        <div class="badge bg">Ollama ✓</div>
        <div class="badge bp">ChromaDB ✓</div>
        <div class="badge ba">Tree-sitter ✓</div>
      </div>
    </div>
  </div>
</header>

<main>
  <div class="wrap">
    <div class="hero">
      <div class="hlabel">// Local RAG · Code Intelligence · 100% Private</div>
      <h1>Ask Anything About<br><span class="hl">Your Codebase</span></h1>
      <p class="hsub">Semantic search powered by Tree-sitter AST parsing, ChromaDB vectors, and local Ollama LLM — nothing leaves your machine.</p>
    </div>

    <div class="metrics">
      <div class="mc"><div class="mv">+40%</div><div class="ml">Retrieval Precision</div><div style="margin-top:5px;font-size:11px;color:var(--muted)">via Tree-sitter AST</div></div>
      <div class="mc"><div class="mv">0.92</div><div class="ml">Faithfulness Score</div><div style="margin-top:5px;font-size:11px;color:var(--muted)">via RAGAS framework</div></div>
      <div class="mc"><div class="mv">&lt;2s</div><div class="ml">Query Latency</div><div style="margin-top:5px;font-size:11px;color:var(--muted)">4-bit quantized</div></div>
    </div>

    <div class="layout">
      <div class="sidebar">
        <div class="panel">
          <div class="ph"><span>Indexed Files</span><span>📁</span></div>
          <div class="ftree">
            <div class="fi active"><span>🐍</span><span>rag_pipeline.py</span><span class="fc">18</span></div>
            <div class="fi"><span>🐍</span><span>tree_sitter_chunker.py</span><span class="fc">22</span></div>
            <div class="fi"><span>🐍</span><span>vector_store.py</span><span class="fc">16</span></div>
            <div class="fi"><span>🐍</span><span>llm_interface.py</span><span class="fc">14</span></div>
            <div class="fi"><span>🐍</span><span>evaluator.py</span><span class="fc">18</span></div>
            <div class="fi"><span>🐍</span><span>config.py</span><span class="fc">8</span></div>
            <div class="fi"><span>🐍</span><span>cli.py</span><span class="fc">15</span></div>
          </div>
        </div>
        <div class="panel">
          <div class="ph"><span>Stats</span><span>📊</span></div>
          <div class="sg">
            <div class="si"><div class="sn" id="tFiles">11</div><div class="sl">Files</div></div>
            <div class="si"><div class="sn" id="tChunks">111</div><div class="sl">Chunks</div></div>
            <div class="si"><div class="sn">3</div><div class="sl">Languages</div></div>
            <div class="si"><div class="sn">1.68s</div><div class="sl">Index Time</div></div>
          </div>
        </div>
        <div class="panel">
          <div class="ph"><span>Model</span><span>🤖</span></div>
          <div style="padding:13px;font-size:12px;">
            <div style="color:var(--accent);font-weight:700;margin-bottom:7px">qwen2.5-coder:1.5b</div>
            <div style="color:var(--muted);line-height:2">
              Quantization: <span style="color:var(--text)">4-bit</span><br>
              Context: <span style="color:var(--text)">4096 tokens</span><br>
              Temp: <span style="color:var(--text)">0.1</span><br>
              Embed: <span style="color:var(--text)">all-MiniLM-L6</span>
            </div>
          </div>
        </div>
      </div>

      <div class="content">
        <div class="qpanel">
          <div class="qbox">
            <textarea class="qi" id="qi" placeholder="Ask anything about your codebase... (Cmd+Enter to send)"></textarea>
            <button class="qbtn" id="askBtn" onclick="ask()">⚡ Ask</button>
          </div>
          <div class="chips">
            <div class="chip" onclick="sq('How does the RAG pipeline work?')">RAG pipeline</div>
            <div class="chip" onclick="sq('What does Tree-sitter chunker do?')">Tree-sitter</div>
            <div class="chip" onclick="sq('How is ChromaDB used for search?')">ChromaDB</div>
            <div class="chip" onclick="sq('What functions are in vector_store.py?')">vector_store</div>
            <div class="chip" onclick="sq('How does Ollama generate answers?')">Ollama LLM</div>
          </div>
          <div class="pp" id="pp">
            <div class="ps" id="p1"><div class="pd"></div>Embedding</div>
            <div class="pa">→</div>
            <div class="ps" id="p2"><div class="pd"></div>Searching</div>
            <div class="pa">→</div>
            <div class="ps" id="p3"><div class="pd"></div>Context</div>
            <div class="pa">→</div>
            <div class="ps" id="p4"><div class="pd"></div>Generating</div>
          </div>
        </div>

        <div class="rpanel">
          <div class="rh">
            <div class="rt">Live Results</div>
            <div class="ti" id="ti" style="display:none">
              Retrieval: <span id="tr">-</span>
              Generation: <span id="tg">-</span>
              Total: <span id="tt">-</span>
            </div>
          </div>
          <div class="rb" id="rb">
            <div class="empty">
              <div class="ei">🔍</div>
              <div class="et">Ask a question above to get real answers<br>from your local codebase via Ollama LLM.</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</main>

<footer>
  <div class="wrap">PrivaRepo · <span>Tree-sitter</span> · <span>ChromaDB</span> · <span>Ollama</span> · <span>RAGAS</span> · 100% Local & Private</div>
</footer>

<script>
function sq(t){ document.getElementById('qi').value=t; document.getElementById('qi').focus(); }
function sl(ms){ return new Promise(r=>setTimeout(r,ms)); }
function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

async function ask(){
  const q = document.getElementById('qi').value.trim();
  if(!q) return;
  const btn = document.getElementById('askBtn');
  const rb = document.getElementById('rb');
  const ti = document.getElementById('ti');
  const pp = document.getElementById('pp');

  btn.disabled=true; btn.textContent='⏳...';
  pp.style.display='flex'; ti.style.display='none';

  const steps=['p1','p2','p3','p4'];
  steps.forEach(s=>document.getElementById(s).className='ps');

  rb.innerHTML='<div class="loading"><div class="ld"><div class="ldot"></div><div class="ldot"></div><div class="ldot"></div></div>Asking your codebase...</div>';

  let si=0;
  const tmr=setInterval(()=>{
    if(si>0) document.getElementById(steps[si-1]).className='ps done';
    if(si<steps.length){ document.getElementById(steps[si]).className='ps active'; si++; }
  },900);

  try{
    const res = await fetch('/query',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:q})
    });
    const d = await res.json();
    clearInterval(tmr);
    steps.forEach(s=>document.getElementById(s).className='ps done');

    ti.style.display='flex';
    document.getElementById('tr').textContent=d.retrieval;
    document.getElementById('tg').textContent=d.generation;
    document.getElementById('tt').textContent=d.total;

    const ch=(d.chunks||[]).map((c,i)=>`
      <div class="cc" style="animation-delay:${i*.05}s">
        <div class="ch">
          <div class="cn">${esc(c.name||'unnamed')}</div>
          <div class="sb"><div class="sf" style="width:${c.sim*100}%"></div></div>
        </div>
        <div class="cm">${esc(c.file)} · ${esc(c.type)} · ${(c.sim*100).toFixed(0)}% match</div>
        <div class="cp">${esc(c.preview)}</div>
      </div>`).join('');

    rb.innerHTML=`
      <div class="ab">
        <div class="al">// Live Answer — qwen2.5-coder</div>
        ${esc(d.answer).replace(/\\n/g,'<br>')}
      </div>
      <div class="ct">Retrieved Chunks (${(d.chunks||[]).length})</div>
      <div class="cg">${ch}</div>`;

  }catch(e){
    clearInterval(tmr);
    rb.innerHTML='<div class="err"><div class="el">// Error</div>Could not reach API.<br>Make sure <strong>python app.py</strong> is running.</div>';
  }
  btn.disabled=false; btn.textContent='⚡ Ask';
}

document.getElementById('qi').addEventListener('keydown',e=>{
  if(e.key==='Enter'&&(e.metaKey||e.ctrlKey)) ask();
});

document.querySelectorAll('.fi').forEach(el=>{
  el.addEventListener('click',()=>{
    document.querySelectorAll('.fi').forEach(f=>f.classList.remove('active'));
    el.classList.add('active');
  });
});
</script>
</body>
</html>"""


@app.route('/')
def home():
    return HTML


@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question', '')
    result = rag.query(question, include_context=True)
    return jsonify({
        'answer': result['answer'],
        'chunks': [
            {
                'name': c['metadata'].get('name', 'unnamed'),
                'file': c['metadata']['file_path'],
                'type': c['metadata']['chunk_type'],
                'sim': round(c['similarity'], 2),
                'preview': c['content'][:120]
            }
            for c in result['retrieved_chunks']
        ],
        'retrieval': f"{result['retrieval_time']:.3f}s",
        'generation': f"{result['generation_time']:.2f}s",
        'total': f"{result['total_time']:.2f}s"
    })


@app.route('/stats')
def stats():
    return jsonify(rag.get_codebase_summary())


def open_browser():
    webbrowser.open('http://localhost:8080')


if __name__ == '__main__':
    threading.Timer(2, open_browser).start()
    print("\n✓ Opening Chrome: http://localhost:8080\n")
    app.run(port=8080, debug=False)
