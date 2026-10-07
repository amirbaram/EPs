import re

with open("serve_ep.py", "r") as f:
    code = f.read()

# Add AI Section in HTML
html_search = """        <!-- 5. Larsson State -->"""
html_replace = """        <!-- AI Scores -->
        <div style="padding:10px 14px; border-bottom:1px solid #1c2538;">
          <div style="font-weight:700; color:#cbd5e1; margin-bottom:5px; display:flex; justify-content:space-between; align-items:center;">
            <span>🤖 AI Confidence & Alpha Rank (Day 5)</span>
          </div>
          <div style="display:grid; grid-template-columns: 1fr 1fr; gap:6px;" id="ai_grid">
             <!-- Populated by JS -->
          </div>
        </div>
        <!-- 5. Larsson State -->"""

if html_search in code:
    code = code.replace(html_search, html_replace)

# Update JS loadChart
js_search = """    // 1. Sector / Theme Context (Right Pane)
    const d = data.dossier;"""
js_replace = """    // 1. Sector / Theme Context (Right Pane)
    const d = data.dossier;
    
    if (d && d.ai_scores) {
        let aiHtml = '';
        const targets = { '50': '+50%', '100': '+100%', '150': '+150%', '200': '+200%' };
        for (const [k, title] of Object.entries(targets)) {
            const prob = d.ai_scores[`ml_${k}`];
            if (prob == null) continue;
            const pct = (prob * 100).toFixed(1) + '%';
            let color = 'var(--muted)';
            if (prob > 0.4 && k === '50') color = 'var(--green)';
            if (prob > 0.15 && k !== '50') color = 'var(--green)';
            if (prob > 0.5) color = 'var(--gold)';
            aiHtml += `<div style="background:#131a26; padding:6px; border-radius:4px; border:1px solid #1e293b; text-align:center;">
              <div style="font-size:10px; color:#94a3b8; margin-bottom:3px;">${title} Target</div>
              <div style="font-size:14px; font-weight:bold; color:${color};">${pct}</div>
            </div>`;
        }
        document.getElementById('ai_grid').innerHTML = aiHtml;
    } else {
        document.getElementById('ai_grid').innerHTML = '<div style="color:var(--muted); font-size:11px;">AI Scores not available</div>';
    }
"""

if js_search in code:
    code = code.replace(js_search, js_replace)

with open("serve_ep.py", "w") as f:
    f.write(code)

print("Dossier patched!")
