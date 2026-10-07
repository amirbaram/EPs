import re

with open("serve_ep.py", "r") as f:
    code = f.read()

row_insertion = """      <td ${c60}>${e.ret_60d != null ? (e.ret_60d >= 0 ? '+' : '') + e.ret_60d.toFixed(1) + '%' : '·'}</td>
      <td style="font-weight:700; color:var(--gold);">${e.max_gain != null ? e.max_gain.toFixed(1) + '%' : '·'}</td>
      <td class="tl"><span class="tag ${tagCls}">${e.outcome || '·'}</span></td>
      <td style="${e.ml_50 > 0.40 ? 'background-color:rgba(34, 197, 94, 0.2); font-weight:bold; color:#22c55e;' : ''}">${e.ml_50 != null ? (e.ml_50 * 100).toFixed(1) + '%' : '·'}</td>
      <td style="${e.ml_150 > 0.15 ? 'background-color:rgba(34, 197, 94, 0.2); font-weight:bold; color:#22c55e;' : ''}">${e.ml_150 != null ? (e.ml_150 * 100).toFixed(1) + '%' : '·'}</td>"""

code = code.replace("""      <td ${c60}>${e.ret_60d != null ? (e.ret_60d >= 0 ? '+' : '') + e.ret_60d.toFixed(1) + '%' : '·'}</td>
      <td style="font-weight:700; color:var(--gold);">${e.max_gain != null ? e.max_gain.toFixed(1) + '%' : '·'}</td>
      <td class="tl"><span class="tag ${tagCls}">${e.outcome || '·'}</span></td>""", row_insertion)

with open("serve_ep.py", "w") as f:
    f.write(code)
print("serve_ep.py patched for ML columns!")
