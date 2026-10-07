import re

with open("serve_ep_tracker.py", "r") as f:
    code = f.read()

# Add to the table headers
header_search = """          <th>P&amp;L %</th>
          <th>Open R</th>"""
header_replace = """          <th>P&amp;L %</th>
          <th>Open R</th>
          <th title="AI Probability for +50% target (Day 5 Live Inference)">AI 50</th>
          <th title="AI Probability for +150% target (Day 5 Live Inference)">AI 150</th>"""

if header_search in code:
    code = code.replace(header_search, header_replace)

row_search = """      <td style="color:${openColor}; font-weight:700;">${e.open_r != null ? (e.open_r>0?'+':'')+e.open_r.toFixed(1)+' R' : '·'}</td>
    </tr>`;
  }).join('');"""
row_replace = """      <td style="color:${openColor}; font-weight:700;">${e.open_r != null ? (e.open_r>0?'+':'')+e.open_r.toFixed(1)+' R' : '·'}</td>
      <td style="${e.ml_50 > 0.4 ? 'background-color:rgba(34, 197, 94, 0.2); font-weight:bold; color:#22c55e;' : ''}">${e.ml_50 != null ? (e.ml_50 * 100).toFixed(1) + '%' : '·'}</td>
      <td style="${e.ml_150 > 0.15 ? 'background-color:rgba(34, 197, 94, 0.2); font-weight:bold; color:#22c55e;' : ''}">${e.ml_150 != null ? (e.ml_150 * 100).toFixed(1) + '%' : '·'}</td>
    </tr>`;
  }).join('');"""

if row_search in code:
    code = code.replace(row_search, row_replace)

# Change colspan from 10 to 12
code = code.replace('colspan="10"', 'colspan="12"')

with open("serve_ep_tracker.py", "w") as f:
    f.write(code)
print("serve_ep_tracker patched with table columns!")
