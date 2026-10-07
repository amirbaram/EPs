import re

with open("serve_ep_tracker.py", "r") as f:
    code = f.read()

row_search = """      <td style="color:${rColor}; font-weight:700;">${e.curr_r >= 0 ? '+' : ''}${e.curr_r} R</td>
    </tr>`;
  }).join('');"""
row_replace = """      <td style="color:${rColor}; font-weight:700;">${e.curr_r >= 0 ? '+' : ''}${e.curr_r} R</td>
      <td style="${e.ml_50 > 0.4 ? 'background-color:rgba(34, 197, 94, 0.2); font-weight:bold; color:#22c55e;' : ''}">${e.ml_50 != null ? (e.ml_50 * 100).toFixed(1) + '%' : '·'}</td>
      <td style="${e.ml_150 > 0.15 ? 'background-color:rgba(34, 197, 94, 0.2); font-weight:bold; color:#22c55e;' : ''}">${e.ml_150 != null ? (e.ml_150 * 100).toFixed(1) + '%' : '·'}</td>
    </tr>`;
  }).join('');"""

if row_search in code:
    code = code.replace(row_search, row_replace)
    print("tracker table replaced!")

with open("serve_ep_tracker.py", "w") as f:
    f.write(code)

