import re

with open("serve_ep.py", "r") as f:
    code = f.read()

# Add AI Section in HTML
html_search = """    <!-- Card 2: Sector & Theme Momentum Backdrop -->"""
html_replace = """    <!-- AI Scores -->
    <div style="padding:10px 14px; border-bottom:1px solid #1c2538;">
      <div style="font-weight:700; color:#cbd5e1; margin-bottom:5px; display:flex; justify-content:space-between; align-items:center;">
        <span>🤖 AI Confidence & Alpha Rank (Day 5)</span>
      </div>
      <div style="display:grid; grid-template-columns: 1fr 1fr; gap:6px;" id="ai_grid">
         <!-- Populated by JS -->
      </div>
    </div>
    <!-- Card 2: Sector & Theme Momentum Backdrop -->"""

if html_search in code:
    code = code.replace(html_search, html_replace)
    print("HTML ai_grid added!")
else:
    print("HTML failed!")

with open("serve_ep.py", "w") as f:
    f.write(code)

