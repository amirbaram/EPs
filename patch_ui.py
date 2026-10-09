import sys

content = open("serve_ep_tracker.py").read()

# Replace the generic DYN STOP label with a clear one.
content = content.replace(
    'DYN STOP: <span id="ai_stop_val_v1"></span>',
    '🤖 ML Dynamic Stoploss: <span id="ai_stop_val_v1"></span>'
)
content = content.replace(
    'DYN STOP: <span id="ai_stop_val_v2"></span>',
    '🤖 ML Dynamic Stoploss: <span id="ai_stop_val_v2"></span>'
)

# Insert the Re-Entry panel under Exhaustion
reentry_html_v1 = """             <div id="ai_exhaustion_v1" style="color: #8b5cf6; padding: 4px; border: 1px solid #8b5cf6; border-radius: 4px; display: none; text-align: center; margin-top: 5px;">🔥 EXHAUSTION: <span id="ai_exh_val_v1"></span></div>
             <div id="ai_reentry_v1" style="color: #34d399; padding: 4px; border: 1px solid #34d399; border-radius: 4px; display: none; text-align: center; margin-top: 5px;">♻️ ML TRADE 2/3 RE-ENTRY: <span id="ai_reentry_val_v1"></span></div>"""
content = content.replace(
    '<div id="ai_exhaustion_v1" style="color: #8b5cf6; padding: 4px; border: 1px solid #8b5cf6; border-radius: 4px; display: none; text-align: center; margin-top: 5px;">🔥 EXHAUSTION: <span id="ai_exh_val_v1"></span></div>',
    reentry_html_v1
)

reentry_html_v2 = """             <div id="ai_exhaustion_v2" style="color: #8b5cf6; padding: 4px; border: 1px solid #8b5cf6; border-radius: 4px; display: none; text-align: center; margin-top: 5px;">🔥 EXHAUSTION: <span id="ai_exh_val_v2"></span></div>
             <div id="ai_reentry_v2" style="color: #34d399; padding: 4px; border: 1px solid #34d399; border-radius: 4px; display: none; text-align: center; margin-top: 5px;">♻️ ML TRADE 2/3 RE-ENTRY: <span id="ai_reentry_val_v2"></span></div>"""
content = content.replace(
    '<div id="ai_exhaustion_v2" style="color: #8b5cf6; padding: 4px; border: 1px solid #8b5cf6; border-radius: 4px; display: none; text-align: center; margin-top: 5px;">🔥 EXHAUSTION: <span id="ai_exh_val_v2"></span></div>',
    reentry_html_v2
)

# Now wire it into the javascript
js_reentry = """
  if (ev.prob_exhaustion != null) {
"""
js_reentry_insert = """
  if (ev.prob_reentry != null && ev.prob_reentry > 0.40) {
      document.getElementById('ai_reentry_v1').style.display = 'block';
      let reProb = (ev.prob_reentry * 100).toFixed(1) + '%';
      if (ev.prob_reentry > 0.60) { reProb += ' (BUY TRIGGER)'; }
      document.getElementById('ai_reentry_val_v1').textContent = reProb;
      adv_warnings_v1 = true;
  } else {
      document.getElementById('ai_reentry_v1').style.display = 'none';
  }
  
  if (ev.prob_exhaustion != null) {
"""
content = content.replace(js_reentry, js_reentry_insert)

# Ensure the V2 ones exist as well
js_reentry_v2 = """
  if (ev.prob_exhaustion != null) {
      document.getElementById('ai_exhaustion_v2').style.display = 'block';
"""
js_reentry_insert_v2 = """
  if (ev.prob_reentry != null && ev.prob_reentry > 0.40) {
      document.getElementById('ai_reentry_v2').style.display = 'block';
      let reProb = (ev.prob_reentry * 100).toFixed(1) + '%';
      if (ev.prob_reentry > 0.60) { reProb += ' (BUY TRIGGER)'; }
      document.getElementById('ai_reentry_val_v2').textContent = reProb;
      adv_warnings_v2 = true;
  } else {
      document.getElementById('ai_reentry_v2').style.display = 'none';
  }

  if (ev.prob_exhaustion != null) {
      document.getElementById('ai_exhaustion_v2').style.display = 'block';
"""
content = content.replace(js_reentry_v2, js_reentry_insert_v2)


open("serve_ep_tracker.py", "w").write(content)
