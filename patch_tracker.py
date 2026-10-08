with open("serve_ep_tracker.py", "r") as f:
    text = f.read()

old_html = """            <div>
                <span class="text-gray-400">P(200%)</span>
                <span style="color:#9C27B0;" id="ai_v2_200">--%</span>
            </div>
        </div>
    </div>"""

new_html = """            <div>
                <span class="text-gray-400">P(200%)</span>
                <span style="color:#9C27B0;" id="ai_v2_200">--%</span>
            </div>
        </div>
        <div id="ai_adv_warnings_v2" style="margin-top: 15px; font-weight: bold; font-size: 0.9em; display: none;">
            <div id="ai_toxic_warning_v2" style="color: #ff4444; padding: 4px; border: 1px solid #ff4444; border-radius: 4px; display: none; text-align: center; margin-bottom: 8px;">⚠️ TOXIC FLOW DETECTED</div>
            <div id="ai_dynamic_stop_v2" style="color: #ff9800; padding: 4px; border: 1px solid #ff9800; border-radius: 4px; display: none; text-align: center;">DYNAMIC STOP: <span id="ai_stop_val_v2"></span></div>
        </div>
    </div>"""
text = text.replace(old_html, new_html)

old_js = """        if (data.probs) {
            document.getElementById('ai_v2_50').textContent = (data.probs.prob_50 * 100).toFixed(1) + '%';
            document.getElementById('ai_v2_100').textContent = (data.probs.prob_100 * 100).toFixed(1) + '%';
            document.getElementById('ai_v2_150').textContent = (data.probs.prob_150 * 100).toFixed(1) + '%';
            document.getElementById('ai_v2_200').textContent = (data.probs.prob_200 * 100).toFixed(1) + '%';
        }"""

new_js = """        if (data.probs) {
            document.getElementById('ai_v2_50').textContent = (data.probs.prob_50 * 100).toFixed(1) + '%';
            document.getElementById('ai_v2_100').textContent = (data.probs.prob_100 * 100).toFixed(1) + '%';
            document.getElementById('ai_v2_150').textContent = (data.probs.prob_150 * 100).toFixed(1) + '%';
            document.getElementById('ai_v2_200').textContent = (data.probs.prob_200 * 100).toFixed(1) + '%';
            
            let adv_warnings = false;
            if (data.probs.is_toxic) {
                document.getElementById('ai_toxic_warning_v2').style.display = 'block';
                adv_warnings = true;
            } else {
                document.getElementById('ai_toxic_warning_v2').style.display = 'none';
            }
            
            if (data.probs.dynamic_stop_loss_pct != null) {
                document.getElementById('ai_dynamic_stop_v2').style.display = 'block';
                document.getElementById('ai_stop_val_v2').textContent = (data.probs.dynamic_stop_loss_pct * 100).toFixed(1) + '%';
                adv_warnings = true;
            } else {
                document.getElementById('ai_dynamic_stop_v2').style.display = 'none';
            }
            
            if (adv_warnings) {
                document.getElementById('ai_adv_warnings_v2').style.display = 'block';
            } else {
                document.getElementById('ai_adv_warnings_v2').style.display = 'none';
            }
        }"""
text = text.replace(old_js, new_js)

with open("serve_ep_tracker.py", "w") as f:
    f.write(text)
