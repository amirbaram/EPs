import requests, json

r = requests.get("http://127.0.0.1:8783/api/tracker_events")
data = r.json()
events = data.get("events", [])
print(f"Total tracked events: {len(events)}")

for e in events:
    score = 0.0
    rvol = e.get("rvol", 1.0)
    dvol = e.get("dvol_m", 0.0)
    c_pos = e.get("close_pos", 0.5)
    thm_score = e.get("theme_score", 50)
    
    score += min(10.0, rvol) * 5.0
    score += min(30.0, dvol / 5.0)
    score += (c_pos - 0.5) * 40.0
    score += (thm_score / 100.0) * 20.0
    if e.get("held_48h", False): score += 15.0
    if e.get("above_sma50", False): score += 15.0
    if e.get("is_novel_9m", False): score += 20.0
    if e.get("is_thematic_veto", False) and not e.get("is_idiosyncratic", False):
        score -= 50.0
    if e.get("held_d1_low", True):
        score += 10.0
    else:
        score -= 100.0 # stopped out
    e["composite_score"] = round(score, 1)

events.sort(key=lambda x: x["composite_score"], reverse=True)

print("\n--- TOP 12 RANKED LIVE SETUPS ---")
for i, e in enumerate(events[:12]):
    sym = e["symbol"]
    dt = e["event_date"]
    day = e["bars_since"] + 1
    arch = e.get("trader_archetype", "Standard")
    gap = e["gap_pct"]
    rvol = e["rvol"]
    dvol = e["dvol_m"]
    cp = e["close_pos"]
    r = e["curr_r"]
    ret = e["ret_pct"]
    status = e["status"]
    score = e["composite_score"]
    t2 = e.get("t2_info", {})
    t2_str = t2.get("status_desc", "None") if t2 else "None"
    
    print(f"\n{i+1}. {sym} ({dt}, Day {day})")
    print(f"   • Archetype: {arch} | Score: {score}")
    print(f"   • Microstructure: Gap +{gap}%, RVOL {rvol}x, ${dvol}M Traded, ClosePos {cp}")
    print(f"   • Open Performance: {r:+.2f} R ({ret:+.1f}%) | Status: {status}")
    print(f"   • Levels: Entry ${e['entry_price']:.2f}, Stop ${e['stop_price']:.2f} (-{e['risk_pct']:.1f}%), Add ${e['add_price']:.2f}, Current ${e['curr_price']:.2f}")
    print(f"   • Theme: {e.get('sector')} / {e.get('theme')} (Score: {e.get('theme_score')}/100)")
    print(f"   • Trade 2/3 Status: {t2_str}")
    print(f"   • Action Headline: {e.get('action_headline')}")
    print(f"   • Action Plan:\n{e.get('action_plan')}")
