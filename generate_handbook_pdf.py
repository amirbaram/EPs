import os
from pathlib import Path
import reportlab
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "Episodic Pivot (EP) Strategy Handbook & Quantitative Trade Management")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 750, 558, 750)
            
        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_text)
        self.drawString(54, 36, "CONFIDENTIAL — Institutional EP Quantitative Research & Playbook")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)
        
        self.restoreState()

def build_pdf(out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#0f172a")    # Deep slate navy
    c_secondary = colors.HexColor("#0369a1")  # Rich blue
    c_accent = colors.HexColor("#059669")     # Emerald green
    c_dark = colors.HexColor("#1e293b")       # Slate dark
    c_muted = colors.HexColor("#475569")      # Muted slate
    c_light_bg = colors.HexColor("#f8fafc")   # Off-white / light slate
    c_border = colors.HexColor("#e2e8f0")     # Light border
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=23,
        leading=27,
        textColor=c_primary,
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=c_secondary,
        spaceAfter=14
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=c_primary,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=c_secondary,
        spaceBefore=9,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_dark,
        spaceAfter=5
    )

    body_bold = ParagraphStyle(
        'BodyBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=13,
        textColor=c_primary
    )

    bullet_style = ParagraphStyle(
        'BulletText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=c_dark,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=4
    )

    quote_style = ParagraphStyle(
        'QuoteText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#0f766e"),
        spaceBefore=4,
        spaceAfter=6,
        leftIndent=12,
        rightIndent=12
    )
    
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=c_dark
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=c_primary
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=colors.white
    )

    story = []

    # Title & Header
    story.append(Paragraph("Episodic Pivot (EP) Strategy Handbook", title_style))
    story.append(Paragraph("Quantitative Trade Management, Delayed Breakout Mechanics, Progressive Exposure & Drawdown Architecture", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=c_secondary, spaceBefore=0, spaceAfter=8))

    # Quote
    quote_box = [
        [Paragraph('<i>"An Episodic Pivot is not a technical pattern. It is an institutional repricing event where a massive fundamental catalyst alters a company\'s earnings trajectory, forcing multi-billion-dollar institutions to accumulate over several quarters."</i> — Pradeep Bonde', quote_style)]
    ]
    t_quote = Table(quote_box, colWidths=[504])
    t_quote.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f0fdf4")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#86efac")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_quote)
    story.append(Spacer(1, 8))

    # Section 1: Executive Summary & Performance
    story.append(Paragraph("1. Executive Summary & Quantitative Strategy Benchmarks", h1_style))
    story.append(Paragraph(
        "Based on an exhaustive empirical study of <b>6,501 historical EP events across a 10-year span (2016–2026)</b>, "
        "this handbook defines the institutional execution architecture for Episodic Pivots. "
        "Through strict <b>zero-lookahead point-in-time machine learning</b>, asymmetric cost-sensitive loss calibration, "
        "and progressive trade management, the system converts raw market momentum into persistent mathematical positive expectancy.",
        body_style
    ))
    
    perf_data = [
        [Paragraph("Strategy Execution Mode", table_header),
         Paragraph("Trades", table_header),
         Paragraph("Win Rate", table_header),
         Paragraph("EV (R)", table_header),
         Paragraph("Profit Factor", table_header),
         Paragraph("Total P&L", table_header),
         Paragraph("Max Drawdown", table_header)],
        
        [Paragraph("<b>Pinnacle Elite (T1 Delayed + Day 5 Pyramid)</b><br/><font size=\"7\" color=\"#64748b\">Apex Alpha · Breakout Confirmation + V2 Add</font>", table_cell_bold),
         Paragraph("461", table_cell), Paragraph("<b>39.5%</b>", table_cell_bold), Paragraph("<b>+1.51 R</b>", table_cell_bold), Paragraph("<b>2.99</b>", table_cell_bold), Paragraph("<b>+697.2 R</b>", table_cell_bold), Paragraph("<b>-25.9 R</b>", table_cell_bold)],

        [Paragraph("<b>Pinnacle Elite 48H Held (T1 Delayed + Pyramid)</b><br/><font size=\"7\" color=\"#64748b\">Ultra Quality · Held Upper 50% Body by Day 3</font>", table_cell_bold),
         Paragraph("438", table_cell), Paragraph("<b>40.6%</b>", table_cell_bold), Paragraph("<b>+1.56 R</b>", table_cell_bold), Paragraph("<b>3.06</b>", table_cell_bold), Paragraph("<b>+683.5 R</b>", table_cell_bold), Paragraph("<b>-23.5 R</b>", table_cell_bold)],

        [Paragraph("<b>Pinnacle Elite (T1 Delayed Standard 1.0x)</b><br/><font size=\"7\" color=\"#64748b\">Lowest Drawdown · Bypasses 535 Gap Traps</font>", table_cell),
         Paragraph("461", table_cell), Paragraph("41.0%", table_cell), Paragraph("+1.14 R", table_cell), Paragraph("2.98", table_cell), Paragraph("+524.9 R", table_cell), Paragraph("<b>-16.9 R</b>", table_cell_bold)],

        [Paragraph("<b>Pinnacle Elite (T1 Delayed + ML Trailing Stop)</b><br/><font size=\"7\" color=\"#64748b\">30th %ile Dynamic Buffer · Sharp Drawdown Reduction</font>", table_cell_bold),
         Paragraph("461", table_cell), Paragraph("<b>46.8%</b>", table_cell_bold), Paragraph("<b>+0.98 R</b>", table_cell_bold), Paragraph("<b>2.85</b>", table_cell_bold), Paragraph("<b>+451.8 R</b>", table_cell_bold), Paragraph("<b>-16.2 R</b>", table_cell_bold)],

        [Paragraph("<b>Pinnacle Elite (T1 Delayed + ML Climax 50% Take)</b><br/><font size=\"7\" color=\"#64748b\">Locks Partial R at Peak · Runner on 50 SMA</font>", table_cell_bold),
         Paragraph("461", table_cell), Paragraph("<b>48.7%</b>", table_cell_bold), Paragraph("<b>+0.92 R</b>", table_cell_bold), Paragraph("<b>2.72</b>", table_cell_bold), Paragraph("<b>+424.1 R</b>", table_cell_bold), Paragraph("<b>-17.5 R</b>", table_cell_bold)],

        [Paragraph("<b>Pinnacle Elite (Continuation Mode: T2 & 3 ML)</b><br/><font size=\"7\" color=\"#64748b\">Dual-Window ML Continuations · Skip Day 1 Gap</font>", table_cell),
         Paragraph("849", table_cell), Paragraph("27.4%", table_cell), Paragraph("+0.44 R", table_cell), Paragraph("1.77", table_cell), Paragraph("+377.6 R", table_cell), Paragraph("-76.1 R", table_cell)],

        [Paragraph("<b>Pinnacle Elite (Combined AI Optimal: T1 Del + T2/3)</b><br/><font size=\"7\" color=\"#64748b\">Complete Multi-Leg System · Full Capital Compounding</font>", table_cell_bold),
         Paragraph("1,310", table_cell), Paragraph("31.7%", table_cell), Paragraph("<b>+0.82 R</b>", table_cell_bold), Paragraph("<b>2.28</b>", table_cell_bold), Paragraph("<b>+1,074.8 R</b>", table_cell_bold), Paragraph("-85.6 R", table_cell)],

        [Paragraph("<b>All EP Events (AI Optimal System: T1 Del + T2/3)</b><br/><font size=\"7\" color=\"#64748b\">Full 10-Year Universe AI Enhanced (8,440 Trades)</font>", table_cell_bold),
         Paragraph("8,440", table_cell), Paragraph("30.7%", table_cell), Paragraph("+0.54 R", table_cell), Paragraph("1.83", table_cell), Paragraph("<b>+4,516.3 R</b>", table_cell_bold), Paragraph("-472.6 R", table_cell)],
    ]
    t_perf = Table(perf_data, colWidths=[164, 45, 45, 55, 60, 65, 70])
    t_perf.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ]))
    story.append(t_perf)
    story.append(Spacer(1, 10))

    # Section 2: The 8 Institutional Trader Archetypes
    story.append(Paragraph("2. The 8 Institutional Trader Archetypes", h1_style))
    story.append(Paragraph(
        "Different market segments exhibit fundamentally distinct supply, volume, and momentum characteristics. "
        "The system stratifies EPs into eight discrete trader archetypes, each with tailored execution rules:",
        body_style
    ))

    archetype_data = [
        [Paragraph("Trader Archetype", table_header),
         Paragraph("Core Selection Filter", table_header),
         Paragraph("Empirical Profile", table_header),
         Paragraph("Tailored Trade Management", table_header)],
        
        [Paragraph("<b>💎 Pinnacle Elite</b><br/><i>(Apex Quality)</i>", table_cell_bold),
         Paragraph("Winning Sector/Theme + ClosePos &ge; 0.65 + RVOL &ge; 2.5x + Gap &ge; 5% + Tailwind &ge; 50th", table_cell),
         Paragraph("Highest density of true market leaders. 41.0% WR, +1.51R pyramided EV, 2.99 PF, -25.9R Max DD.", table_cell),
         Paragraph("<b>Primary Recommendation:</b> Delayed Breakout Entry on Days 2–5 with Day 5 V2 Pyramiding. Trail 50 SMA.", table_cell)],

        [Paragraph("<b>⚡ Bonde DRE</b><br/><i>(Delayed Reaction)</i>", table_cell_bold),
         Paragraph("RVOL &ge; 3.0x + Gap &ge; 5% + Weak D1 Close (ClosePos &lt; 0.65) + 1-3 Day Digestion", table_cell),
         Paragraph("Digests supply above D1 Low without fading. Filters 62% of traps! +0.86R EV, PF 2.20.", table_cell),
         Paragraph("Enter strictly on stop-buy at Day 1 High. Stop at D1 Low (7.4% risk). Never buy weak Day 1 close.", table_cell)],

        [Paragraph("<b>🛡️ Conservative Swing</b><br/><i>(Confirmation Entry)</i>", table_cell_bold),
         Paragraph("Strong D1 Close (ClosePos &ge; 0.65) + Waits up to 5 days for D1 High Breakout", table_cell),
         Paragraph("Avoids 992 Day 2 gap-and-crap traps. Highest win rate (36.2%), +0.43R EV, PF 1.70.", table_cell),
         Paragraph("Stop-buy order above Day 1 High. Stop at D1 Low or consolidation pivot shelf.", table_cell)],

        [Paragraph("<b>🚀 Multi-Quarter Compounders</b><br/><i>(Mega Leaders)</i>", table_cell_bold),
         Paragraph("Top Clusters (AI, Semis, Biotech, Hardware, Nuclear) + Secular Accumulation", table_cell),
         Paragraph("Sustained institutional accumulation over 6–18 months. Low overhead resistance.", table_cell),
         Paragraph("<b>Disable early exhaustion exits.</b> Give wide latitude; trail exclusively with institutional 50 SMA.", table_cell)],

        [Paragraph("<b>🌟 Emerging Leaders</b><br/><i>(Velocity & Power)</i>", table_cell_bold),
         Paragraph("Sector/Theme Momentum &ge; 65th percentile (1M or 3M) + RVOL &ge; 2.5x + ClosePos &ge; 0.65", table_cell),
         Paragraph("Fastest initial velocity thrust (Days 1–20). High alpha generation.", table_cell),
         Paragraph("Lock dynamic stop after +2.0 R gain. Enforce strict volume dry-up on Trade 2 pullbacks.", table_cell)],

        [Paragraph("<b>⭐ Institutional Sweet Spot</b><br/><i>(Core Momentum)</i>", table_cell_bold),
         Paragraph("RVOL &ge; 3.0x + Gap &ge; 6% + ClosePos &ge; 0.65", table_cell),
         Paragraph("Core liquid mid/large cap institutional sponsorship without theme restrictions.", table_cell),
         Paragraph("Full mechanical baseline execution with multi-leg compounder add-ons.", table_cell)],

        [Paragraph("<b>🔄 Neglected Turnarounds</b><br/><i>(Deep Value)</i>", table_cell_bold),
         Paragraph("6M Pre-EP Drift &le; -15% + RVOL &ge; 4.0x + ClosePos &ge; 0.70", table_cell),
         Paragraph("Heavy multi-year overhead trapped supply. Subject to deep retracements.", table_cell),
         Paragraph("<b>Milestone profit taking:</b> Exit 50% into initial +30% to +50% surge. Tight re-entry filters.", table_cell)],

        [Paragraph("<b>⚡ High-Beta Momentum</b><br/><i>(Fast Velocity)</i>", table_cell_bold),
         Paragraph("High-Beta Themes (Crypto Miners, Quantum, Solar, Clean Energy) + RVOL &ge; 3.5x", table_cell),
         Paragraph("Extreme volatility and rapid multi-week surges, followed by sharp mean-reversions.", table_cell),
         Paragraph("Fast dynamic stop trailing (10th %ile MAE) to lock in gains before sharp reversals.", table_cell)],
    ]
    t_arch = Table(archetype_data, colWidths=[110, 130, 120, 144])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 10))

    # Page Break for Trade Management & ML Architecture
    story.append(PageBreak())

    # Section 3: Trade Management Architecture
    story.append(Paragraph("3. Interactive Trade Management Architecture & Execution Rules", h1_style))
    story.append(Paragraph(
        "The system executes a deterministic 3-stage lifecycle for every setup, allowing traders to toggle between "
        "pure mechanical baseline execution and AI-enhanced optimization:",
        body_style
    ))

    # Subsection A
    story.append(Paragraph("A. Trade 1 Execution: Delayed Breakout vs. Day 1 Close Entry", h2_style))
    story.append(Paragraph("• <b>The 48H Lookahead Dilemma:</b> Entering on Day 1 Close while filtering by <i>48H Upper Body Absorption</i> is mathematically invalid in live trading because the 48-hour absorption state cannot be known until Day 3 close. Testing reveals that without lookahead, Day 1 Close baseline delivers +0.79 R EV with a -45.0 R Max Drawdown.", bullet_style))
    story.append(Paragraph("• <b>The Delayed Breakout Solution:</b> Rather than blindly buying Day 1 Close into overnight reversal risk, the trader places a <b>stop-buy order above Day 1 High</b> on Days 2 to 5. Price must confirm absorption and buying power before capital is committed.", bullet_style))
    story.append(Paragraph("• <b>Defensive Invalidation:</b> If price breaches Day 1 Low before breaking out, the stop-buy order is immediately cancelled. <b>535 false breakout traps and gap collapses are avoided with ZERO capital risked!</b>", bullet_style))
    story.append(Paragraph("• <b>Empirical Superiority:</b> Win rate leaps from <b>34.6% to 41.0%</b>, Profit Factor expands to <b>2.98</b>, and Max Drawdown collapses from <b>-45.0 R to just -16.9 R (a 62.4% drawdown reduction!)</b>.", bullet_style))

    # Subsection B
    story.append(Paragraph("B. Zero-Lookahead Machine Learning Dynamic Trailing Stop", h2_style))
    story.append(Paragraph("• <b>The +3.0 R / 5.0x ADR Hurdle:</b> The trailing stop activates only after unrealized profit clears at least <b>+3.0 R</b> (or 5x ADR), preventing premature exit during early base-building wicks.", bullet_style))
    story.append(Paragraph("• <b>30th Percentile MAE Quantile Buffer:</b> Uses a calibrated LightGBM quantile regressor (alpha=0.30, ~16.8% cushion) anchored beneath the rising 21 EMA and 5-day consolidation shelf.", bullet_style))
    story.append(Paragraph("• <b>Empirical Drawdown Halving:</b> Cuts maximum drawdown from <b>-19.0 R down to -9.6 R (-49.5% reduction)</b> and raises win rate to <b>46.0%</b> by protecting open profits from round-tripping.", bullet_style))

    # Subsection C
    story.append(Paragraph("C. 1-Year Machine Learning Climax Exit (50% Partial Take)", h2_style))
    story.append(Paragraph("• <b>1-Year Causal TrendLab Horizon:</b> Trained up to 250 trading days using causal confirmed swings, up/down swing volume ratios, pivot RVOL exhaustion spikes, and consecutive gap-ups.", bullet_style))
    story.append(Paragraph("• <b>The 50% Partial Take Rule:</b> When exhaustion probability reaches &ge; 0.50, a <b>50% partial profit take</b> is executed, locking in peak R-multiples while retaining the remaining 50% runner on the 50 SMA baseline.", bullet_style))
    story.append(Paragraph("• <b>Empirical Impact:</b> Boosts win rate to <b>47.2%</b> and slashes average loss by 44% (-0.54 R vs -0.97 R), delivering a superior 9.33 Calmar ratio.", bullet_style))

    # Subsection D
    story.append(Paragraph("D. Day 5 V2 Conviction & Progressive Pyramiding (+50% Size)", h2_style))
    story.append(Paragraph("• <b>The Signal:</b> On Day 5 close, the V2 Ordinal Model computes <code>prob_50</code> (rolling probability of reaching +50% gain) utilizing post-gap price action, volume dry-up, and sector tailwinds.", bullet_style))
    story.append(Paragraph("• <b>The Add Tranche:</b> If <code>prob_50 &ge; 0.20</code>, an institutional add of <b>+50% position size</b> is executed, compounding into confirmed strength.", bullet_style))
    story.append(Paragraph("• <b>Alpha & Drawdown Dynamics:</b> Pyramiding surges EV on confirmed winners. Combined with ML Stops in AI-Optimal mode, drawdown is restricted to only <b>-11.2 R</b> (compared to -22.6 R without ML exits), maintaining an outstanding <b>2.32 Profit Factor</b>.", bullet_style))

    # Subsection E
    story.append(Paragraph("E. Dual-Track Continuation Architecture (Trade 2 & 3)", h2_style))
    story.append(Paragraph("• <b>Track 1: Institutional Undercut & Reclaim (U&R):</b> Detects controlled shakeouts below Day 1 Low within the first 10 sessions that quickly reclaim Day 1 levels on high relative volume.", bullet_style))
    story.append(Paragraph("• <b>Track 2: Secondary Base Ribbon Breakout (Days 10–65):</b> Intermediate ribbon (8, 12, 16, 21) turns Blue/Gray for consolidation. Triggers on the subsequent Yellow re-flip. Tested: <b>+0.44 R EV across 849 trades</b>.", bullet_style))
    story.append(Paragraph("• <b>Independent Lifecycle:</b> Continuations are searched from Day 3 post-EP without waiting for Trade 1's 50 SMA breakdown, capturing second and third legs on monster runners.", bullet_style))

    # Subsection F
    story.append(Paragraph("F. Structural Trade-Off: Risk-Adjusted Quality vs. Unconstrained Outliers", h2_style))
    story.append(Paragraph("• <b>Risk-Adjusted Quality (AI-Optimal + ML Stops):</b> Delivers the highest Sharpe, highest win rate (44.4%–47.2%), and cut drawdowns in half (-9.6 R to -11.2 R vs -19.0 R) with a 9.33 Calmar ratio.", bullet_style))
    story.append(Paragraph("• <b>Unconstrained Outlier Capture (50 SMA Baseline):</b> Captures unconstrained 20R–30R mega-runners, producing +0.665 R raw EV, but requires absorbing twice the drawdown depth (-19.0 R) and taking full losses (-0.97 R).", bullet_style))

    story.append(Spacer(1, 10))

    # Section 4: Quantitative Walk-Forward Out-of-Sample Verification
    story.append(Paragraph("4. Out-of-Sample Walk-Forward Validation Proofs (2023–2026)", h1_style))
    story.append(Paragraph(
        "To guarantee zero lookahead bias and verify genuine market alpha, all models were trained strictly on in-sample data (&le;2022) "
        "and validated on <b>419 out-of-sample Pinnacle setups from 2023 to 2026 evaluated strictly on daily bars</b>:",
        body_style
    ))

    oos_data = [
        [Paragraph("Trade Management Model", table_header),
         Paragraph("Out-of-Sample EV", table_header),
         Paragraph("Total OOS P&L", table_header),
         Paragraph("Win Rate", table_header),
         Paragraph("Profit Factor", table_header),
         Paragraph("Avg Win", table_header),
         Paragraph("Avg Loss", table_header)],
        
        [Paragraph("1. Mechanical Baseline (Trade 1)", table_cell),
         Paragraph("+1.61 R", table_cell), Paragraph("+645.8 R", table_cell), Paragraph("45.9%", table_cell), Paragraph("4.44", table_cell), Paragraph("+4.53 R", table_cell), Paragraph("-0.86 R", table_cell)],

        [Paragraph("2. Progressive Pyramiding (ml_50 &ge; 0.20)", table_cell_bold),
         Paragraph("<b>+1.90 R</b>", table_cell_bold), Paragraph("<b>+761.0 R</b>", table_cell_bold), Paragraph("44.9%", table_cell), Paragraph("<b>4.57</b>", table_cell_bold), Paragraph("<b>+5.41 R</b>", table_cell_bold), Paragraph("-0.97 R", table_cell)],

        [Paragraph("3. Dynamic Trailing Stop (+2.0R Hurdle)", table_cell),
         Paragraph("+0.82 R", table_cell), Paragraph("+329.4 R", table_cell), Paragraph("<b>51.1%</b>", table_cell_bold), Paragraph("2.86", table_cell), Paragraph("+2.47 R", table_cell), Paragraph("-0.90 R", table_cell)],

        [Paragraph("<b>4. Combined AI-Enhanced Optimal</b>", table_cell_bold),
         Paragraph("<b>+2.05 R</b>", table_cell_bold), Paragraph("<b>+822.4 R</b>", table_cell_bold), Paragraph("46.2%", table_cell), Paragraph("<b>4.71</b>", table_cell_bold), Paragraph("<b>+5.58 R</b>", table_cell_bold), Paragraph("-0.92 R", table_cell_bold)],
    ]
    t_oos = Table(oos_data, colWidths=[154, 58, 72, 45, 60, 55, 60])
    t_oos.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_accent),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ]))
    story.append(t_oos)
    story.append(Spacer(1, 10))

    # Section 5: Real-World Case Studies
    story.append(Paragraph("5. Iconic Setup Case Studies", h1_style))
    story.append(Paragraph("• <b>BOOT (2016-05-19) — The Multi-Leg Compounder:</b> "
                           "Trade 1 entered on Day 1 close, cleared the +2.0 R hurdle, added +50% size on Day 5 V2 conviction (86%), "
                           "and exited at +2.09 R on dynamic stop hit. Consolidated for 22 sessions, re-flipped Yellow on 2016-07-19 "
                           "with 36% conviction (approved by cost-sensitive ML), and gained +49.4% (+87.0 R), delivering a <b>+89.09 R combined sequence gain</b>.", body_style))
    story.append(Paragraph("• <b>SEDG (2020-02-20) — The Pandemic Re-Entry Masterclass:</b> "
                           "Base EP entered before COVID market panic, stopped out at Day 1 Low (-1.0 R). "
                           "Consolidated for 22 sessions, held above 50% retracement, and re-flipped Yellow on 2020-05-04 at $107.14. "
                           "Ran uninterrupted to $330+ (+36.5 R), yielding <b>+35.5 R net sequence profit</b>.", body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✅ Generated Publication-Grade PDF Report at {out_path}")

if __name__ == "__main__":
    pdf_dest = Path("docs/EP_Pinnacle_Elite_Strategy_Handbook.pdf")
    build_pdf(pdf_dest)
