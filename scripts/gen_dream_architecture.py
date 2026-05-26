"""Generate Image 1: Dream Pass Architecture Diagram"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG      = '#0D1117'
ROW_ALT = '#161B22'

def rbox(ax, cx, cy, w, h, fc, ec, lw=1.8, lines=None,
         fs=8.8, tc='#E6EDF3', bold_first=False, pad=0.12, alpha=1.0):
    patch = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f'round,pad={pad}', fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=3
    )
    ax.add_patch(patch)
    if lines:
        n   = len(lines)
        gap = min(0.215, (h * 0.70) / max(n, 1))
        top = cy + (n - 1) * gap / 2
        for i, ln in enumerate(lines):
            weight = 'bold' if (bold_first and i == 0) else 'normal'
            size   = fs + 0.5 if (bold_first and i == 0) else fs
            ax.text(cx, top - i * gap, ln,
                    ha='center', va='center',
                    fontsize=size, color=tc, fontweight=weight, zorder=4)


def arrow(ax, x1, y1, x2, y2, color='#30363D', lw=1.8, ms=13):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=ms), zorder=5)


fig, ax = plt.subplots(figsize=(16, 11.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 16)
ax.set_ylim(0, 11.5)
ax.axis('off')

# Title
ax.text(8, 11.0, 'AI Agent Memory  +  Dream Pass Architecture',
        ha='center', va='center', fontsize=18, color='#F0F6FC', fontweight='bold', zorder=4)
ax.text(8, 10.55, 'How Turtle bridges the session-reset gap  *  3-stage extraction pipeline',
        ha='center', va='center', fontsize=10.5, color='#6E7681', zorder=4)
ax.plot([0.4, 15.6], [10.18, 10.18], color='#21262D', lw=1.2)

# ── LEFT: 3 Memory Layers ─────────────────────────────────────────────────
ax.text(2.85, 9.88, '3 MEMORY LAYERS', ha='center', fontsize=8.2,
        color='#484F58', fontweight='bold', zorder=4)

rbox(ax, 2.85, 9.15, 5.0, 0.78,
     '#0D2137', '#388BFD', lw=2.2,
     lines=['LAYER 1  --  Working Memory', 'Active context window  (token-limited, ephemeral)'],
     fs=8.8, tc='#79C0FF', bold_first=True)
ax.text(2.85, 8.62, '[!]  Resets on every session end  --  the core problem',
        ha='center', fontsize=7.8, color='#E3B341', zorder=4, style='italic')

rbox(ax, 2.85, 8.0, 5.0, 0.78,
     '#0A2A1B', '#3FB950', lw=2.2,
     lines=['LAYER 2  --  Episodic / External Memory', 'Retrieved at session start  (RAG  *  topic files)'],
     fs=8.8, tc='#7EE787', bold_first=True)
ax.text(2.85, 7.47, '[+]  Dream Pass writes here  --  persists across sessions',
        ha='center', fontsize=7.8, color='#3FB950', zorder=4, style='italic')

rbox(ax, 2.85, 6.85, 5.0, 0.78,
     '#1A1A22', '#484F58', lw=1.5,
     lines=['LAYER 3  --  Semantic / Weights', 'Knowledge baked into model  (fine-tune only)'],
     fs=8.8, tc='#8B949E', bold_first=True)
ax.text(2.85, 6.32, '[~]  Not modified by Dream Pass  --  too slow / expensive',
        ha='center', fontsize=7.8, color='#484F58', zorder=4, style='italic')

# Bidirectional Layer1 <-> Layer2
ax.annotate('', xy=(2.85, 8.40), xytext=(2.85, 8.63),
            arrowprops=dict(arrowstyle='<->', color='#388BFD',
                            lw=1.6, mutation_scale=12), zorder=5)
ax.text(4.0, 8.52, 'injected at\nsession start',
        ha='left', va='center', fontsize=7.2, color='#58A6FF', zorder=5)

# Dream Pass -> Layer 2 curved arrow
ax.annotate('', xy=(5.35, 7.98), xytext=(6.55, 4.62),
            arrowprops=dict(arrowstyle='->', color='#39D353', lw=2.6,
                            mutation_scale=16,
                            connectionstyle='arc3,rad=-0.25'), zorder=5)
ax.text(5.15, 6.28, 'Dream Pass\nwrites here',
        ha='center', va='center', fontsize=8.5, color='#39D353',
        zorder=5, fontweight='bold')

# ── CENTER: Pipeline ──────────────────────────────────────────────────────
CX = 9.0

ax.text(CX, 9.88, 'EXTRACTION PIPELINE', ha='center', fontsize=8.2,
        color='#484F58', fontweight='bold', zorder=4)

rbox(ax, CX, 9.18, 3.8, 0.68,
     '#1C2128', '#30363D', lw=1.5,
     lines=['INPUT -- User Conversation', 'Messages captured per turn'],
     fs=8.8, tc='#8B949E', bold_first=True)
arrow(ax, CX, 8.84, CX, 8.52, '#30363D')

rbox(ax, CX, 8.18, 3.8, 0.68,
     '#0D2137', '#388BFD', lw=2.0,
     lines=['STAGE A  --  Deterministic', 'Regex + rule-based  *  runs on every turn'],
     fs=8.8, tc='#79C0FF', bold_first=True)
ax.text(CX + 2.06, 8.18, 'fast / free', ha='left', va='center',
        fontsize=7.5, color='#388BFD', zorder=5, style='italic')
arrow(ax, CX, 7.84, CX, 7.52, '#30363D')

rbox(ax, CX, 7.18, 3.8, 0.68,
     '#21134B', '#8957E5', lw=2.0,
     lines=['STAGE B  --  LLM Turn Extractor', 'Llama-3.1-8b  *  runs at session end'],
     fs=8.8, tc='#D2A8FF', bold_first=True)
ax.text(CX + 2.06, 7.18, 'cheap / fast', ha='left', va='center',
        fontsize=7.5, color='#8957E5', zorder=5, style='italic')
arrow(ax, CX, 6.84, CX, 6.52, '#30363D')

rbox(ax, CX, 6.18, 3.8, 0.68,
     '#2D1B00', '#D29922', lw=2.0,
     lines=['CONFIRMATION GATE', 'Pending candidates  (applied = False)'],
     fs=8.8, tc='#E3B341', bold_first=True)
ax.text(CX + 2.06, 6.18, 'queue', ha='left', va='center',
        fontsize=7.5, color='#D29922', zorder=5, style='italic')

arrow(ax, CX, 5.84, CX, 5.22, '#39D353', lw=2.5, ms=16)

# Dream Pass hero
rbox(ax, CX, 4.72, 4.25, 1.0,
     '#041A0D', '#39D353', lw=3.2, pad=0.18,
     lines=['** STAGE C  --  DREAM PASS **',
            'Llama-3.3-70b  *  async  *  batch LLM review',
            'Decides:  PROMOTE  or  DROP  each candidate'],
     fs=9.2, tc='#7EE787', bold_first=True)

arrow(ax, CX, 4.22, CX, 3.84, '#30363D')

rbox(ax, CX, 3.50, 3.8, 0.68,
     '#2D0000', '#F85149', lw=2.0,
     lines=['APPEND-ONLY JOURNAL  (.jsonl)', 'Every decision permanent & auditable'],
     fs=8.8, tc='#FF7B72', bold_first=True)
arrow(ax, CX, 3.16, CX, 2.84, '#30363D')

rbox(ax, CX, 2.50, 3.8, 0.68,
     '#0D2130', '#1F6FEB', lw=2.0,
     lines=['REPLAYER  --  Deterministic', 'Projects events --> human-readable markdown'],
     fs=8.8, tc='#79C0FF', bold_first=True)
arrow(ax, CX, 2.16, CX, 1.84, '#30363D')

rbox(ax, CX, 1.50, 3.8, 0.68,
     '#0A2A1B', '#3FB950', lw=2.5,
     lines=['TOPIC FILES  (Output  -->  Layer 2)',
            'identity  *  preferences  *  workflow  *  contacts  *  projects'],
     fs=8.8, tc='#7EE787', bold_first=True)

# ── RIGHT: Metadata ───────────────────────────────────────────────────────
RX = 13.4

ax.text(RX, 9.88, 'TRIGGERS & SAFETY', ha='center', fontsize=8.2,
        color='#484F58', fontweight='bold', zorder=4)

rbox(ax, RX, 8.48, 4.4, 2.15,
     '#041A0D', '#39D353', lw=2.0,
     lines=['DREAM PASS TRIGGERS',
            '------------------------------------',
            '(1)  >= 3 pending candidates',
            '           OR',
            '(2)  >= 24 h since last pass',
            '      +  >= 1 candidate waiting',
            '',
            'Runs async via asyncio.create_task'],
     fs=8.5, tc='#7EE787', bold_first=True)

rbox(ax, RX, 6.40, 4.4, 1.88,
     '#2D0000', '#F85149', lw=2.0,
     lines=['SAFETY MECHANISMS',
            '------------------------------------',
            '(1)  Snapshot before run',
            '(2)  Sanity check  (shrink < 30%)',
            '(3)  Full rollback on failure'],
     fs=8.5, tc='#FF7B72', bold_first=True)

rbox(ax, RX, 4.65, 4.4, 1.88,
     '#0D1F38', '#388BFD', lw=2.0,
     lines=['MEMORY EVENT SCHEMA',
            '------------------------------------',
            'kind  *  topic  *  key  *  value',
            'confidence  *  source',
            'extractor: "dream_pass"'],
     fs=8.5, tc='#79C0FF', bold_first=True)

rbox(ax, RX, 2.82, 4.4, 1.68,
     '#0A2A1B', '#3FB950', lw=2.0,
     lines=['DECAY RULES',
            '------------------------------------',
            'identity  -->  never expires',
            'all others  -->  30-day decay',
            'migration events  -->  permanent'],
     fs=8.5, tc='#7EE787', bold_first=True)

rbox(ax, RX, 1.50, 4.4, 0.82,
     '#1C2128', '#484F58', lw=1.5,
     lines=['TURTLE_PERSONAL_MEMORY_DREAM_PASS_ENABLED=1   (default OFF)'],
     fs=7.9, tc='#8B949E')

# connector arrow: Dream Pass box -> Triggers box
ax.annotate('', xy=(11.15, 7.55), xytext=(11.13, 4.75),
            arrowprops=dict(arrowstyle='->', color='#39D353', lw=1.5,
                            mutation_scale=11), zorder=5)

# footer
ax.plot([0.4, 15.6], [0.48, 0.48], color='#21262D', lw=1.2)
ax.text(8, 0.25,
        'Turtle Voice  *  Stage C (Dream Pass) is disabled by default  *  '
        'TURTLE_PERSONAL_MEMORY_DREAM_PASS_ENABLED=1  to enable',
        ha='center', fontsize=7.5, color='#30363D', zorder=4)

plt.tight_layout(pad=0.4)
plt.savefig('docs/dream_pass_architecture.png', dpi=160,
            bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.close()
print("Image 1 saved -> docs/dream_pass_architecture.png")
