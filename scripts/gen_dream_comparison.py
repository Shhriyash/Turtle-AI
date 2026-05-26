"""Generate Image 2: Claude Dreaming vs Turtle Dream Pass -- Comparison Table"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG = '#0D1117'

ROWS = [
    ('What it reviews',
     'Agentic tool / task patterns\nFiletype quirks, tool workarounds\nProcedural operational memory',
     'Personal facts about YOU\nName, preferences, routines\nContacts, projects, corrections'),
    ('Schema',
     'Unstructured / opaque\nNo public schema\n"Learned intuition" compression',
     'Strictly typed events\nkind  *  topic  *  key  *  value\nconfidence  *  extractor'),
    ('Audit trail',
     'None visible\nOpaque internal state\nCannot be inspected',
     'Append-only JSONL journal\nEvery decision permanent\n& fully inspectable'),
    ('Rollback',
     'None\nNo undo mechanism',
     'Snapshot before run\nSanity checks  (shrink < 30%)\nContradiction-event rollback'),
    ('Trigger',
     'Anthropic-managed\nContinuous / always-on\nNo user control',
     '>= 3 pending candidates\nOR >= 24 h elapsed + >= 1 pending\nasyncio.create_task  (non-blocking)'),
    ('User control',
     'Opaque\nCannot disable or inspect',
     'TURTLE_PERSONAL_MEMORY_\nDREAM_PASS_ENABLED=1\nOFF by default'),
    ('Memory decay',
     'Unknown / unspecified',
     '30-day decay on non-identity\nIdentity facts never expire\nMigration events permanent'),
    ('Who benefits',
     "The agent's operational skills\nHow to use tools better",
     "The user's personal continuity\nContext carries across every session"),
]

CLAUDE_DARK   = '#21134B'
CLAUDE_BORDER = '#8957E5'
CLAUDE_TEXT   = '#D2A8FF'
CLAUDE_HEAD   = '#B39DFF'

TURTLE_DARK   = '#0A2A1B'
TURTLE_BORDER = '#3FB950'
TURTLE_TEXT   = '#7EE787'
TURTLE_HEAD   = '#56D364'

DIM_DARK   = '#1C2128'
DIM_BORDER = '#21262D'
DIM_TEXT   = '#C9D1D9'

ROW_ALT = '#161B22'

FIG_W, FIG_H = 16, 14.2
COL_L  = [0.55, 5.25, 10.65]   # left edge of each column
COL_W  = [4.45, 5.15, 5.15]
ROW_H  = 1.37
HDR_H  = 1.05
TOP_Y  = FIG_H - 2.15

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')

# Title
ax.text(FIG_W / 2, FIG_H - 0.60,
        'Claude Dreaming  vs  Turtle Dream Pass',
        ha='center', va='center', fontsize=18.5, color='#F0F6FC', fontweight='bold', zorder=4)
ax.text(FIG_W / 2, FIG_H - 1.10,
        'Same concept. Very different philosophy.',
        ha='center', va='center', fontsize=12, color='#6E7681', zorder=4)
ax.plot([0.4, FIG_W - 0.4], [FIG_H - 1.48, FIG_H - 1.48], color='#21262D', lw=1.2)

# Column headers
hy = TOP_Y + HDR_H / 2

# Dimension header
ax.add_patch(FancyBboxPatch(
    (COL_L[0] - 0.12, TOP_Y - 0.05), COL_W[0] + 0.12, HDR_H + 0.05,
    boxstyle='round,pad=0.10', fc=DIM_DARK, ec=DIM_BORDER, lw=1.5, zorder=3))
ax.text(COL_L[0] + COL_W[0] / 2, hy, 'DIMENSION',
        ha='center', va='center', fontsize=10.5, color='#6E7681', fontweight='bold', zorder=4)

# Claude header
ax.add_patch(FancyBboxPatch(
    (COL_L[1] - 0.08, TOP_Y - 0.05), COL_W[1], HDR_H + 0.05,
    boxstyle='round,pad=0.10', fc=CLAUDE_DARK, ec=CLAUDE_BORDER, lw=2.8, zorder=3))
ax.text(COL_L[1] + COL_W[1] / 2, hy + 0.13, 'Claude Dreaming',
        ha='center', va='center', fontsize=12.5, color=CLAUDE_HEAD, fontweight='bold', zorder=4)
ax.text(COL_L[1] + COL_W[1] / 2, hy - 0.24,
        'Anthropic  *  May 2026  *  Research Preview',
        ha='center', va='center', fontsize=8.5, color='#8957E5', zorder=4)

# Turtle header
ax.add_patch(FancyBboxPatch(
    (COL_L[2] - 0.08, TOP_Y - 0.05), COL_W[2] + 0.20, HDR_H + 0.05,
    boxstyle='round,pad=0.10', fc=TURTLE_DARK, ec=TURTLE_BORDER, lw=2.8, zorder=3))
ax.text(COL_L[2] + (COL_W[2] + 0.20) / 2, hy + 0.13, 'Turtle Dream Pass',
        ha='center', va='center', fontsize=12.5, color=TURTLE_HEAD, fontweight='bold', zorder=4)
ax.text(COL_L[2] + (COL_W[2] + 0.20) / 2, hy - 0.24,
        'Open source  *  Stage C  *  core/dream_pass.py',
        ha='center', va='center', fontsize=8.5, color='#3FB950', zorder=4)

# Data rows
for i, (dim, claude, turtle) in enumerate(ROWS):
    rt   = TOP_Y - (i + 1) * ROW_H
    mid  = rt + ROW_H / 2
    rbg  = ROW_ALT if i % 2 else BG

    # stripe
    ax.add_patch(FancyBboxPatch(
        (0.3, rt + 0.04), FIG_W - 0.6, ROW_H - 0.08,
        boxstyle='round,pad=0.06', fc=rbg, ec='none', lw=0, zorder=2, alpha=0.55))

    # Dimension cell
    ax.add_patch(FancyBboxPatch(
        (COL_L[0] - 0.12, rt + 0.09), COL_W[0], ROW_H - 0.18,
        boxstyle='round,pad=0.08', fc=DIM_DARK, ec=DIM_BORDER, lw=1.2, zorder=3))
    ax.text(COL_L[0] + COL_W[0] / 2, mid, dim,
            ha='center', va='center', fontsize=10.5, color=DIM_TEXT, fontweight='bold', zorder=4)

    # Claude cell
    ax.add_patch(FancyBboxPatch(
        (COL_L[1] - 0.08, rt + 0.09), COL_W[1], ROW_H - 0.18,
        boxstyle='round,pad=0.08', fc=CLAUDE_DARK, ec='#3D2080', lw=1.2, alpha=0.82, zorder=3))
    ax.text(COL_L[1] + COL_W[1] / 2, mid, claude,
            ha='center', va='center', fontsize=9.0, color=CLAUDE_TEXT,
            zorder=4, linespacing=1.5, multialignment='center')

    # Turtle cell
    ax.add_patch(FancyBboxPatch(
        (COL_L[2] - 0.08, rt + 0.09), COL_W[2] + 0.20, ROW_H - 0.18,
        boxstyle='round,pad=0.08', fc=TURTLE_DARK, ec='#1A5928', lw=1.2, alpha=0.82, zorder=3))
    ax.text(COL_L[2] + (COL_W[2] + 0.20) / 2, mid, turtle,
            ha='center', va='center', fontsize=9.0, color=TURTLE_TEXT,
            zorder=4, linespacing=1.5, multialignment='center')

    # divider
    ax.plot([0.4, FIG_W - 0.4], [rt + 0.04, rt + 0.04], color='#21262D', lw=0.8, zorder=2)

# Footer
bot = TOP_Y - len(ROWS) * ROW_H
ax.plot([0.4, FIG_W - 0.4], [bot + 0.05, bot + 0.05], color='#21262D', lw=1.2)
ax.text(FIG_W / 2, bot - 0.23,
        'Harvey (legal AI) enabled Claude Dreaming  -->  ~6x task completion jump  *  '
        'Both solve the same problem with very different levels of transparency & control',
        ha='center', fontsize=8.2, color='#484F58', zorder=4)

plt.tight_layout(pad=0.5)
plt.savefig('docs/dream_comparison_table.png', dpi=160,
            bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.close()
print("Image 2 saved -> docs/dream_comparison_table.png")
