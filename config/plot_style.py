"""Figure settings for the paper figures (08_figures): colours, fonts, sizes and the small helpers the
figure scripts call as `vs.<name>`. A4 print context ("paper_large", 160 mm full width).

Fonts: IBM Plex Sans (SIL Open Font License) is used when available. Put the .ttf files in
config/fonts/ (or point HSP_FONT_DIR at a folder with them); otherwise matplotlib falls back to
DejaVu Sans and the figures differ only in typeface.
"""
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.colors import to_rgb

FONT_DIR = Path(os.environ.get("HSP_FONT_DIR", Path(__file__).resolve().parent / "fonts"))

# ---------------------------------------------------------------------------
# Tailwind CSS colors (v3 hex values)
# ---------------------------------------------------------------------------
TW = {
    "slate-50": "#F8FAFC", "slate-100": "#F1F5F9", "slate-200": "#E2E8F0", "slate-300": "#CBD5E1",
    "slate-400": "#94A3B8", "slate-500": "#64748B", "slate-600": "#475569", "slate-700": "#334155",
    "slate-800": "#1E293B", "slate-900": "#0F172A",
    "stone-500": "#78716C",
    "emerald-500": "#10B981", "emerald-300": "#6EE7B7", "teal-500": "#14B8A6", "red-500": "#EF4444",
    "indigo-400": "#818CF8", "orange-400": "#FB923C", "purple-500": "#A855F7", "purple-600": "#9333EA",
    "fuchsia-500": "#D946EF", "sky-300": "#7DD3FC", "sky-400": "#38BDF8", "pink-400": "#F472B6", "yellow-400": "#FACC15",
    "red-600": "#DC2626", "red-700": "#B91C1C",
    "orange-500": "#F97316", "orange-600": "#EA580C",
    "amber-400": "#FBBF24", "amber-500": "#F59E0B", "amber-600": "#D97706",
    "yellow-500": "#EAB308",
    "lime-600": "#65A30D",
    "emerald-600": "#059669", "emerald-700": "#047857",
    "teal-600": "#0D9488",
    "cyan-500": "#06B6D4", "cyan-700": "#0E7490",
    "sky-500": "#0EA5E9", "sky-600": "#0284C7",
    "blue-50": "#EFF6FF", "blue-100": "#DBEAFE", "blue-200": "#BFDBFE", "blue-300": "#93C5FD",
    "blue-400": "#60A5FA", "blue-500": "#3B82F6", "blue-600": "#2563EB", "blue-700": "#1D4ED8",
    "blue-800": "#1E40AF", "blue-900": "#1E3A8A",
    "indigo-600": "#4F46E5",
    "violet-600": "#7C3AED", "violet-700": "#6D28D9",
    "fuchsia-600": "#C026D3",
    "pink-500": "#EC4899",
    "rose-100": "#FFE4E6", "rose-300": "#FDA4AF", "rose-400": "#FB7185", "rose-600": "#E11D48",
    "rose-700": "#BE123C",
    "teal-100": "#CCFBF1", "teal-300": "#5EEAD4", "teal-700": "#0F766E",
    "orange-100": "#FFEDD5", "orange-300": "#FDBA74", "orange-700": "#C2410C",
}

# Model palette: [0] = main model, [1:] = comparisons, last = neutral baseline.
PALETTE_KEYS = {
    "final_a": ["red-500", "blue-600", "teal-500", "amber-500", "fuchsia-600", "sky-400", "#11235F", "slate-500"],
}
PALETTES = {k: [TW.get(c, c) for c in v] for k, v in PALETTE_KEYS.items()}
OBS = "#000000"
GRID = TW["slate-200"]
DEFAULT_PALETTE = "final_a"
DEFAULT_CONTEXT = "paper_large"   # A4 print

# Fixed model -> slot mapping so a model keeps its color in every figure.
MODEL_ORDER = ["HyperReg", "LSTM", "BMA", "RF", "XGB", "TCN", "GR4J", "Benchmark"]
_active_palette = PALETTES[DEFAULT_PALETTE]
_active_context = DEFAULT_CONTEXT


def set_models(names):
    """Fix the color order for this figure set: names[0] = main model (palette[0], red), then comparisons.
    Call once per script with the project's model names; unknown names are appended on first use."""
    MODEL_ORDER[:] = list(names) + [m for m in MODEL_ORDER if m not in names]


def _slot(name):
    if name not in MODEL_ORDER:
        MODEL_ORDER.append(name)
    return MODEL_ORDER.index(name)


def model_color(name, palette=None):
    pal = PALETTES[palette] if palette else _active_palette
    return OBS if name in ("Observed", "Obs") else pal[_slot(name) % len(pal)]


# ---------------------------------------------------------------------------
# Axis labels: units in [], unitless metrics have no unit marker
# ---------------------------------------------------------------------------
LABELS = {
    "kge": "KGE", "nse": "NSE", "dkge": r"$\Delta$KGE", "r": "r",
    "q": r"Q [mm d$^{-1}$]", "p": r"P [mm d$^{-1}$]", "pet": r"PET [mm d$^{-1}$]",
    "area": r"Area [km$^2$]", "bias": "Bias [%]", "elev": "Elevation [m]",
    "lat": "Latitude [°]", "lon": "Longitude [°]", "aridity": "Aridity index",
    "cdf": "Cumulative fraction of basins", "count": "Number of basins", "date": "Date",
}

# ---------------------------------------------------------------------------
# Contexts: sizes are physical (figure rendered at final size, never rescaled)
# ---------------------------------------------------------------------------
MM = 1 / 25.4
CONTEXTS = {
    # A4 text width = 210 - 2*25 mm = 160 mm
    "paper_large": dict(tick=8, label=9, legend=8, title=9, panel=10, annot=7,
                        lw=1.0, lw_main=1.6, lw_axes=0.7, ms=3.5, mew=0.35, tick_len=3.5, tick_w=0.7,
                        widths={"full": 160 * MM, "twothirds": 105 * MM, "half": 78 * MM}),
}
for _c in CONTEXTS.values():  # line-width roles: thin = unimportant runs (e.g. all hyperparameter sets),
    _c["lw_thin"], _c["lw_thick"] = _c["lw"] * 0.7, _c["lw"] * 1.5  # thick only on request


def figsize(context=None, width="full", aspect=None, height=None):
    """(w, h) in inches. aspect = h / w."""
    c = CONTEXTS[context or DEFAULT_CONTEXT]
    w = c["widths"][width]
    if height is None:
        height = w * aspect if aspect else c.get("height", w * 0.62)
    return (w, height)


DEFAULT_FONT = "IBM Plex Sans"


def register_fonts():
    for f in sorted(FONT_DIR.glob("*.ttf")):
        fm.fontManager.addfont(str(f))
    # fonts in FONT_DIR win over system fonts of the same family
    cached = {e.name for e in fm.fontManager.ttflist if Path(e.fname).parent == FONT_DIR}
    fm.fontManager.ttflist = [e for e in fm.fontManager.ttflist
                              if e.name not in cached or Path(e.fname).parent == FONT_DIR]
    fm.fontManager._findfont_cached.cache_clear()


def has_glyph(name, char):
    from matplotlib import ft2font
    path = fm.findfont(fm.FontProperties(family=name))
    return ord(char) in ft2font.FT2Font(path).get_charmap()


register_fonts()




# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------
def apply_style(context=None, font=None, palette=None, grid=True):
    """Set rcParams for a context. Returns the context dict (font sizes, line widths)."""
    global _active_palette, _active_context
    _active_context = context or DEFAULT_CONTEXT
    c = CONTEXTS[_active_context]
    font = font or DEFAULT_FONT
    _active_palette = PALETTES[palette or DEFAULT_PALETTE]
    mpl.rcdefaults()
    mpl.rcParams.update({
        "font.family": "sans-serif" if "Serif" not in font else "serif",
        "font.sans-serif": [font, "DejaVu Sans"], "font.serif": [font, "DejaVu Serif"],
        "mathtext.fontset": "custom", "mathtext.rm": font, "mathtext.it": font, "mathtext.cal": font,
        "mathtext.bf": f"{font}:bold", "mathtext.default": "regular", "mathtext.fallback": "cm",
        "axes.unicode_minus": has_glyph(font, "\u2212"),
        "font.size": c["label"], "axes.labelsize": c["label"], "axes.titlesize": c["title"],
        "xtick.labelsize": c["tick"], "ytick.labelsize": c["tick"], "legend.fontsize": c["legend"],
        "figure.titlesize": c["title"] + 2, "axes.titleweight": "normal", "axes.labelweight": "normal",
        "lines.linewidth": c["lw"], "lines.markersize": c["ms"], "patch.linewidth": c["lw_axes"],
        "axes.linewidth": c["lw_axes"], "axes.edgecolor": TW["slate-800"], "axes.labelcolor": "black",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=_active_palette),
        "axes.grid": grid, "grid.color": GRID, "grid.linewidth": c["lw_axes"] * 0.8, "axes.axisbelow": True,
        "xtick.major.size": c["tick_len"], "ytick.major.size": c["tick_len"],
        "xtick.major.width": c["tick_w"], "ytick.major.width": c["tick_w"],
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "legend.handlelength": 1.8, "legend.borderaxespad": 0.3,
        "legend.columnspacing": 1.2, "legend.handletextpad": 0.5,
        "boxplot.medianprops.color": "black", "boxplot.flierprops.markersize": c["ms"] * 0.6,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "figure.dpi": 100, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 2 / 72, "figure.constrained_layout.w_pad": 2 / 72,
        "image.cmap": "viridis",
    })
    return c


PANEL_STYLES = {"paren": "({})", "plain": "{}", "upper": "{}", "upper_paren": "({})"}


def panel_label(ax, i, style="plain", loc="outside", bold=True, title=None, size=None, title_loc="inline"):
    """Panel label. loc='outside': left-aligned above the axes (as title, layout-aware);
    loc='inside': upper-left corner inside the axes. title: optional panel title, either
    title_loc='inline' ('(a) Title' left-aligned, never collides) or 'center'."""
    if title and loc == "outside" and title_loc == "inline":
        return inline_title(ax, i, title, style=style, bold=bold)
    letter = "abcdefghijklmnopqrstuvwxyz"[i]
    letter = letter.upper() if style.startswith("upper") else letter
    lab = PANEL_STYLES[style].format(letter)
    size = size or mpl.rcParams["axes.titlesize"] + 1
    weight = "bold" if bold else "normal"
    if loc == "outside":
        if title:
            ax.set_title(f"{lab}  ", loc="left", fontweight=weight, fontsize=size)
            ax.set_title(title, loc="center", fontsize=mpl.rcParams["axes.titlesize"])
        else:
            ax.set_title(lab, loc="left", fontweight=weight, fontsize=size)
    else:
        txt = ax.text(0.03, 0.97, lab, transform=ax.transAxes, ha="left", va="top",
                      fontweight=weight, fontsize=size, zorder=10)
        if title:
            ax.set_title(title, fontsize=mpl.rcParams["axes.titlesize"])
        return txt


def inline_title(ax, lab_i, title, style="plain", bold=True):
    """'(a) Title' in one left-aligned line above the axes, label bold, title regular."""
    letter = "abcdefghijklmnopqrstuvwxyz"[lab_i]
    lab = PANEL_STYLES[style].format(letter.upper() if style.startswith("upper") else letter)
    lab = r"$\bf{" + lab + "}$" if bold else lab
    ax.set_title(f"{lab} {title}", loc="left")


PNG = ("png",)


def savefig(fig, stem, outdir, formats=PNG, dpi=300, tight=True, transparent=False):
    """Save PNG on a white background (transparent=True only on request); formats=("png", "pdf") adds a PDF.
    tight=True trims outer whitespace (pad 0.02 in); tight=False keeps the exact size."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    kw = dict(bbox_inches="tight", pad_inches=0.02) if tight else {}
    kw.update(transparent=True) if transparent else kw.update(facecolor="white")
    paths = []
    for ext in formats:
        p = outdir / f"{stem}.{ext}"
        # the PDF canvas re-reads this rcParam; figures with an engine already keep theirs
        with mpl.rc_context({"figure.constrained_layout.use": False}):
            fig.savefig(p, dpi=dpi, **kw)
        paths.append(p)
    plt.close(fig)
    return paths


def shades(color, n, darken=0.45, min_alpha=0.45):
    """n variants of one model color for gradient lines: [0] = the color itself (current / main, opaque),
    later ones progressively darker and more transparent. Returns [(rgba), ...]."""
    rgb = np.array(to_rgb(color))
    out = []
    for k in range(n):
        f = k / max(n - 1, 1)
        out.append((*(rgb * (1 - darken * f)), 1 - (1 - min_alpha) * f))
    return out
