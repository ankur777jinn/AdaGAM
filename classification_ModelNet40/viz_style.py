"""Shared matplotlib style for all figures (validated categorical palette, one-hue sequential ramp,
recessive axes). Colours follow the entity (run name), never its rank."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#ffffff"
TEXT = "#191919"
TEXT_2 = "#666666"
GRID = "#e6e6e6"
NEUTRAL = "#9c9c9c"

# categorical slots, fixed order (validated palette; adjacent-pair CVD-safe in this order)
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
# run name -> slot, so a run keeps its colour whichever runs are present
RUN_SLOT = {"base": 0, "adagam": 1, "base_aug": 2, "adagam_aug": 3}
SEQ_BLUE = ["#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)


def run_tag(run_name: str) -> str:
    """'pointMLPElite-adagam_aug-42' -> 'adagam_aug'."""
    parts = run_name.split("-")
    return "-".join(parts[1:-1]) if len(parts) >= 3 else run_name


def run_color(run_name: str, fallback_index: int = 0) -> str:
    tag = run_tag(run_name)
    return SERIES[RUN_SLOT.get(tag, 4 + fallback_index % 4)]


def run_linestyle(gam: str) -> str:
    return "-" if gam == "adagam" else "--"     # secondary encoding: identity is never colour alone


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "text.color": TEXT,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
        "lines.linewidth": 2.0, "lines.markersize": 5,
    })
