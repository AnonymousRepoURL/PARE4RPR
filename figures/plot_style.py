"""Shared typography at the final 3.5-inch paper column width."""

FIGURE_WIDTH = 3.5
FONT_SIZES = {
    "values": 6,
    "system": 6,
    "dataset": 6,
    "ticks": 6,
    "axis": 6,
    "legend": 6,
    "title": 6,
}


def legend_bottom(fig, legend):
    """Measure the legend, using one column if full names exceed the width."""
    fig.canvas.draw()
    bounds = legend.get_window_extent().transformed(fig.transFigure.inverted())
    if bounds.x0 < 0.01 or bounds.x1 > 0.99:
        handles = legend.legend_handles
        labels = [label.get_text() for label in legend.get_texts()]
        legend.remove()
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=1,
            prop={"size": FONT_SIZES["legend"], "weight": "bold"},
            frameon=False,
            borderaxespad=0,
            handlelength=1.6,
            handletextpad=0.4,
            labelspacing=0.4,
        )
        fig.canvas.draw()
        bounds = legend.get_window_extent().transformed(fig.transFigure.inverted())
    if bounds.x0 < 0 or bounds.x1 > 1:
        raise ValueError("Legend exceeds the 3.5-inch figure width")
    return bounds.y0
