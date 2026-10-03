from __future__ import annotations

"""Neutral research-style plotting helpers (vendored from the visualization-design skill; the binding of its DESIGN.md).

This module intentionally keeps the style layer small:
- white canvas
- muted text + grid
- soft gray default surfaces
- restrained accent colors
- optional rounded bar ends

The goal is not to replace normal Matplotlib usage. The goal is to let agents
keep normal plotting code and add only the minimum style needed to land in the
same visual family as the neutral diagram examples.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
try:  # only the demo plot helpers need pandas
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

# ---------------------------------------------------------------------------
# Design tokens
#
# These constants are this module's binding of the Editorial Neutral design
# system. ../DESIGN.md is the source of truth; change it there first.
# ---------------------------------------------------------------------------

CANVAS = "#ffffff"
TEXT = "#171717"
MUTED = "#5a5a5a"
SUBTLE = "#8a8a8a"
GRID = "#ececec"
SPINE = "#a8a8a8"
REGION = "#fafafa"


@dataclass(frozen=True)
class Tone:
    fill: str
    edge: str


GRAY = Tone(fill="#f5f5f5", edge="#b3b3b3")
GREEN = Tone(fill="#edf3e8", edge="#8a9f7a")
PINK = Tone(fill="#f5e7ec", edge="#bd7c8f")
BLUE = Tone(fill="#e8f0f9", edge="#7f9fc4")

GRAY_FILL, GRAY_EDGE = GRAY.fill, GRAY.edge
GREEN_FILL, GREEN_EDGE = GREEN.fill, GREEN.edge
PINK_FILL, PINK_EDGE = PINK.fill, PINK.edge
BLUE_FILL, BLUE_EDGE = BLUE.fill, BLUE.edge

PALETTE: dict[str, Tone] = {
    "gray": GRAY,
    "green": GREEN,
    "pink": PINK,
    "blue": BLUE,
}


# ---------------------------------------------------------------------------
# Base style
# ---------------------------------------------------------------------------


def use_diagram_chart_style() -> None:
    """Apply the shared neutral style with light-touch rc overrides."""
    mpl.rcParams.update(
        {
            "figure.facecolor": CANVAS,
            "savefig.facecolor": CANVAS,
            "axes.facecolor": CANVAS,
            "axes.edgecolor": SPINE,
            "axes.labelcolor": MUTED,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.85,
            "grid.alpha": 1.0,
            "axes.grid": False,
            "axes.axisbelow": True,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Inter",
                "Clear Sans",
                "Noto Sans",
                "Helvetica Neue",
                "Arial",
                "Noto Sans CJK JP",
                "DejaVu Sans",
            ],
            "axes.titlesize": 12.5,
            "axes.titleweight": "normal",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 9.5,
            "svg.fonttype": "none",
        }
    )


def make_figure(figsize: tuple[float, float] = (9.2, 4.8)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize, dpi=180, layout="constrained")
    fig.set_facecolor(CANVAS)
    ax.set_facecolor(CANVAS)
    return fig, ax


def make_subplots(
    nrows: int,
    ncols: int,
    *,
    figsize: tuple[float, float],
) -> tuple[plt.Figure, list[list[plt.Axes]]]:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=180, layout="constrained")
    fig.set_facecolor(CANVAS)
    if nrows == 1 and ncols == 1:
        axes_grid = [[axes]]
    elif nrows == 1:
        axes_grid = [list(axes)]
    elif ncols == 1:
        axes_grid = [[ax] for ax in axes]
    else:
        axes_grid = [list(row) for row in axes]
    for row in axes_grid:
        for ax in row:
            ax.set_facecolor(CANVAS)
    return fig, axes_grid


def color_pair(name: str = "blue") -> tuple[str, str]:
    try:
        tone = PALETTE[name]
    except KeyError as exc:
        valid = ", ".join(sorted(PALETTE))
        raise ValueError(f"Unknown palette key '{name}'. Valid keys: {valid}") from exc
    return tone.fill, tone.edge


# ---------------------------------------------------------------------------
# Axes styling
# ---------------------------------------------------------------------------


def style_axes(
    ax: plt.Axes,
    *,
    grid_axis: str = "y",
    show_left_axis: bool = False,
    show_bottom_axis: bool = True,
) -> None:
    """Keep layout normal, only soften the axis treatment."""
    ax.grid(False)
    ax.set_axisbelow(True)

    if grid_axis in {"x", "both"}:
        ax.xaxis.grid(True)
    if grid_axis in {"y", "both"}:
        ax.yaxis.grid(True)

    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)

    if show_left_axis:
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(SPINE)
        ax.spines["left"].set_linewidth(0.9)

    if show_bottom_axis:
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(SPINE)
        ax.spines["bottom"].set_linewidth(0.9)

    ax.tick_params(axis="both", length=0, pad=6)


def set_chart_title(
    ax: plt.Axes,
    title: str,
    subtitle: str | None = None,
    *,
    title_pad: float = 30.0,
    subtitle_y: float = 1.01,
    title_size: float = 13.0,
    subtitle_size: float = 9.5,
) -> tuple[Text, Text | None]:
    """Set a left-aligned chart title and optional subtitle with safe spacing.

    Use this instead of mixing ``ax.set_title(...)`` and ad hoc ``ax.text(...)``
    calls. The title gets enough pad to sit above the subtitle; the returned
    artists are tagged so ``assert_title_subtitle_clear`` and ``save_figure``
    can catch accidental collisions.
    """
    title_artist = ax.set_title(
        title,
        loc="left",
        pad=title_pad if subtitle else min(title_pad, 12.0),
        color=TEXT,
        fontsize=title_size,
    )
    subtitle_artist = None
    if subtitle:
        subtitle_artist = ax.text(
            0.0,
            subtitle_y,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=subtitle_size,
            color=MUTED,
        )
    ax._editorial_title_text = title_artist  # type: ignore[attr-defined]
    ax._editorial_subtitle_text = subtitle_artist  # type: ignore[attr-defined]
    return title_artist, subtitle_artist


def assert_title_subtitle_clear(ax: plt.Axes, *, min_gap_px: float = 2.0) -> None:
    """Raise if an editorial title/subtitle pair overlaps or nearly touches."""
    title_artist = getattr(ax, "_editorial_title_text", None)
    subtitle_artist = getattr(ax, "_editorial_subtitle_text", None)
    if title_artist is None or subtitle_artist is None:
        return

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_box = title_artist.get_window_extent(renderer=renderer)
    subtitle_box = subtitle_artist.get_window_extent(renderer=renderer)
    vertical_gap = title_box.y0 - subtitle_box.y1
    if title_box.overlaps(subtitle_box) or vertical_gap < min_gap_px:
        raise AssertionError(
            "Chart title and subtitle overlap or are too close. "
            "Use set_chart_title with more title_pad or a lower subtitle_y."
        )


def assert_figure_titles_clear(fig: plt.Figure, *, min_gap_px: float = 2.0) -> None:
    """Check every axes using ``set_chart_title`` for title/subtitle overlap."""
    for ax in fig.axes:
        assert_title_subtitle_clear(ax, min_gap_px=min_gap_px)


def _pixels_to_data(ax: plt.Axes, dx_px: float, dy_px: float) -> tuple[float, float]:
    inv = ax.transData.inverted()
    x0, y0 = inv.transform((0.0, 0.0))
    x1, y1 = inv.transform((dx_px, dy_px))
    return abs(x1 - x0), abs(y1 - y0)


def _path_round_right(x: float, y: float, width: float, height: float, rx: float, ry: float) -> MplPath:
    """Horizontal bar: flat left stem, rounded right end."""
    rx = min(rx, width / 2.0)
    ry = min(ry, height / 2.0)
    verts = [
        (x, y),
        (x + width - rx, y),
        (x + width, y),
        (x + width, y + ry),
        (x + width, y + height - ry),
        (x + width, y + height),
        (x + width - rx, y + height),
        (x, y + height),
        (x, y),
        (x, y),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    return MplPath(verts, codes)


def _path_round_left(x: float, y: float, width: float, height: float, rx: float, ry: float) -> MplPath:
    """Horizontal bar: rounded left end, flat right stem."""
    rx = min(rx, width / 2.0)
    ry = min(ry, height / 2.0)
    verts = [
        (x + rx, y),
        (x + width, y),
        (x + width, y + height),
        (x + rx, y + height),
        (x, y + height),
        (x, y + height - ry),
        (x, y + ry),
        (x, y),
        (x + rx, y),
        (x + rx, y),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.CLOSEPOLY,
    ]
    return MplPath(verts, codes)


def _path_round_top(x: float, y: float, width: float, height: float, rx: float, ry: float) -> MplPath:
    """Vertical bar: flat bottom stem, rounded top end."""
    rx = min(rx, width / 2.0)
    ry = min(ry, height / 2.0)
    verts = [
        (x, y),
        (x, y + height - ry),
        (x, y + height),
        (x + rx, y + height),
        (x + width - rx, y + height),
        (x + width, y + height),
        (x + width, y + height - ry),
        (x + width, y),
        (x, y),
        (x, y),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    return MplPath(verts, codes)


def _path_round_bottom(x: float, y: float, width: float, height: float, rx: float, ry: float) -> MplPath:
    """Vertical bar: rounded bottom end, flat top stem."""
    rx = min(rx, width / 2.0)
    ry = min(ry, height / 2.0)
    verts = [
        (x + rx, y),
        (x + width - rx, y),
        (x + width, y),
        (x + width, y + ry),
        (x + width, y + height),
        (x, y + height),
        (x, y + ry),
        (x, y),
        (x + rx, y),
        (x + rx, y),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.CLOSEPOLY,
    ]
    return MplPath(verts, codes)


def _replace_bar_patches(
    container: BarContainer,
    *,
    ax: plt.Axes,
    fill: str,
    edge: str,
    linewidth: float,
    rounding_size: float,
    orientation: str,
) -> None:
    """Replace rectangular bar patches with rounded far-end bars.

    The bar side that grows away from the axis is rounded; the stem side stays
    flat. This matches the diagram-like look more closely than fully rounded
    rectangles.
    """
    rx_data, ry_data = _pixels_to_data(ax, rounding_size, rounding_size)

    for rect in container.patches:
        x0, y0 = rect.get_x(), rect.get_y()
        width, height = rect.get_width(), rect.get_height()
        left = min(x0, x0 + width)
        bottom = min(y0, y0 + height)
        width_abs = abs(width)
        height_abs = abs(height)

        if width_abs == 0 or height_abs == 0:
            rect.set_visible(False)
            continue

        if orientation == "horizontal":
            if width >= 0:
                path = _path_round_right(left, bottom, width_abs, height_abs, rx_data, ry_data)
            else:
                path = _path_round_left(left, bottom, width_abs, height_abs, rx_data, ry_data)
        elif orientation == "vertical":
            if height >= 0:
                path = _path_round_top(left, bottom, width_abs, height_abs, rx_data, ry_data)
            else:
                path = _path_round_bottom(left, bottom, width_abs, height_abs, rx_data, ry_data)
        else:
            raise ValueError("orientation must be 'horizontal' or 'vertical'")

        rounded = PathPatch(
            path,
            facecolor=fill,
            edgecolor=edge,
            linewidth=linewidth,
            transform=ax.transData,
            zorder=rect.get_zorder(),
            clip_on=True,
            joinstyle="round",
        )
        ax.add_patch(rounded)
        rect.set_visible(False)


def soften_barh(
    container: BarContainer,
    ax: plt.Axes,
    *,
    fill: str = BLUE_FILL,
    edge: str = BLUE_EDGE,
    linewidth: float = 0.9,
    rounding_size: float = 6.0,
) -> None:
    _replace_bar_patches(
        container,
        ax=ax,
        fill=fill,
        edge=edge,
        linewidth=linewidth,
        rounding_size=rounding_size,
        orientation="horizontal",
    )


def soften_bar(
    container: BarContainer,
    ax: plt.Axes,
    *,
    fill: str = BLUE_FILL,
    edge: str = BLUE_EDGE,
    linewidth: float = 0.9,
    rounding_size: float = 6.0,
) -> None:
    _replace_bar_patches(
        container,
        ax=ax,
        fill=fill,
        edge=edge,
        linewidth=linewidth,
        rounding_size=rounding_size,
        orientation="vertical",
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def editorial_barh(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    palette: str = "blue",
    show_labels: bool = True,
    label_fmt=str,
    axis_padding: float | None = None,
    grid_axis: str = "x",
    show_left_axis: bool = False,
    show_bottom_axis: bool = True,
    rounding_size: float = 6.0,
    **bar_kwargs,
) -> BarContainer:
    fill, edge = color_pair(palette)
    defaults = {"linewidth": 0.9, "height": 0.66, "color": fill, "edgecolor": edge}
    defaults.update(bar_kwargs)
    container = ax.barh(data[y], data[x], **defaults)
    ax.invert_yaxis()
    max_value = float(pd.Series(data[x]).max())
    padding = axis_padding if axis_padding is not None else max(max_value * 0.10, 1.0)
    ax.set_xlim(0, max_value + padding)
    ax.margins(y=0.08)
    ax.figure.canvas.draw()
    soften_barh(container, ax, fill=fill, edge=edge, linewidth=defaults["linewidth"], rounding_size=rounding_size)
    if show_labels:
        ax.bar_label(
            container,
            labels=[label_fmt(v) for v in data[x]],
            padding=4,
            color=MUTED,
            fontsize=10.0,
        )
    style_axes(ax, grid_axis=grid_axis, show_left_axis=show_left_axis, show_bottom_axis=show_bottom_axis)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    return container


def editorial_bar(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    palette: str = "blue",
    show_labels: bool = True,
    label_fmt=str,
    y_padding: float | None = None,
    grid_axis: str = "y",
    show_left_axis: bool = False,
    show_bottom_axis: bool = True,
    rounding_size: float = 6.0,
    **bar_kwargs,
) -> BarContainer:
    fill, edge = color_pair(palette)
    defaults = {"linewidth": 0.9, "width": 0.68, "color": fill, "edgecolor": edge}
    defaults.update(bar_kwargs)
    container = ax.bar(data[x], data[y], **defaults)
    max_value = float(pd.Series(data[y]).max())
    padding = y_padding if y_padding is not None else max(max_value * 0.12, 1.0)
    ax.set_ylim(0, max_value + padding)
    ax.figure.canvas.draw()
    soften_bar(container, ax, fill=fill, edge=edge, linewidth=defaults["linewidth"], rounding_size=rounding_size)
    if show_labels:
        ax.bar_label(
            container,
            labels=[label_fmt(v) for v in data[y]],
            padding=4,
            color=MUTED,
            fontsize=10.0,
        )
    style_axes(ax, grid_axis=grid_axis, show_left_axis=show_left_axis, show_bottom_axis=show_bottom_axis)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    return container


def editorial_line(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    palette: str = "blue",
    fill_area: bool = True,
    marker_size: float = 24,
    **line_kwargs,
) -> None:
    fill, edge = color_pair(palette)
    defaults = {"linewidth": 1.8, "color": edge, "solid_capstyle": "round"}
    defaults.update(line_kwargs)
    ax.plot(data[x], data[y], **defaults)
    ax.scatter(data[x], data[y], s=marker_size, facecolor=fill, edgecolor=edge, linewidth=0.8, zorder=3)
    if fill_area:
        ax.fill_between(data[x], data[y], color=fill, alpha=1.0, zorder=1)
    style_axes(ax, grid_axis="y", show_left_axis=False, show_bottom_axis=True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def editorial_scatter(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str | None = None,
    palette_map: dict[str, str] | None = None,
    size: float = 40,
) -> None:
    if hue is None:
        fill, edge = color_pair("blue")
        ax.scatter(data[x], data[y], s=size, facecolor=fill, edgecolor=edge, linewidth=0.9)
    else:
        palette_map = palette_map or {}
        ordered_groups = list(dict.fromkeys(data[hue]))
        handles: list[Line2D] = []
        for group in ordered_groups:
            subset = data[data[hue] == group]
            fill, edge = color_pair(palette_map.get(group, "blue"))
            ax.scatter(subset[x], subset[y], s=size, facecolor=fill, edgecolor=edge, linewidth=0.9, zorder=3)
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=fill,
                    markeredgecolor=edge,
                    markeredgewidth=0.9,
                    markersize=6,
                    label=str(group),
                )
            )
        ax.legend(handles=handles, frameon=False, ncol=min(4, len(handles)), loc="upper left")
    style_axes(ax, grid_axis="y", show_left_axis=False, show_bottom_axis=True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))


def editorial_pie(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    values: str,
    labels: str,
    palette_cycle: list[str] | None = None,
    show_legend: bool = True,
) -> None:
    palette_cycle = palette_cycle or ["blue", "green", "pink", "gray"]
    tones = [PALETTE[name] for name in palette_cycle[: len(data)]]
    wedges, _, autotexts = ax.pie(
        data[values],
        labels=None,
        colors=[tone.fill for tone in tones],
        autopct="%1.0f%%",
        startangle=90,
        counterclock=False,
        wedgeprops={"linewidth": 0.9},
        textprops={"color": MUTED, "fontsize": 9.0},
        pctdistance=0.70,
    )
    for wedge, tone in zip(wedges, tones):
        wedge.set_edgecolor(tone.edge)
    for autotext in autotexts:
        autotext.set_color(TEXT)
        autotext.set_fontsize(9.0)
    if show_legend:
        ax.legend(
            wedges,
            [f"{label} {share}%" for label, share in zip(data[labels], data[values])],
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
        )
    ax.set_aspect("equal")


# ---------------------------------------------------------------------------
# Example datasets
# ---------------------------------------------------------------------------


def demo_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": ["Plan", "Build", "Check", "Ship"],
            "value": [37, 116, 389, 486],
        }
    )


def line_demo_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-02-27",
                    "2026-02-28",
                    "2026-03-01",
                    "2026-03-05",
                    "2026-03-06",
                    "2026-03-07",
                    "2026-03-08",
                    "2026-03-09",
                ]
            ),
            "chunks": [37, 389, 116, 35, 455, 486, 458, 386],
        }
    )
    df["cumulative_chunks"] = df["chunks"].cumsum()
    return df


def scatter_demo_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "latency": [110, 145, 180, 205, 240, 275, 320, 360],
            "quality": [58, 63, 68, 71, 79, 84, 87, 91],
            "group": ["Blue", "Blue", "Blue", "Green", "Green", "Pink", "Pink", "Gray"],
        }
    )


def pie_demo_dataframe() -> pd.DataFrame:
    return pd.DataFrame({"segment": ["Output", "Review", "Docs", "Overhead"], "share": [46, 24, 18, 12]})


def _demo_date_ticks(df: pd.DataFrame, column: str) -> pd.Series:
    if len(df) >= 8:
        return df[column].iloc[[0, 1, 2, 4, 6, 7]]
    return df[column]


# ---------------------------------------------------------------------------
# Example builders
# ---------------------------------------------------------------------------


def save_figure(fig: plt.Figure, path: str | Path, *, check_titles: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if check_titles:
        assert_figure_titles_clear(fig)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def demo_horizontal_bar(output_dir: str | Path, stem: str = "data_viz_horizontal_bar_final") -> Path:
    df = demo_dataframe()
    fig, ax = make_figure(figsize=(10.2, 4.6))
    editorial_barh(ax, df, x="value", y="label")
    set_chart_title(ax, "Phase totals", "Reusable title and subtitle spacing")
    return save_figure(fig, Path(output_dir) / f"{stem}.png")


def demo_vertical_bar(output_dir: str | Path, stem: str = "data_viz_vertical_bar_final") -> Path:
    df = demo_dataframe()
    fig, ax = make_figure(figsize=(8.2, 4.6))
    editorial_bar(ax, df, x="label", y="value")
    set_chart_title(ax, "Phase totals")
    return save_figure(fig, Path(output_dir) / f"{stem}.png")


def demo_line_area(output_dir: str | Path, stem: str = "data_viz_line_area_final") -> Path:
    df = line_demo_dataframe()
    fig, ax = make_figure(figsize=(10.0, 4.6))
    editorial_line(ax, df, x="date", y="cumulative_chunks")
    ticks = _demo_date_ticks(df, "date")
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks.dt.strftime("%b %d"))
    ax.set_ylabel("Cumulative output")
    set_chart_title(ax, "Cumulative output")
    return save_figure(fig, Path(output_dir) / f"{stem}.png")


def demo_scatter(output_dir: str | Path, stem: str = "data_viz_scatter_final") -> Path:
    df = scatter_demo_dataframe()
    fig, ax = make_figure(figsize=(8.8, 4.6))
    editorial_scatter(
        ax,
        df,
        x="latency",
        y="quality",
        hue="group",
        palette_map={"Blue": "blue", "Green": "green", "Pink": "pink", "Gray": "gray"},
    )
    ax.set_xlim(95, 385)
    ax.set_ylim(52, 96)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Quality score")
    set_chart_title(ax, "Latency vs quality")
    return save_figure(fig, Path(output_dir) / f"{stem}.png")


def demo_pie(output_dir: str | Path, stem: str = "data_viz_pie_final") -> Path:
    df = pie_demo_dataframe()
    fig, ax = make_figure(figsize=(7.4, 4.8))
    editorial_pie(ax, df, values="share", labels="segment")
    set_chart_title(ax, "Work split")
    return save_figure(fig, Path(output_dir) / f"{stem}.png")


def demo_gallery(output_dir: str | Path, stem: str = "data_viz_gallery_final") -> Path:
    df = demo_dataframe()
    line_df = line_demo_dataframe()
    scatter_df = scatter_demo_dataframe()

    fig, axes = make_subplots(2, 2, figsize=(12.2, 8.0))
    fig.suptitle("Shared style across plot types", fontsize=14.5, color=TEXT)

    editorial_barh(axes[0][0], df, x="value", y="label")
    set_chart_title(axes[0][0], "Horizontal bars")

    editorial_bar(axes[0][1], df, x="label", y="value")
    set_chart_title(axes[0][1], "Vertical bars")

    editorial_line(axes[1][0], line_df, x="date", y="cumulative_chunks")
    ticks = _demo_date_ticks(line_df, "date")
    axes[1][0].set_xticks(ticks)
    axes[1][0].set_xticklabels(ticks.dt.strftime("%b %d"))
    set_chart_title(axes[1][0], "Line + area")
    axes[1][0].set_ylabel("Cumulative output")

    editorial_scatter(
        axes[1][1],
        scatter_df,
        x="latency",
        y="quality",
        hue="group",
        palette_map={"Blue": "blue", "Green": "green", "Pink": "pink", "Gray": "gray"},
    )
    axes[1][1].set_xlim(95, 385)
    axes[1][1].set_ylim(52, 96)
    axes[1][1].set_xlabel("Latency (ms)")
    axes[1][1].set_ylabel("Quality score")
    set_chart_title(axes[1][1], "Scatter")

    return save_figure(fig, Path(output_dir) / f"{stem}.png")


def build_demo_bundle(output_dir: str | Path) -> list[Path]:
    use_diagram_chart_style()
    output_dir = Path(output_dir)
    return [
        demo_horizontal_bar(output_dir),
        demo_vertical_bar(output_dir),
        demo_line_area(output_dir),
        demo_scatter(output_dir),
        demo_pie(output_dir),
        demo_gallery(output_dir),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate neutral diagram-style chart demos.")
    parser.add_argument("--output-dir", default="tmp/design-chart-demos")
    args = parser.parse_args()

    for output in build_demo_bundle(args.output_dir):
        print(output)


if __name__ == "__main__":
    main()
