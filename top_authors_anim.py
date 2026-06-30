"""Animated line chart of the top-5 most active ELOG authors over time.

Each author's *rolling 4-week weekly average* is drawn as a line. A vertical time
cursor sweeps left -> right; lines are revealed up to the cursor, and at every
weekly step the current top-5 authors are highlighted and labelled while everyone
else stays faint. The result is exported as an MP4.

Run:  ./.venv/Scripts/python.exe top_authors_anim.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import imageio_ffmpeg
import matplotlib as mpl
import numpy as np
import polars as pl

# Headless Agg backend + the ffmpeg binary bundled with imageio-ffmpeg.
mpl.use("Agg")
mpl.rcParams.update(
    {
        "animation.ffmpeg_path": imageio_ffmpeg.get_ffmpeg_exe(),
        "font.family": "Arial",
    }
)

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  (must follow rcParams setup)
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation  # noqa: E402

from elog_parser import parse_export  # noqa: E402

# --- Tunables --------------------------------------------------------------
START_DATE: str | None = None  # inclusive UTC date, "YYYY-MM-DD"; None = first entry
END_DATE: str | None = None    # inclusive UTC date, "YYYY-MM-DD"; None = last entry
TITLE = "lblogbook.cern.ch/SciFi"
INCLUDE_LOGO = True
INCLUDE_PARROT = True
WINDOW_WEEKS = 4           # rolling window length for the per-author count
SAMPLES_PER_WEEK = 7       # time resolution: 7 = daily sampling (smoother lines)
TOP_N = 5                  # how many authors to highlight each frame
FPS = 30                   # frames per second  -> higher = smoother motion
WEEKS_PER_SECOND = 1      # data weeks per second of video -> lower = slower
END_HOLD_SECONDS = 1.5     # freeze on the final frame for this long
BACKGROUND_ALPHA = 0.08    # opacity of non-top-N "context" lines
HIGHLIGHT_LW = 2.6         # line width for the current top-N
BACKGROUND_LW = 0.5        # line width for everyone else
VIEW_WEEKS = 6           # width of the sliding viewport in the zoom animation
Y_EASE_DAYS = 10         # ease-in/out span for the auto y-axis (0 = instant)
DPI = 240                  # 8 x 4.5 inches at 240 dpi = 1920 x 1080 pixels
VIDEO_CRF = 24             # H.264 constant-quality compression (higher = smaller)
VIDEO_THREADS = 2          # threads per encoder; bounds total CPU/memory use
N_WORKERS = min(6, max(1, (os.cpu_count() or 4) - 1))
FIGURE_SIZE = (8, 4.5)
AXES_LEFT = 0.12
AXES_RIGHT = 0.66
AXES_TOP = 0.90
AXES_BOTTOM = 0.16
HEADER_ASSET_HEIGHT = 80
EXPORT_PATH = Path(__file__).parent / "data" / "export.xml"
LOGO_PATH = Path(__file__).parent / "data" / "scifi_logo2a_michel.png"
PARROT_PATH = Path(__file__).parent / "data" / "scifi_parrot.gif"
OUTPUT_ZOOM_MP4 = Path(__file__).parent / "top5_authors_zoom.mp4"
OUTPUT_ZOOM_FRAME = Path(__file__).parent / "top5_authors_zoom_frame.png"

# Extended qualitative palette assembled from Paul Tol's muted, bright, and
# vibrant schemes. It is large enough to keep every simultaneously highlighted
# author distinct across the full sliding viewport.
CB_PALETTE = [
    "#332288",  # indigo
    "#CC6677",  # rose
    "#44AA99",  # teal
    "#999933",  # olive
    "#88CCEE",  # cyan
    "#882255",  # wine
    "#117733",  # green
    "#AA4499",  # purple
    "#DDCC77",  # sand
    "#4477AA",  # blue
    "#EE6677",  # coral
    "#228833",  # emerald
    "#CCBB44",  # mustard
    "#66CCEE",  # sky
    "#AA3377",  # magenta
    "#0077BB",  # strong blue
    "#EE7733",  # orange
    "#009988",  # turquoise
    "#EE3377",  # pink
    "#CC3311",  # red
]


def _parse_date_bound(value: str | None, name: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f'{name} must use the "YYYY-MM-DD" format, got {value!r}') from exc


def filter_date_range(
    df: pl.DataFrame, start_date: str | None = START_DATE, end_date: str | None = END_DATE
) -> pl.DataFrame:
    """Keep entries within the inclusive UTC calendar-date range."""
    start = _parse_date_bound(start_date, "START_DATE")
    end = _parse_date_bound(end_date, "END_DATE")
    if start is not None and end is not None and start > end:
        raise ValueError("START_DATE must be on or before END_DATE")

    entry_date = pl.col("date").dt.date()
    if start is not None:
        df = df.filter(entry_date >= start)
    if end is not None:
        df = df.filter(entry_date <= end)
    if df.is_empty():
        raise ValueError("No ELOG entries fall within START_DATE and END_DATE")
    return df


def _samples_per_second() -> float:
    """Time samples consumed per second of video (speed is set in weeks/second)."""
    return WEEKS_PER_SECOND * SAMPLES_PER_WEEK


def _render_frames(n_samples: int) -> int:
    """Total video frames: a continuous sweep plus a hold on the last frame."""
    sweep = round((n_samples - 1) / _samples_per_second() * FPS)
    return int(sweep) + 1 + int(round(END_HOLD_SECONDS * FPS))


def _frame_to_sample(i: int, n_samples: int) -> tuple[int, float]:
    """Map render-frame index -> (sample index ``k``, fraction into next sample).

    Frames advance ``samples_per_second / FPS`` samples each, so motion is
    interpolated between samples instead of jumping a whole step.
    """
    t = min(i * _samples_per_second() / FPS, n_samples - 1)
    k = int(t)
    frac = 0.0 if k >= n_samples - 1 else t - k
    return k, frac


def _lerp(a, b, frac):
    """Linear interpolation that also works for datetimes (b - a is a timedelta)."""
    return a + (b - a) * frac


def _ease_envelope(y: np.ndarray, radius: int) -> np.ndarray:
    """Smooth a y-axis target so it eases in/out, while never dipping below ``y``.

    A rolling max (dilation) of ``radius`` first widens every peak to create
    headroom, then a triangular kernel of the same radius smooths it into eased
    S-shaped transitions. Because the smoothing support never exceeds the
    dilation radius, the result is guaranteed >= ``y`` everywhere (no clipping).
    """
    if radius < 1:
        return y
    win = 2 * radius + 1
    dilated = np.pad(y, radius, mode="edge")
    dilated = np.lib.stride_tricks.sliding_window_view(dilated, win).max(axis=1)
    kernel = np.concatenate([np.arange(1, radius + 2), np.arange(radius, 0, -1)]).astype(float)
    kernel /= kernel.sum()
    smoothed = np.convolve(np.pad(dilated, radius, mode="edge"), kernel, mode="valid")
    return np.maximum(smoothed, y)


def _short_name(name: str) -> str:
    """Compact author label: 'First Last Extra' -> 'First L.' to fit the gutter."""
    parts = name.split()
    return f"{parts[0]} {parts[-1][0]}." if len(parts) >= 2 else name


def _assign_viewport_colors(x, authors, top_per_time, view) -> dict[str, str]:
    """Assign distinct colors to authors highlighted in the same viewport."""
    frequency = {author: 0 for author in authors}
    adjacency: dict[str, set[str]] = {author: set() for author in authors}
    palette_rgb = [np.asarray(mpl.colors.to_rgb(color)) for color in CB_PALETTE]

    for index, names in enumerate(top_per_time):
        for author in names:
            frequency[author] += 1

        lo = int(np.searchsorted(x, x[index] - view, side="left"))
        visible = set().union(*top_per_time[lo : index + 1])
        for author in visible:
            adjacency[author].update(visible - {author})

    color_indices: dict[str, int] = {}
    while len(color_indices) < len(authors):
        unassigned = [author for author in authors if author not in color_indices]
        author = max(
            unassigned,
            key=lambda name: (
                len({color_indices[n] for n in adjacency[name] if n in color_indices}),
                len(adjacency[name]),
                frequency[name],
                name,
            ),
        )
        used = {color_indices[n] for n in adjacency[author] if n in color_indices}
        available = [index for index in range(len(CB_PALETTE)) if index not in used]
        if not available:
            raise ValueError("CB_PALETTE is too small for conflict-free viewport colors")
        color_indices[author] = (
            max(
                available,
                key=lambda index: min(
                    np.linalg.norm(palette_rgb[index] - palette_rgb[neighbor])
                    for neighbor in used
                ),
            )
            if used
            else available[0]
        )

    return {author: CB_PALETTE[index] for author, index in color_indices.items()}


def rolling_top_authors(
    df: pl.DataFrame, window_weeks: int = WINDOW_WEEKS, top_n: int = TOP_N
) -> tuple[list, dict[str, np.ndarray], list[list[str]], int]:
    """Compute rolling weekly-average post counts and the per-day top-N authors.

    Each author's metric is the average number of posts per week in the trailing
    ``window_weeks``, evaluated once per day (a full calendar grid, gaps filled
    with 0) so the lines are continuous.

    Returns ``(times, series, top_per_time, max_roll)`` where:
      * ``times`` is the sorted list of daily timestamps (one per sample),
      * ``series`` maps each author *ever* in the top-N to its weekly-average
        array (aligned with ``times``),
      * ``top_per_time[f]`` is the ordered list of top-N author names at sample f,
      * ``max_roll`` is the peak weekly average.
    """
    df = df.drop_nulls("date")
    window_days = window_weeks * 7

    per_day = (
        df.with_columns(pl.col("date").dt.truncate("1d").alias("day"))
        .group_by("author", "day")
        .len()
        .rename({"len": "n"})
    )

    # Complete daily calendar so the trailing-window count spans real days, not
    # just days that happen to have posts.
    day_min, day_max = per_day["day"].min(), per_day["day"].max()
    days_df = pl.select(
        pl.datetime_range(day_min, day_max, interval="1d", time_zone="UTC").alias("day")
    )
    authors_df = per_day.select("author").unique()
    times = days_df["day"].to_list()

    # Full author x day grid -> 0-fill -> per-author rolling sum over the window.
    grid = (
        authors_df.join(days_df, how="cross")
        .join(per_day, on=["author", "day"], how="left")
        .with_columns(pl.col("n").fill_null(0))
        .sort("author", "day")
        .with_columns(
            (
                pl.col("n")
                .rolling_sum(window_size=window_days, min_samples=1)
                .over("author")
                / window_weeks
            ).alias("roll")
        )
    )

    # Per-day ranking -> ordered top-N (ignore authors with no recent posts).
    ranked = (
        grid.filter(pl.col("roll") > 0)
        .sort(["day", "roll"], descending=[False, True])
        .group_by("day", maintain_order=True)
        .head(top_n)
    )
    top_by_day = {
        row["day"]: row["author"]
        for row in ranked.group_by("day", maintain_order=True)
        .agg(pl.col("author"))
        .iter_rows(named=True)
    }
    top_per_week = [top_by_day.get(d, []) for d in times]
    weeks = times

    # Keep lines only for authors that ever reach a top-N (manageable subset).
    ever_top = ranked.select("author").unique()["author"].to_list()
    sub = grid.filter(pl.col("author").is_in(ever_top))
    # Cast to float: the rolling counts are uint32, and interpolating a *falling*
    # line (b - a < 0) would otherwise underflow to a huge value.
    series = {
        author: part.sort("day")["roll"].to_numpy().astype(np.float64)
        for author, part in sub.partition_by("author", as_dict=True, include_key=True).items()
    }
    # ``partition_by`` keys are 1-tuples -> normalise to plain author strings.
    series = {(k[0] if isinstance(k, tuple) else k): v for k, v in series.items()}

    max_roll = float(grid["roll"].max())
    return weeks, series, top_per_week, max_roll


def build_zoom_animation(weeks, series, top_per_week, max_roll):
    """Build the sliding-viewport animation.

    A ``VIEW_WEEKS``-wide window is anchored at the current week and drags
    through time, with month/year date ticks and a y-axis that auto-fits every
    line visible in the window so all scores in the frame stay on-axis.
    """
    # tz-naive datetimes -> clean date-axis ticks and timedelta arithmetic.
    x = np.array([w.replace(tzinfo=None) for w in weeks])
    view = timedelta(weeks=VIEW_WEEKS)
    authors = sorted(series)
    series_arr = {a: np.asarray(series[a], dtype=float) for a in authors}

    colors = _assign_viewport_colors(x, authors, top_per_week, view)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.subplots_adjust(left=AXES_LEFT, right=AXES_RIGHT, top=AXES_TOP, bottom=AXES_BOTTOM)
    ax.set_title(TITLE, fontsize=13, fontweight="bold")

    lines: dict[str, mpl.lines.Line2D] = {}
    for a in authors:
        (ln,) = ax.plot(
            [], [], color="0.6", lw=BACKGROUND_LW, alpha=BACKGROUND_ALPHA, solid_capstyle="round"
        )
        lines[a] = ln

    cursor = ax.axvline(x[0], color="0.3", lw=1.0, ls="--", alpha=0.7)
    labels = [
        ax.text(0, 0, "", fontsize=13, fontweight="bold", va="center", ha="left", clip_on=False)
        for _ in range(TOP_N)
    ]

    # Day/month/year date ticks: ConciseDateFormatter labels days and folds the
    # month/year context in at boundaries + the offset, so it stays uncluttered.
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator, show_offset=True)
    formatter.formats[3] = "%d %b"          # tick that introduces a new day
    formatter.zero_formats[3] = "%d %b"
    formatter.offset_formats[2] = "%Y"      # context shown once, on the right
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.get_offset_text().set_fontsize(12)
    ax.xaxis.get_offset_text().set_fontweight("bold")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True, min_n_ticks=2))
    ax.set_ylabel("Avg. N. of Logs per Week", fontsize=11)
    ax.yaxis.set_label_coords(-0.11, 0.5)
    ax.tick_params(labelsize=11)
    ax.grid(False)

    n = len(weeks)
    # Per-sample tallest visible value, then eased so the y-axis accelerates and
    # decelerates instead of snapping; lerp'd between samples for sub-step glide.
    ymax_week = np.empty(n)
    for k in range(n):
        lo = int(np.searchsorted(x, x[k] - view, side="left"))
        ymax_week[k] = max((series_arr[a][lo : k + 1].max() for a in authors), default=1.0)
    ymax_week = np.maximum(ymax_week, 1.0)
    ymax_week = _ease_envelope(ymax_week, round(Y_EASE_DAYS * SAMPLES_PER_WEEK / 7))

    def update(f):
        k, frac = _frame_to_sample(f, n)
        x_hi = _lerp(x[k], x[k + 1], frac) if frac else x[k]
        x_lo = x_hi - view
        lo = int(np.searchsorted(x, x_lo, side="left"))  # first index inside window
        ymax = _lerp(ymax_week[k], ymax_week[k + 1], frac) if frac else ymax_week[k]

        # Start one sample left of the window so the line crosses the left edge
        # and gets clipped smoothly, instead of popping when a vertex scrolls out.
        lo_draw = max(0, lo - 1)
        for a, ln in lines.items():
            ys = series_arr[a]
            xs = x[lo_draw : k + 1]
            yy = ys[lo_draw : k + 1].astype(float)
            if frac:
                xs = np.append(xs, x_hi)
                yy = np.append(yy, _lerp(ys[k], ys[k + 1], frac))
            # Hide only *flat* runs at 0 (interior zeros), keeping a zero that sits
            # next to activity so the line still descends to / rises from 0.
            zero = yy <= 0
            prev_zero = np.empty_like(zero)
            prev_zero[0] = True
            prev_zero[1:] = zero[:-1]
            next_zero = np.empty_like(zero)
            next_zero[-1] = True
            next_zero[:-1] = zero[1:]
            yy = np.where(zero & prev_zero & next_zero, np.nan, yy)
            ln.set_data(xs, yy)
            ln.set_color("0.6")
            ln.set_linewidth(BACKGROUND_LW)
            ln.set_alpha(BACKGROUND_ALPHA)
            ln.set_zorder(1)

        # Keep an author's coloured line as long as they are top-N *anywhere in
        # the visible window*, so their past activity doesn't vanish the moment
        # they drop out of the current week. It only fades once it scrolls off
        # the left edge. The current week's leaders are emphasised on top.
        current = top_per_week[k]
        window_top: set[str] = set()
        for j in range(lo, k + 1):
            window_top.update(top_per_week[j])

        for a in window_top:
            ln = lines[a]
            ln.set_color(colors[a])
            ln.set_linewidth(HIGHLIGHT_LW * 0.7)
            ln.set_alpha(0.85)
            ln.set_zorder(4)

        for a in current:
            ln = lines[a]
            ln.set_color(colors[a])
            ln.set_linewidth(HIGHLIGHT_LW)
            ln.set_alpha(1.0)
            ln.set_zorder(5)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0, ymax * 1.06)
        for tick_label in ax.get_xticklabels():
            has_month_text = any(character.isalpha() for character in tick_label.get_text())
            tick_label.set_fontweight("bold" if has_month_text else "normal")
        cursor.set_xdata([x_hi, x_hi])

        # Label the current top-N at the cursor (right edge). Labels follow their
        # lines directly, including when multiple labels overlap.
        for i, lab in enumerate(labels):
            if i < len(current):
                a = current[i]
                ys = series_arr[a]
                y = _lerp(ys[k], ys[k + 1], frac) if frac else float(ys[k])
                value = f"{y:.2f}".rstrip("0").rstrip(".")
                lab.set_text(f"  {_short_name(a)} ({value})")
                lab.set_position((x_hi, y))
                lab.set_color(colors[a])
            else:
                lab.set_text("")

        return list(lines.values()) + labels + [cursor]

    anim = FuncAnimation(fig, update, frames=_render_frames(n), interval=1000 / FPS, blit=False)
    return anim, fig, update


def _render_segment(payload) -> str:
    """Worker: render a contiguous frame range [lo, hi) to its own MP4 segment.

    Runs in a separate process so many frame ranges rasterise in parallel.
    """
    data, lo, hi, seg_path = payload
    _anim, fig, update = build_zoom_animation(*data)
    writer = FFMpegWriter(
        fps=FPS,
        codec="libx264",
        bitrate=-1,
        extra_args=[
            "-preset", "medium",
            "-crf", str(VIDEO_CRF),
            "-threads", str(VIDEO_THREADS),
            "-pix_fmt", "yuv420p",
        ],
    )
    with writer.saving(fig, seg_path, dpi=DPI):
        for f in range(lo, hi):
            update(f)
            writer.grab_frame()
    _anim._draw_was_started = True
    plt.close(fig)
    return seg_path


def _render_parallel(data, mp4_path, frame_path, title, workers=N_WORKERS):
    """Render the animation by splitting its frames across worker processes."""
    n_frames = _render_frames(len(data[0]))

    # Final-frame PNG preview (built once in the parent).
    _anim, fig, update = build_zoom_animation(*data)
    update(n_frames - 1)
    fig.savefig(frame_path, dpi=DPI)
    _anim._draw_was_started = True
    plt.close(fig)
    print(f"Saved frame preview -> {frame_path.name}")

    workers = max(1, min(workers, n_frames))
    bounds = np.linspace(0, n_frames, workers + 1).astype(int)
    tmpdir = Path(tempfile.mkdtemp(prefix="zoom_segs_"))
    payloads, segs = [], []
    for i in range(workers):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        if hi <= lo:
            continue
        seg = tmpdir / f"seg_{i:03d}.mp4"
        segs.append(seg)
        payloads.append((data, lo, hi, str(seg)))

    print(f"  rendering {mp4_path.name}: {n_frames} frames across {len(payloads)} workers")
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_render_segment, p) for p in payloads]
        for fut in as_completed(futures):
            fut.result()  # surface worker exceptions
            done += 1
            print(f"\r    segments done: {done}/{len(payloads)}", end="", flush=True)
    print()

    # Loss-less stitch of the segments (same codec/params -> stream copy).
    list_file = tmpdir / "segments.txt"
    list_file.write_text("".join(f"file '{s.as_posix()}'\n" for s in segs))
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c", "copy", "-metadata", f"title={title}", str(mp4_path)],
        check=True, capture_output=True,
    )
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"Saved animation -> {mp4_path.name}")


def _header_overlay_filter(
    include_logo: bool = INCLUDE_LOGO, include_parrot: bool = INCLUDE_PARROT
) -> str:
    """Build the ffmpeg filter for the enabled header assets."""
    if not include_logo and not include_parrot:
        raise ValueError("At least one header asset must be enabled")

    header_bottom = 1 - AXES_TOP
    filters: list[str] = []
    current_video = "[0:v]"
    input_index = 1

    if include_logo:
        filters.append(
            f"[{input_index}:v]format=rgba,scale=-1:{HEADER_ASSET_HEIGHT}:flags=lanczos[logo]"
        )
        output = "[with_logo]" if include_parrot else "[outv]"
        filters.append(
            f"{current_video}[logo]overlay=x={AXES_LEFT}*main_w:"
            f"y={header_bottom}*main_h-overlay_h:format=auto:shortest=1{output}"
        )
        current_video = output
        input_index += 1

    if include_parrot:
        filters.append(
            f"[{input_index}:v]format=rgba,scale=-1:{HEADER_ASSET_HEIGHT}:flags=neighbor[parrot]"
        )
        filters.append(
            f"{current_video}[parrot]overlay=x={AXES_RIGHT}*main_w-overlay_w:"
            f"y={header_bottom}*main_h-overlay_h:format=auto:shortest=1[outv]"
        )

    return ";".join(filters)


def _header_asset_input_args(for_video: bool) -> list[str]:
    args: list[str] = []
    if INCLUDE_LOGO:
        if for_video:
            args.extend(["-loop", "1"])
        args.extend(["-i", str(LOGO_PATH)])
    if INCLUDE_PARROT:
        args.extend(["-ignore_loop", "0" if for_video else "1", "-i", str(PARROT_PATH)])
    return args


def _overlay_header_assets(mp4_path: Path, frame_path: Path) -> None:
    """Add the enabled header assets after the chart has rendered."""
    enabled_assets = [
        asset_path
        for enabled, asset_path in ((INCLUDE_LOGO, LOGO_PATH), (INCLUDE_PARROT, PARROT_PATH))
        if enabled
    ]
    if not enabled_assets:
        return

    for asset_path in enabled_assets:
        if not asset_path.is_file():
            raise FileNotFoundError(f"Missing header asset: {asset_path}")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    overlay_filter = _header_overlay_filter(INCLUDE_LOGO, INCLUDE_PARROT)
    video_tmp = mp4_path.with_name(f".{mp4_path.stem}.overlay{mp4_path.suffix}")
    frame_tmp = frame_path.with_name(f".{frame_path.stem}.overlay{frame_path.suffix}")

    print(f"Adding {len(enabled_assets)} header asset(s)...")
    try:
        subprocess.run(
            [
                ffmpeg, "-y",
                "-i", str(frame_path),
                *_header_asset_input_args(for_video=False),
                "-filter_complex", overlay_filter,
                "-map", "[outv]",
                "-frames:v", "1",
                "-update", "1",
                str(frame_tmp),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                ffmpeg, "-y",
                "-i", str(mp4_path),
                *_header_asset_input_args(for_video=True),
                "-filter_complex", overlay_filter,
                "-map", "[outv]",
                "-map", "0:a?",
                "-map_metadata", "0",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", str(VIDEO_CRF),
                "-threads", str(VIDEO_THREADS),
                "-pix_fmt", "yuv420p",
                "-c:a", "copy",
                "-movflags", "+faststart",
                "-shortest",
                str(video_tmp),
            ],
            check=True,
            capture_output=True,
        )
        os.replace(frame_tmp, frame_path)
        os.replace(video_tmp, mp4_path)
    finally:
        frame_tmp.unlink(missing_ok=True)
        video_tmp.unlink(missing_ok=True)

    print(f"Added header assets -> {frame_path.name}, {mp4_path.name}")


def main() -> None:
    import sys
    import time

    workers = int(sys.argv[1]) if len(sys.argv) > 1 else N_WORKERS
    df = filter_date_range(parse_export(EXPORT_PATH))
    data = rolling_top_authors(df)
    weeks, series = data[0], data[1]
    print(f"{len(weeks)} weekly frames, {len(series)} candidate authors -> {workers} workers")

    t0 = time.perf_counter()
    _render_parallel(data, OUTPUT_ZOOM_MP4, OUTPUT_ZOOM_FRAME, TITLE, workers)
    _overlay_header_assets(OUTPUT_ZOOM_MP4, OUTPUT_ZOOM_FRAME)
    print(f"Done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
