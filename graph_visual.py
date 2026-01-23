#!/usr/bin/env python3
"""
Visualize chunk position data from a text file.

Supports two input formats:
  - Simple 2-column: x,z (one position per line)
  - Extended 6-column: category,col2,col3,x(chunk),z(chunk),col6

Features:
  - Auto-detects input format
  - Heatmap visualization showing point density
  - Grid lines at configurable intervals
  - Histograms showing X and Z distributions
  - Handles large coordinate ranges efficiently

Usage:
  python graph_visual.py [output.png]
"""

import re
import sys
from collections import Counter

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# =============================================================================
# Configuration
# =============================================================================

INPUT_FILE = "4x4_positions.txt"
DEFAULT_OUTPUT = "output.png"

# Visual settings - OUTPUT SIZE BASED
PLOT_SIZE = 4000  # Target plot size in pixels (will be square-ish)
POINT_RADIUS = 6  # Radius of each data point in pixels
GRID_INTERVAL = 1000  # Grid line interval in data units
GRID_LINE_WIDTH = 2  # Thickness of grid lines
BACKGROUND_COLOR = "#1a1a1a"
GRID_COLOR = "#555555"  # Brighter grid for visibility
AXIS_COLOR = "#00BFFF"
TEXT_COLOR = "#FFFFFF"

# Histogram settings
HIST_HEIGHT = 150  # Pixels for X histogram (below plot)
HIST_WIDTH = 150  # Pixels for Z histogram (left of plot)

# Margins
MARGIN_TOP = 50
MARGIN_BOTTOM = 80
MARGIN_LEFT = 100
MARGIN_RIGHT = 100

# Heatmap color gradient (low to high density)
# Blue -> Cyan -> Green -> Yellow -> Red -> Magenta
HEATMAP_COLORS = [
    (0, 0, 255),  # Blue (lowest)
    (0, 255, 255),  # Cyan
    (0, 255, 0),  # Green
    (255, 255, 0),  # Yellow
    (255, 0, 0),  # Red
    (255, 0, 255),  # Magenta (highest)
]


# =============================================================================
# Helper Functions
# =============================================================================


def parse_value(val):
    """Extract the number before any parenthetical, e.g., '4(64)' -> 4"""
    match = re.match(r"^(-?\d+)", str(val).strip())
    return int(match.group(1)) if match else 0


def interpolate_color(t):
    """
    Interpolate through the heatmap gradient.
    t: 0.0 to 1.0, where 0 is lowest density and 1 is highest.
    Returns (r, g, b) tuple.
    """
    if t <= 0:
        return HEATMAP_COLORS[0]
    if t >= 1:
        return HEATMAP_COLORS[-1]

    # Scale t to index range
    n = len(HEATMAP_COLORS) - 1
    idx = t * n
    lower_idx = int(idx)
    upper_idx = min(lower_idx + 1, n)
    frac = idx - lower_idx

    # Linear interpolation between two colors
    c1 = HEATMAP_COLORS[lower_idx]
    c2 = HEATMAP_COLORS[upper_idx]
    r = int(c1[0] + (c2[0] - c1[0]) * frac)
    g = int(c1[1] + (c2[1] - c1[1]) * frac)
    b = int(c1[2] + (c2[2] - c1[2]) * frac)
    return (r, g, b)


def get_font(size=11):
    """Load a font, falling back to default if needed."""
    for font_name in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf"]:
        try:
            return ImageFont.truetype(font_name, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


# =============================================================================
# Data Loading
# =============================================================================


def load_data(filepath):
    """
    Load and parse the data file.

    Supports:
    - Simple 2-column: x,z
    - Extended 6-column: category,col2,col3,x(chunk),z(chunk),col6
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, header=None, comment="#")

    # Detect format based on number of columns
    if len(df.columns) == 2:
        print("  Detected 2-column format (x,z)")
        result = pd.DataFrame(
            {
                "x": df[0].apply(parse_value),
                "z": df[1].apply(parse_value),
            }
        )
    elif len(df.columns) >= 6:
        print("  Detected 6-column format (category,...,x,z,...)")
        result = pd.DataFrame(
            {
                "x": df[3].apply(parse_value),
                "z": df[4].apply(parse_value),
            }
        )
    else:
        raise ValueError(
            f"Unexpected number of columns: {len(df.columns)}. Expected 2 or 6."
        )

    print(f"  Loaded {len(result):,} points")
    print(f"  X range: {result['x'].min():,} to {result['x'].max():,}")
    print(f"  Z range: {result['z'].min():,} to {result['z'].max():,}")
    return result


# =============================================================================
# Plotting
# =============================================================================


def create_plot(df):
    """Create the visualization with heatmap, grid, and histograms."""
    x_min, x_max = df["x"].min(), df["x"].max()
    z_min, z_max = df["z"].min(), df["z"].max()

    x_range = x_max - x_min + 1
    z_range = z_max - z_min + 1

    # Calculate scale factor to fit data into target plot size
    scale = PLOT_SIZE / max(x_range, z_range)

    # Calculate plot dimensions (maintaining aspect ratio)
    plot_width = int(x_range * scale)
    plot_height = int(z_range * scale)

    # Total image dimensions
    total_width = MARGIN_LEFT + HIST_WIDTH + plot_width + MARGIN_RIGHT
    total_height = MARGIN_TOP + plot_height + HIST_HEIGHT + MARGIN_BOTTOM

    print(f"Creating {total_width:,}x{total_height:,} image...")
    print(f"  Plot area: {plot_width:,}x{plot_height:,} pixels")
    print(f"  Scale: {scale:.4f} pixels per unit")

    img = Image.new("RGB", (total_width, total_height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    font = get_font(14)
    font_small = get_font(12)

    # Plot area offsets
    plot_left = MARGIN_LEFT + HIST_WIDTH
    plot_top = MARGIN_TOP

    # -------------------------------------------------------------------------
    # Draw grid lines
    # -------------------------------------------------------------------------
    # Find first grid line positions
    x_grid_start = (
        (x_min // GRID_INTERVAL) + (1 if x_min % GRID_INTERVAL else 0)
    ) * GRID_INTERVAL
    z_grid_start = (
        (z_min // GRID_INTERVAL) + (1 if z_min % GRID_INTERVAL else 0)
    ) * GRID_INTERVAL

    # Vertical grid lines (X axis)
    for x_val in range(x_grid_start, x_max + 1, GRID_INTERVAL):
        x_pixel = int(plot_left + (x_val - x_min) * scale)
        draw.line(
            [x_pixel, plot_top, x_pixel, plot_top + plot_height],
            fill=GRID_COLOR,
            width=GRID_LINE_WIDTH,
        )

    # Horizontal grid lines (Z axis)
    for z_val in range(z_grid_start, z_max + 1, GRID_INTERVAL):
        y_pixel = int(plot_top + (z_max - z_val) * scale)
        draw.line(
            [plot_left, y_pixel, plot_left + plot_width, y_pixel],
            fill=GRID_COLOR,
            width=GRID_LINE_WIDTH,
        )

    # -------------------------------------------------------------------------
    # Count point density for heatmap
    # -------------------------------------------------------------------------
    print("  Computing point density...")
    point_counts = Counter(zip(df["x"], df["z"]))
    max_count = max(point_counts.values()) if point_counts else 1
    print(f"  Max overlap at single position: {max_count}")

    # -------------------------------------------------------------------------
    # Draw data points with heatmap coloring
    # -------------------------------------------------------------------------
    print("  Drawing points...")
    for (x_val, z_val), count in point_counts.items():
        x_pixel = int(plot_left + (x_val - x_min) * scale)
        y_pixel = int(plot_top + (z_max - z_val) * scale)

        # Logarithmic scaling for better visualization when max_count is high
        if max_count > 1:
            import math

            t = math.log1p(count) / math.log1p(max_count)
        else:
            t = 1.0

        color = interpolate_color(t)
        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

        # Draw point as a filled circle/ellipse for visibility
        draw.ellipse(
            [
                x_pixel - POINT_RADIUS,
                y_pixel - POINT_RADIUS,
                x_pixel + POINT_RADIUS,
                y_pixel + POINT_RADIUS,
            ],
            fill=color_hex,
        )

    # -------------------------------------------------------------------------
    # Draw border around plot
    # -------------------------------------------------------------------------
    draw.rectangle(
        [plot_left - 1, plot_top - 1, plot_left + plot_width, plot_top + plot_height],
        outline=AXIS_COLOR,
        width=1,
    )

    # -------------------------------------------------------------------------
    # Axis labels and ticks
    # -------------------------------------------------------------------------
    # Determine tick interval (aim for ~10-20 ticks)
    def nice_interval(range_val):
        """Calculate a nice tick interval."""
        approx = range_val / 10
        magnitude = 10 ** (len(str(int(approx))) - 1)
        normalized = approx / magnitude
        if normalized < 2:
            return magnitude
        elif normalized < 5:
            return 2 * magnitude
        else:
            return 5 * magnitude

    x_tick_interval = nice_interval(x_range)
    z_tick_interval = nice_interval(z_range)

    # X-axis ticks (bottom)
    x_tick_start = (
        (x_min // x_tick_interval) + (1 if x_min % x_tick_interval else 0)
    ) * x_tick_interval
    for x_val in range(x_tick_start, x_max + 1, x_tick_interval):
        x_pixel = int(plot_left + (x_val - x_min) * scale)
        draw.line(
            [x_pixel, plot_top + plot_height, x_pixel, plot_top + plot_height + 5],
            fill=AXIS_COLOR,
        )
        text = f"{x_val:,}"
        bbox = draw.textbbox((0, 0), text, font=font_small)
        text_w = bbox[2] - bbox[0]
        draw.text(
            (x_pixel - text_w // 2, plot_top + plot_height + 8),
            text,
            fill=TEXT_COLOR,
            font=font_small,
        )

    # Z-axis ticks (right side)
    z_tick_start = (
        (z_min // z_tick_interval) + (1 if z_min % z_tick_interval else 0)
    ) * z_tick_interval
    for z_val in range(z_tick_start, z_max + 1, z_tick_interval):
        y_pixel = int(plot_top + (z_max - z_val) * scale)
        draw.line(
            [plot_left + plot_width, y_pixel, plot_left + plot_width + 5, y_pixel],
            fill=AXIS_COLOR,
        )
        text = f"{z_val:,}"
        bbox = draw.textbbox((0, 0), text, font=font_small)
        text_h = bbox[3] - bbox[1]
        draw.text(
            (plot_left + plot_width + 8, y_pixel - text_h // 2),
            text,
            fill=TEXT_COLOR,
            font=font_small,
        )

    # Axis titles
    # X-axis title
    x_title = "X Coordinate"
    bbox = draw.textbbox((0, 0), x_title, font=font)
    draw.text(
        (
            plot_left + plot_width // 2 - (bbox[2] - bbox[0]) // 2,
            plot_top + plot_height + HIST_HEIGHT + 30,
        ),
        x_title,
        fill=TEXT_COLOR,
        font=font,
    )

    # Z-axis title (would need rotation for proper display, skip for now)

    # -------------------------------------------------------------------------
    # Histograms
    # -------------------------------------------------------------------------
    x_counts = df["x"].value_counts().to_dict()
    z_counts = df["z"].value_counts().to_dict()

    max_x_count = max(x_counts.values()) if x_counts else 1
    max_z_count = max(z_counts.values()) if z_counts else 1

    # X histogram (below plot)
    hist_top = plot_top + plot_height + 25
    for x_val, count in x_counts.items():
        bar_height = int((count / max_x_count) * (HIST_HEIGHT - 10))
        x_pixel = int(plot_left + (x_val - x_min) * scale)
        if bar_height > 0:
            draw.line(
                [
                    x_pixel,
                    hist_top + HIST_HEIGHT - bar_height,
                    x_pixel,
                    hist_top + HIST_HEIGHT,
                ],
                fill=AXIS_COLOR,
            )

    # Z histogram (left of plot)
    hist_right = plot_left - 10
    for z_val, count in z_counts.items():
        bar_width = int((count / max_z_count) * (HIST_WIDTH - 10))
        y_pixel = int(plot_top + (z_max - z_val) * scale)
        if bar_width > 0:
            draw.line(
                [hist_right - bar_width, y_pixel, hist_right, y_pixel],
                fill=AXIS_COLOR,
            )

    # -------------------------------------------------------------------------
    # Title
    # -------------------------------------------------------------------------
    title = f"Chunk Positions ({len(df):,} points)"
    bbox = draw.textbbox((0, 0), title, font=font)
    draw.text(
        (plot_left + plot_width // 2 - (bbox[2] - bbox[0]) // 2, 8),
        title,
        fill=TEXT_COLOR,
        font=font,
    )

    # -------------------------------------------------------------------------
    # Legend (heatmap scale)
    # -------------------------------------------------------------------------
    legend_width = 150
    legend_height = 15
    legend_x = total_width - MARGIN_RIGHT - legend_width
    legend_y = 8

    # Draw gradient bar
    for i in range(legend_width):
        t = i / (legend_width - 1)
        color = interpolate_color(t)
        draw.line(
            [legend_x + i, legend_y, legend_x + i, legend_y + legend_height],
            fill=f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
        )

    # Legend labels
    draw.text(
        (legend_x - 5, legend_y), "1", fill=TEXT_COLOR, font=font_small, anchor="ra"
    )
    draw.text(
        (legend_x + legend_width + 5, legend_y),
        f"{max_count}",
        fill=TEXT_COLOR,
        font=font_small,
    )
    draw.rectangle(
        [legend_x - 1, legend_y - 1, legend_x + legend_width, legend_y + legend_height],
        outline=AXIS_COLOR,
    )

    return img


# =============================================================================
# Main
# =============================================================================


def main():
    input_file = INPUT_FILE
    output_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT

    df = load_data(input_file)
    img = create_plot(df)

    print(f"Saving to {output_file}...")
    img.save(output_file, optimize=True)
    print("Done!")


if __name__ == "__main__":
    main()
