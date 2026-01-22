#!/usr/bin/env python3
"""
Visualize chunk data from a text file.

Input format (6 columns, comma delimited):
  col1: category (9, 12, 15)
  col2, col3: ignored
  col4: x position (number with optional parenthetical to discard)
  col5: z position (number with optional parenthetical to discard)
  col6: ignored

Example: 9,3,3,4(64),-9(-144),64622
"""

import re
import sys

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Configuration
POINT_SIZE = 3  # Each data point is 3x3 pixels
BACKGROUND_COLOR = "#1a1a1a"
GRID_COLOR = "#333333"
AXIS_COLOR = "#00BFFF"
TEXT_COLOR = "#FFFFFF"

# Category colors
COLORS = {
    9: "#00BFFF",  # deep sky blue
    12: "#FF3333",  # vibrant red
    15: "#33FF33",  # vibrant green
}


def parse_value(val):
    """Extract the number before any parenthetical, e.g., '4(64)' -> 4"""
    match = re.match(r"^(-?\d+)", str(val).strip())
    return int(match.group(1)) if match else 0


def load_data(filepath):
    """Load and parse the data file."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, header=None, comment="#")

    result = pd.DataFrame(
        {
            "x": df[3].apply(parse_value),
            "z": df[4].apply(parse_value),
            "category": df[0].astype(int),
        }
    )

    print(f"Loaded {len(result):,} points")
    print(f"X range: {result['x'].min()} to {result['x'].max()}")
    print(f"Z range: {result['z'].min()} to {result['z'].max()}")
    return result


def create_plot(df):
    """Create the plot image with grid lines, data points, and histograms."""
    x_min, x_max = df["x"].min(), df["x"].max()
    z_min, z_max = df["z"].min(), df["z"].max()

    # Calculate image size based on data range
    plot_width = (x_max - x_min + 1) * POINT_SIZE
    plot_height = (z_max - z_min + 1) * POINT_SIZE

    # Histogram settings
    hist_height = 80  # Height for X histogram (below plot)
    hist_width = 80  # Width for Z histogram (left of plot)

    # Margins for axis labels
    margin_left = 50 + hist_width
    margin_right = 50
    margin_top = 20
    margin_bottom = 35 + hist_height

    width = plot_width + margin_left + margin_right
    height = plot_height + margin_top + margin_bottom

    print(f"Creating {width}x{height} image...")
    img = Image.new("RGB", (width, height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except:
        font = ImageFont.load_default()

    # Draw grid lines first (behind data)
    # X-axis grid lines (multiples of 9) - +1 to center in 3x3 cell
    x_start = ((x_min // 9) + 1) * 9 if x_min % 9 != 0 else x_min
    for x_val in range(x_start, x_max + 1, 9):
        x_pixel = margin_left + (x_val - x_min) * POINT_SIZE + 1
        draw.line(
            [x_pixel, margin_top, x_pixel, margin_top + plot_height], fill=GRID_COLOR
        )

    # Z-axis grid lines (multiples of 9) - +1 to center in 3x3 cell
    z_start = ((z_min // 9) + 1) * 9 if z_min % 9 != 0 else z_min
    for z_val in range(z_start, z_max + 1, 9):
        y_pixel = margin_top + (z_max - z_val) * POINT_SIZE + 1
        draw.line(
            [margin_left, y_pixel, margin_left + plot_width, y_pixel], fill=GRID_COLOR
        )

    # Count hits per location for category 9 (blue) heatmapping
    blue_counts = {}
    for _, row in df.iterrows():
        if row["category"] == 9:
            key = (row["x"], row["z"])
            blue_counts[key] = blue_counts.get(key, 0) + 1

    max_blue_count = max(blue_counts.values()) if blue_counts else 1
    print(f"Blue heatmap: max overlap = {max_blue_count}")

    # Draw data points on top of grid
    for _, row in df.iterrows():
        x = margin_left + (row["x"] - x_min) * POINT_SIZE + 1
        y = margin_top + (z_max - row["z"]) * POINT_SIZE + 1

        if row["category"] == 9:
            # Heatmap: blue (#0000FF) to magenta (#FF00FF)
            key = (row["x"], row["z"])
            count = blue_counts[key]
            t = count / max_blue_count
            r = int(255 * t)
            g = 0
            b = 255
            color = f"#{r:02x}{g:02x}{b:02x}"
        else:
            color = COLORS.get(row["category"], "#FFFFFF")

        draw.rectangle([x, y, x, y], fill=color)

    # Draw border
    draw.rectangle(
        [
            margin_left - 1,
            margin_top - 1,
            margin_left + plot_width,
            margin_top + plot_height,
        ],
        outline=AXIS_COLOR,
        width=1,
    )

    # X-axis ticks and labels - +1 to center
    for x_val in range(x_start, x_max + 1, 9):
        x_pixel = margin_left + (x_val - x_min) * POINT_SIZE + 1
        draw.line(
            [x_pixel, margin_top + plot_height, x_pixel, margin_top + plot_height + 4],
            fill=AXIS_COLOR,
        )
        text = str(x_val)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text(
            (x_pixel - text_w // 2, margin_top + plot_height + 6),
            text,
            fill=TEXT_COLOR,
            font=font,
        )

    # Z-axis ticks and labels - +1 to center
    for z_val in range(z_start, z_max + 1, 9):
        y_pixel = margin_top + (z_max - z_val) * POINT_SIZE + 1
        draw.line(
            [margin_left + plot_width, y_pixel, margin_left + plot_width + 4, y_pixel],
            fill=AXIS_COLOR,
        )
        text = str(z_val)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_h = bbox[3] - bbox[1]
        draw.text(
            (margin_left + plot_width + 6, y_pixel - text_h // 2),
            text,
            fill=TEXT_COLOR,
            font=font,
        )

    # --- Histograms ---
    # Count occurrences per X and Z coordinate
    x_counts = df["x"].value_counts().to_dict()
    z_counts = df["z"].value_counts().to_dict()

    max_x_count = max(x_counts.values()) if x_counts else 1
    max_z_count = max(z_counts.values()) if z_counts else 1

    # X histogram (below the plot)
    hist_top = margin_top + plot_height + 35
    for x_val in range(x_min, x_max + 1):
        count = x_counts.get(x_val, 0)
        if count == 0:
            continue
        bar_height = int((count / max_x_count) * (hist_height - 5))
        x_pixel = margin_left + (x_val - x_min) * POINT_SIZE + 1
        draw.line(
            [
                x_pixel,
                hist_top + hist_height - bar_height,
                x_pixel,
                hist_top + hist_height,
            ],
            fill=AXIS_COLOR,
        )

    # Z histogram (left of the plot)
    hist_right = margin_left - 10
    for z_val in range(z_min, z_max + 1):
        count = z_counts.get(z_val, 0)
        if count == 0:
            continue
        bar_width = int((count / max_z_count) * (hist_width - 5))
        y_pixel = margin_top + (z_max - z_val) * POINT_SIZE + 1
        draw.line(
            [hist_right - bar_width, y_pixel, hist_right, y_pixel],
            fill=AXIS_COLOR,
        )

    return img


def main():
    input_file = "refined_results_noskip.txt"
    output_file = sys.argv[1] if len(sys.argv) > 1 else "output.png"

    df = load_data(input_file)
    img = create_plot(df)

    print(f"Saving to {output_file}...")
    img.save(output_file)
    print("Done!")


if __name__ == "__main__":
    main()
