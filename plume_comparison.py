#!/usr/bin/env python3

import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
import matplotlib.cm as cm

def extract_polygons(csv_files, label_map):
    extracted_polygons = {}
    global_reference_y = None

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['Polygon_Points'] = df['Polygon Points'].apply(literal_eval)
        first_polygon = np.array(df['Polygon_Points'].iloc[0], dtype=np.float64)
        lowest_y = np.max(first_polygon[:, 1])
        if global_reference_y is None or lowest_y < global_reference_y:
            global_reference_y = lowest_y

    for csv_file in csv_files:
        label = label_map[os.path.basename(csv_file)]
        df = pd.read_csv(csv_file)
        df['Polygon_Points'] = df['Polygon Points'].apply(literal_eval)
        df['Aligned_Polygon'] = align_polygons(df, global_reference_y)
        extracted_polygons[label] = df

    return extracted_polygons

def align_polygons(df, global_reference_y):
    aligned_polygons = []
    for points in df['Polygon_Points']:
        points = np.array(points, dtype=np.float64)
        if len(points) == 0:
            aligned_polygons.append([])
            continue
        centroid_x = np.mean(points[:, 0])
        lowest_y = np.max(points[:, 1])
        points[:, 0] -= centroid_x
        points[:, 1] -= (lowest_y - global_reference_y)
        aligned_polygons.append(points.tolist())
    return aligned_polygons

def get_height_range(df):
    height_cols = [col for col in df.columns if col.startswith("Height_meters")]
    if not height_cols:
        return None, None
    height_data = df[height_cols]
    min_vals = height_data.min(axis=1)
    max_vals = height_data.max(axis=1)
    return min_vals, max_vals

def plot_overlayed_polygons(extracted_polygons, timestamps, output_dir, color_map):
    for timestamp in timestamps:
        fig, ax = plt.subplots(figsize=(8, 6))
        for label, df in extracted_polygons.items():
            if timestamp < len(df):
                polygon = np.array(df.iloc[timestamp]['Aligned_Polygon'])
                if polygon.size > 0:
                    ax.plot(polygon[:, 0], polygon[:, 1], label=label, color=color_map[label])
        ax.set_xlabel("X-Coordinate (Aligned)")
        ax.set_ylabel("Y-Coordinate (Aligned to Global Baseline)")
        if timestamp == -1:
            ax.set_title("Overlay of Last Annotation")
        else:
            ax.set_title(f"Overlay of Polygons at {timestamp} sec")
        ax.legend()
        ax.grid()
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='datalim')
        plt.savefig(os.path.join(output_dir, f"overlay_{timestamp}sec.png"))
        plt.close()

def plot_expansion(extracted_polygons, output_dir, color_map):
    plt.figure(figsize=(10, 6))
    for label, df in extracted_polygons.items():
        plt.plot(df.index / 3, df['Expansion_percent'], label=label, color=color_map[label])
    plt.xlabel("Time (seconds)")
    plt.ylabel("Plume Expansion (%)")
    plt.title("Plume Expansion Over Time")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "plume_expansion.png"))
    plt.close()

def plot_height_ranges_and_area(extracted_polygons, raw_files, label_map, output_dir, color_map):
    fig1, ax1 = plt.subplots(figsize=(12, 7))  # Height ranges
    fig2, ax2 = plt.subplots(figsize=(12, 7))  # Area over time

    for csv_file in raw_files:
        file_name = os.path.basename(csv_file)
        label = label_map[file_name]
        df = pd.read_csv(csv_file)
        min_vals, max_vals = get_height_range(df)

        if min_vals is not None:
            time_axis = np.arange(len(df)) / 3
            ax1.fill_between(time_axis, min_vals, max_vals, color=color_map[label], alpha=0.3, label=label)
            ax1.plot(time_axis, min_vals, color=color_map[label], linewidth=1.2)
            ax1.plot(time_axis, max_vals, color=color_map[label], linewidth=1.2)

            if 'Area' in df.columns:
                ax2.plot(time_axis, df['Area'], label=label, color=color_map[label])

    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Plume Height (meters)")
    ax1.set_title("Plume Height Ranges by Angle")
    ax1.legend()
    ax1.grid()

    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Annotated Plume Area (pixels²)")
    ax2.set_title("Plume Area Over Time")
    ax2.legend()
    ax2.grid()

    fig1.savefig(os.path.join(output_dir, "plume_height_ranges.png"))
    fig2.savefig(os.path.join(output_dir, "plume_area_over_time.png"))
    plt.close(fig1)
    plt.close(fig2)

def plot_every_fifth_frame(extracted_polygons, output_dir, color_map):
    for label, df in extracted_polygons.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = cm.get_cmap('coolwarm')
        total_steps = len(range(0, len(df), 5))
        colors = [cmap(i / (total_steps - 1)) for i in range(total_steps)]
        for i, frame_idx in enumerate(range(0, len(df), 5)):
            polygon = np.array(df.iloc[frame_idx]['Aligned_Polygon'])
            if polygon.size > 0:
                ax.plot(polygon[:, 0], polygon[:, 1], color=colors[i], label=f"t={frame_idx // 3}s")
        ax.set_xlabel("X-Coordinate (Aligned)")
        ax.set_ylabel("Y-Coordinate (Aligned to Global Baseline)")
        ax.set_title(f"Shape Evolution for {label}")
        ax.legend()
        ax.grid()
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='datalim')
        plt.savefig(os.path.join(output_dir, f"shape_evolution_{label}.png"))
        plt.close()

def main():
    input_folder = input("Enter the folder containing CSV files: ").strip()
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    if not csv_files:
        print("No CSV files found. Exiting.")
        return

    print("\nDetected CSV files:")
    for i, f in enumerate(csv_files):
        print(f"  [{i+1}] {os.path.basename(f)}")

    label_map = {}
    for f in csv_files:
        label = input(f"Enter label for {os.path.basename(f)}: ").strip()
        label_map[os.path.basename(f)] = label

    output_dir = os.path.join(input_folder, "plots")
    os.makedirs(output_dir, exist_ok=True)

    ordered_labels = [label_map[os.path.basename(f)] for f in csv_files]
    color_cmap = cm.get_cmap('Set2', len(ordered_labels))
    color_map = {label: color_cmap(i) for i, label in enumerate(ordered_labels)}

    extracted_polygons = extract_polygons(csv_files, label_map)

    timestamps = [3, 5, -1]
    plot_overlayed_polygons(extracted_polygons, timestamps, output_dir, color_map)
    plot_expansion(extracted_polygons, output_dir, color_map)
    plot_height_ranges_and_area(extracted_polygons, csv_files, label_map, output_dir, color_map)
    plot_every_fifth_frame(extracted_polygons, output_dir, color_map)

    print("\nAll plots saved in:", output_dir)

if __name__ == "__main__":
    main()
