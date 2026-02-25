import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ast import literal_eval
import math
from matplotlib.ticker import MaxNLocator

# Get user input for CSV file, distance, and frame height
csv_file = input("Enter the path to the CSV file (or press Enter for default): ").strip() or '/Users/davduj/plume_analysis/scripts/annotation/polygon_annotations_increased_contrast_frames.csv'
distance = float(input("Enter the distance to the plume (in meters): "))
frame_height = int(input("Enter the vertical resolution of the frames (e.g., 720): "))

# Ask if shadow-estimation FOVs exist
shadow_exists = input("Do Shadow Estimation FOVs exist? (y/n): ").strip().lower() == 'y'

if shadow_exists:
    shadow_fovs = list(map(float, input("Enter the FOVs from Shadow Estimation (comma-separated, e.g., 30,35): ").split(",")))
    broad_fovs = list(map(float, input("Enter the FOVs from Broad Estimation (comma-separated, e.g., 40,45): ").split(",")))
else:
    fovs = list(map(float, input("Enter the FOVs to evaluate (comma-separated, e.g., 40,60): ").split(",")))

# Load and process data
df = pd.read_csv(csv_file)
df['Polygon_Points'] = df['Polygon Points'].apply(literal_eval)
df['Frame_Number'] = df['Frame'].apply(lambda x: int(x.split('_')[1].split('.')[0]))
df = df.sort_values('Frame_Number')

# Initialize dictionaries
if shadow_exists:
    broad_heights = {fov: [] for fov in broad_fovs}
    shadow_heights = {fov: [] for fov in shadow_fovs}
else:
    heights_meters = {fov: [] for fov in fovs}

areas = []
reference_area = None

# Calculate heights and areas
for _, row in df.iterrows():
    pixel_height = row['Height (px)']
    
    if shadow_exists:
        for fov in broad_fovs:
            angle_per_pixel = fov / frame_height
            theta = pixel_height * angle_per_pixel
            theta_radians = math.radians(theta)
            height_meters = distance * math.tan(theta_radians)
            broad_heights[fov].append(height_meters)
        for fov in shadow_fovs:
            angle_per_pixel = fov / frame_height
            theta = pixel_height * angle_per_pixel
            theta_radians = math.radians(theta)
            height_meters = distance * math.tan(theta_radians)
            shadow_heights[fov].append(height_meters)
    else:
        for fov in fovs:
            angle_per_pixel = fov / frame_height
            theta = pixel_height * angle_per_pixel
            theta_radians = math.radians(theta)
            height_meters = distance * math.tan(theta_radians)
            heights_meters[fov].append(height_meters)
    
    points = np.array(row['Polygon_Points'], dtype=np.int32)
    area = cv2.contourArea(points)
    areas.append(area)
    
    if reference_area is None:
        reference_area = area

# Add results to DataFrame
df['Area'] = areas
df['Expansion_percent'] = ((df['Area'] - reference_area) / reference_area) * 100

if shadow_exists:
    for fov in broad_fovs:
        df[f'Height_meters_Broad_{fov}'] = broad_heights[fov]
    for fov in shadow_fovs:
        df[f'Height_meters_Shadow_{fov}'] = shadow_heights[fov]
else:
    for fov in fovs:
        df[f'Height_meters_FOV_{fov}'] = heights_meters[fov]

df.to_csv('processed_plume_data.csv', index=False)
print("Processed data saved to processed_plume_data.csv")

# Plot height over time
plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

if shadow_exists:
    broad_colors = ['orange', 'darkorange', 'coral', 'tomato']
    shadow_colors = ['blue', 'navy', 'skyblue', 'royalblue']
    
    for i, fov in enumerate(broad_fovs):
        col = f'Height_meters_Broad_{fov}'
        plt.plot(df['Frame_Number'], df[col], marker='o', color=broad_colors[i % len(broad_colors)],
                 label=f'Height (Proxy FOV via Zoom Level = {fov}°)')
    
    for i, fov in enumerate(shadow_fovs):
        col = f'Height_meters_Shadow_{fov}'
        plt.plot(df['Frame_Number'], df[col], marker='o', color=shadow_colors[i % len(shadow_colors)],
                 label=f'Height (Estimated FOV via Reference Object = {fov}°)')
else:
    for fov in fovs:
        col = f'Height_meters_FOV_{fov}'
        plt.plot(df['Frame_Number'], df[col], marker='o',
                 label=f'Height (Proxy FOV via Zoom Level = {fov}°)')

plt.xlabel('Frame Number')
plt.ylabel('Plume Height (meters)')
plt.title('Plume Height Over Time for Different FOVs')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('plume_height.png')
plt.close()

# Plot area over time
plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(df['Frame_Number'], df['Area'], marker='o', color='blue', label='Area (pixels²)')
plt.xlabel('Frame Number')
plt.ylabel('Area (pixels²)')
plt.title('Plume Area Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('plume_area.png')
plt.close()

# Plot expansion over time
plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(df['Frame_Number'], df['Expansion_percent'], marker='o', color='green', label='Expansion (%)')
plt.xlabel('Frame Number')
plt.ylabel('Expansion (%)')
plt.title('Plume Expansion Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('plume_expansion.png')
plt.close()

# Print summary
print("\nPlume Analysis Summary:")
print("-" * 30)
print(f"Total frames analyzed: {len(df)}")

print("\nHeight Statistics:")
if shadow_exists:
    for fov in broad_fovs:
        col = f'Height_meters_Broad_{fov}'
        print(f"\nProxy FOV via Zoom Level = {fov}°:")
        print(f"  Initial height: {df[col].iloc[0]:.2f} meters")
        print(f"  Final height: {df[col].iloc[-1]:.2f} meters")
        print(f"  Maximum height: {df[col].max():.2f} meters")
    for fov in shadow_fovs:
        col = f'Height_meters_Shadow_{fov}'
        print(f"\nEstimated FOV via Reference Object = {fov}°:")
        print(f"  Initial height: {df[col].iloc[0]:.2f} meters")
        print(f"  Final height: {df[col].iloc[-1]:.2f} meters")
        print(f"  Maximum height: {df[col].max():.2f} meters")
else:
    for fov in fovs:
        col = f'Height_meters_FOV_{fov}'
        print(f"\nProxy FOV via Zoom Level = {fov}°:")
        print(f"  Initial height: {df[col].iloc[0]:.2f} meters")
        print(f"  Final height: {df[col].iloc[-1]:.2f} meters")
        print(f"  Maximum height: {df[col].max():.2f} meters")

print("\nArea Statistics:")
print(f"  Initial area: {df['Area'].iloc[0]:.2f} pixels²")
print(f"  Final area: {df['Area'].iloc[-1]:.2f} pixels²")
print(f"  Maximum area: {df['Area'].max():.2f} pixels²")

print("\nExpansion Statistics:")
print(f"  Maximum expansion: {df['Expansion_percent'].max():.2f}%")
print(f"  Final expansion: {df['Expansion_percent'].iloc[-1]:.2f}%")
