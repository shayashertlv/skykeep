# generate_fake_country.py
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate, scale as shapely_scale
from perlin_noise import PerlinNoise
from skimage import measure
import random
import json
import os
import sys

MAP_SIZE   = 100
RESOLUTION = 200
SCALE      = 5.0
THRESHOLD  = 0.3

def generate_perlin_landmask(res=RESOLUTION, scale=SCALE, threshold=THRESHOLD):
    noise = PerlinNoise(octaves=4)
    vals = [
        noise((i / res * scale, j / res * scale))
        for j in range(res)
        for i in range(res)
    ]
    z = np.array(vals).reshape(res, res)
    z = (z - z.min()) / (z.max() - z.min())
    return z > threshold

def extract_border_mask(land):
    contours = measure.find_contours(land.astype(float), 0.5)
    return max(contours, key=len) if contours else []

def contour_to_polygon(contour, map_size, res, scale_up=2.5):
    scale = (map_size / res) * scale_up
    coords = [(x * scale, y * scale) for y, x in contour]
    poly = Polygon(coords)
    poly = rotate(poly, random.uniform(-20, 20), origin='center')

    minx, miny, maxx, maxy = poly.bounds
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    poly = translate(poly,
                     xoff=(map_size/2 - center_x),
                     yoff=(map_size/2 - center_y))

    poly_width = maxx - minx
    poly_height = maxy - miny
    if poly_width > map_size or poly_height > map_size:
        factor = min(map_size / poly_width, map_size / poly_height) * 0.9
        poly = shapely_scale(poly, xfact=factor, yfact=factor, origin='center')

    final_minx, final_miny, final_maxx, final_maxy = poly.bounds
    center_of_country = ((final_minx + final_maxx)/2,
                         (final_miny + final_maxy)/2)
    return poly, center_of_country

def main():
    max_attempts = 10
    best_poly = None
    best_center = None
    largest_area = 0

    for _ in range(max_attempts):
        land_mask = generate_perlin_landmask()
        contour = extract_border_mask(land_mask)
        if len(contour) > 0:
            poly, center = contour_to_polygon(contour, MAP_SIZE, RESOLUTION)
            area = poly.area
            print(f"Attempt area: {area:.2f}")
            if area > 200:
                best_poly = poly
                best_center = center
                break
            elif area > largest_area:
                best_poly = poly
                best_center = center
                largest_area = area

    if not best_poly:
        print("❌ No valid landmass generated.")
        return

    # Save the center point
    print(f"✅ Center of country at: {best_center}")
    with open("center_of_country.json", "w") as f:
        json.dump({"center_of_country": best_center}, f)

    # Draw and save map (WITHOUT the bird‐zone)
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    x, y = best_poly.exterior.xy
    ax.plot(x, y, color='white', linewidth=1.5)
    ax.fill(x, y, color='white', alpha=0.05)

    # (Bird-zone plotting REMOVED here)

    ax.plot([0,100,100,0,0],
            [0,0,100,100,0],
            color='gray',
            linestyle='--',
            linewidth=0.5)

    ax.set_xlim(0, MAP_SIZE)
    ax.set_ylim(0, MAP_SIZE)
    ax.axis('off')

    plt.savefig(
        "fake_country_with_zones.png",
        dpi=300,
        bbox_inches='tight',
        facecolor=fig.get_facecolor()
    )

    if os.environ.get("HIDE_MAP") != "1":
        plt.show()

if __name__ == "__main__":
    main()
