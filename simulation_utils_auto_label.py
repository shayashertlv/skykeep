# simulation_utils_auto_label.py
"""simulation_utils_auto_label.py

Enhanced simulator for SkyKeep POC:

* **--record**   Save each video frame as PNG and append per‑frame object annotations to
  a single `labels.json`.
* **--headless** Run without opening a GUI window (ideal for overnight dataset gen).
* **--frames N** How many frames to simulate before exit (default: endless unless
  --record is on, then 10 000).
* **--out DIR**  Destination directory (default: `dataset`).

The file keeps the original visual behaviour so you can still watch the radar, but
adds an automated labelling pipeline that costs ~30 LoC.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── choose GUI or headless backend BEFORE pyplot is imported anywhere else
BACKEND = "Agg" if "--headless" in os.sys.argv else "TkAgg"
matplotlib.use(BACKEND)

# ────────────────────────────────────────────────────────────────────────────
# Simulation constants (same as original)
# ────────────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 100, 100
FRAME_MS = 150                  # ≈ 6.67 FPS          (matches original)
FPS = 1000 / FRAME_MS
STEP_FR = int(2.5 * FPS)        # how many dots a stream keeps
LIFESPAN_REAL = int(FPS * 25)   # real targets live longer
LIFESPAN_FAKE = int(FPS * 15 * 0.5)
NOISE_BATCH_PER_FRAME = 3
RESET_INTERVAL_FRAMES = int(25 * FPS)   # regenerate map every ~25 s

blips: list["Blip"] = []   # global for ease of porting original code
streams: list["Stream"] = []
frame_counter = 0
CENTER_OF_COUNTRY = (0.0, 0.0)
bg_img = None  # RGB numpy

# ────────────────────────────────────────────────────────────────────────────
# Primitive objects
# ────────────────────────────────────────────────────────────────────────────
class Blip:
    def __init__(self, x, y, dx, dy, lab, life, size=2, zig=False):
        self.x, self.y = x, y
        self.dx, self.dy = dx, dy
        self.lab = lab            # "bird" | "noise"
        self.life = life
        self.age = 0
        self.size = size
        self.zig = zig            # birds zig‑zag

    # ---------------------------------------------------------------
    def move(self):
        if self.zig and self.age % 10 == 0:
            self.dx *= -1
        self.x += self.dx
        self.y += self.dy
        self.age += 1

    # ---------------------------------------------------------------
    def alive(self):
        return (
            0 <= self.x <= WIDTH and 0 <= self.y <= HEIGHT and self.age < self.life
        )


class Stream:
    def __init__(self, persistent: bool = True):
        self.persist = persistent  # True ⇒ "real_stream"

        # choose random in‑country spawn (avoid edges)
        while True:
            hx, hy = random.uniform(10, 90), random.uniform(10, 90)
            if 10 < hx < 90 and 10 < hy < 90:
                break
        self.hx, self.hy = hx, hy

        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(0.03, 0.11)
        self.dx = np.cos(angle) * speed
        self.dy = np.sin(angle) * speed

        self.lifespan = LIFESPAN_REAL if persistent else LIFESPAN_FAKE
        self.age = 0

        self.xs = [hx]
        self.ys = [hy]
        self.sizes = [2]

    # ---------------------------------------------------------------
    def update(self):
        self.age += 1
        self.hx += self.dx
        self.hy += self.dy
        self.xs.append(self.hx)
        self.ys.append(self.hy)
        self.sizes.append(max(1, 3 - self.age / 60))

        keep = STEP_FR
        self.xs = self.xs[-keep:]
        self.ys = self.ys[-keep:]
        self.sizes = self.sizes[-keep:]

    # ---------------------------------------------------------------
    def is_expired(self):
        out_of_bounds = not (0 <= self.hx <= WIDTH and 0 <= self.hy <= HEIGHT)
        return self.age >= self.lifespan or out_of_bounds

    # ---------------------------------------------------------------
    def emit_size(self):
        return self.sizes[-1] if self.sizes else 1


# ────────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────────

def regenerate_map(generator_py: str = "generate_fake_country.py",
                   python_exe: str | None = None):
    """Run the external map generator and load its outputs."""
    env = os.environ.copy()
    env.update({"HIDE_MAP": "1",          # suppress GUI in generate_fake_country.py
                "MPLBACKEND": "Agg"})     # force headless Matplotlib

    exe = python_exe or env.get("PYTHON_EXE") or os.sys.executable
    subprocess.run([exe, generator_py], env=env, check=True)

    center = tuple(json.load(open("center_of_country.json"))["center_of_country"])
    img = np.asarray(Image.open("fake_country_with_zones.png").convert("RGB"))
    return center, img



def spawn_bird_flock():
    if sum(1 for b in blips if b.lab == "bird") >= 24:
        return
    bx, by = random.uniform(10, 90), random.uniform(10, 90)
    dx, dy = random.uniform(0.06, 0.18), random.uniform(0.06, 0.18)
    for _ in range(12):
        ox, oy = random.uniform(-1, 1), random.uniform(-1, 1)
        blips.append(
            Blip(bx + ox, by + oy, dx, dy, "bird", random.randint(600, 720), 2, True)
        )


def spawn_noise(batch: int = 24):
    for _ in range(batch * 3):
        blips.append(
            Blip(
                random.uniform(0, WIDTH),
                random.uniform(0, HEIGHT),
                random.uniform(-0.15, 0.15),
                random.uniform(-0.15, 0.15),
                "noise",
                random.randint(5, 15),
                size=random.uniform(1, 3),
            )
        )


# ────────────────────────────────────────────────────────────────────────────
# Core simulation function
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
# Core simulation function  – FINAL WORKING VERSION
# ────────────────────────────────────────────────────────────────────────────
def run_simulation(args,
                   generator_py: str = "generate_fake_country.py",
                   python_exe: str | None = None):
    """
    Runs the radar sim, optionally records PNG frames + labels.json.
    Works in GUI (TkAgg) and fully headless (Agg) modes.
    """
    global CENTER_OF_COUNTRY, bg_img, frame_counter

    # ――― initial background ―――
    CENTER_OF_COUNTRY, bg_img = regenerate_map(generator_py, python_exe)

    # ――― set up output dirs if recording ―――
    if args.record:
        out_root   = Path(args.out)
        frames_dir = out_root / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        labels_path = out_root / "labels.json"
        labels: list[dict] = []

    # ――― Matplotlib figure and artists ―――
    fig, ax = plt.subplots(figsize=(5, 5))
    im_obj = ax.imshow(bg_img, extent=[0, 100, 0, 100], interpolation="bilinear")
    ax.set_xlim(0, WIDTH); ax.set_ylim(0, HEIGHT); ax.axis("off")

    stream_scatter = ax.scatter([], [], s=[], c="lime", marker=".", alpha=0.9)
    blip_scatter   = ax.scatter([], [], s=[], c="red",  marker="o", alpha=0.9)

    # ――― init & update callbacks ―――
    def init():
        global frame_counter
        frame_counter = 0
        blips.clear(); streams.clear()

        empty = np.empty((0, 2))
        stream_scatter.set_offsets(empty); stream_scatter.set_sizes([])
        blip_scatter.set_offsets(empty);   blip_scatter.set_sizes([])
        return stream_scatter, blip_scatter

    def update(_):
        global frame_counter, CENTER_OF_COUNTRY, bg_img
        frame_counter += 1

        # refresh map every RESET_INTERVAL_FRAMES
        if frame_counter % RESET_INTERVAL_FRAMES == 0:
            CENTER_OF_COUNTRY, bg_img = regenerate_map(generator_py, python_exe)
            im_obj.set_data(bg_img)

        # spawn new objects
        spawn_noise(NOISE_BATCH_PER_FRAME)
        if random.random() < 1/40:   spawn_bird_flock()
        if random.random() < 1/200:  streams.append(Stream(True))
        if random.random() < 1/80:   streams.append(Stream(False))

        # move & age existing objects
        for b in blips[:]:
            b.move()
            if not b.alive():
                blips.remove(b)

        for s in streams[:]:
            if not s.is_expired():
                s.update()
            else:
                if s.persist:
                    streams.remove(s)
                else:
                    s.xs.pop(0); s.ys.pop(0); s.sizes.pop(0)
                    if not s.xs:
                        streams.remove(s)

        # update scatter data
        xs = np.concatenate([s.xs for s in streams]) if streams else np.empty((0,))
        ys = np.concatenate([s.ys for s in streams]) if streams else np.empty((0,))
        ss = np.concatenate([s.sizes for s in streams]) if streams else np.empty((0,))
        stream_scatter.set_offsets(np.column_stack((xs, ys))); stream_scatter.set_sizes(ss)

        if blips:
            coords = np.array([[b.x, b.y] for b in blips])
            sizes  = np.array([b.size for b in blips])
        else:
            coords = np.empty((0, 2)); sizes = np.empty((0,))
        blip_scatter.set_offsets(coords); blip_scatter.set_sizes(sizes)

        # ――― record frame + labels if requested ―――
        if args.record:
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # RGBA bytes
            h, w = fig.canvas.get_width_height()
            rgb = buf.reshape(h, w, 4)[..., :3]  # drop alpha
            imageio.imwrite(frames_dir / f"{frame_counter:06d}.png", rgb)

            objects = (
                [{"type": "real_stream" if s.persist else "fake_stream",
                  "x": float(s.xs[-1]), "y": float(s.ys[-1]), "size": float(s.emit_size())}
                 for s in streams] +
                [{"type": b.lab, "x": float(b.x), "y": float(b.y), "size": float(b.size)}
                 for b in blips]
            )
            labels.append({"frame_id": frame_counter, "objects": objects})

            if frame_counter >= args.frames:
                plt.close(fig)

        return stream_scatter, blip_scatter

    # ――― build animation (for GUI mode) ―――
    ani = animation.FuncAnimation(
        fig, update, init_func=init,
        interval=FRAME_MS,
        frames=args.frames if args.frames else None,
        cache_frame_data=False,
    )

    # ――― execute headless vs GUI ―――
    if args.headless:
        init()
        total = args.frames or 10000  # safety cap for non-record runs
        for i in range(total):
            update(i)
        plt.close(fig)
    else:
        plt.show()

    # ――― save labels when finished ―――
    if args.record:
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2)
        print(f"\nSaved {len(labels)} frame annotations → {labels_path}\n")


def parse_args():
    p = argparse.ArgumentParser("Synthetic radar simulator with auto‑labelling")
    p.add_argument("--record", action="store_true", help="save PNG frames + labels.json")
    p.add_argument("--headless", action="store_true", help="run without GUI window")
    p.add_argument("--frames", type=int, default=10000,
                   help="number of frames to simulate when --record (default 10000)")
    p.add_argument("--out", type=str, default="dataset", help="output root directory")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # ensure frames >0 when recording
    if not args.record:
        args.frames = None  # unlimited if just visualising
    run_simulation(args)
