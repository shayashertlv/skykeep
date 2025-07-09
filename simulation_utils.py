import os
import json
import subprocess
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 100, 100
FRAME_MS = 150
FPS = 1000 / FRAME_MS
STEP_FR = int(2.5 * FPS)
LIFESPAN_REAL = int(FPS * 25)
LIFESPAN_FAKE = int(FPS * 15 * 0.5)
MAX_STEP = 3
BASE_LEN = 15
FALSE_LEN_FACTOR = 0.65
SEMI = "#0f3d0f"
SPAWN_MARGIN = 10
NOISE_BATCH_PER_FRAME = 3
RESET_INTERVAL_FRAMES = int(25 * FPS)

# Globals managed by the simulation
blips = []
streams = []
frame_counter = 0
CENTER_OF_COUNTRY = None
bg_img = None
im_obj = None

# Utility: regenerate map

def regenerate_map(generator_py_path: str = "generate_fake_country.py",
                   python_executable: str = None):
    """Rerun the country generator and return (center, image array)."""
    env = os.environ.copy()
    env["HIDE_MAP"] = "1"
    exe = python_executable or os.environ.get("PYTHON", "python")
    subprocess.run([exe, generator_py_path], env=env, check=True)
    with open("center_of_country.json") as f:
        data = json.load(f)
    center = tuple(data["center_of_country"])
    img = np.asarray(Image.open("fake_country_with_zones.png").convert("RGB"))
    return center, img

# Entity classes

class Blip:
    def __init__(self, x, y, dx, dy, lab, life, size, zig=False):
        self.x, self.y, self.dx, self.dy = x, y, dx, dy
        self.lab, self.life, self.age, self.size, self.zig = lab, life, 0, size, zig

    def move(self):
        if self.zig and self.age % 10 == 0:
            self.dx *= -1
        self.x += self.dx
        self.y += self.dy
        self.age += 1

    def alive(self):
        return 0 <= self.x <= WIDTH and 0 <= self.y <= HEIGHT and self.age < self.life

class Stream:
    def __init__(self, persistent=True):
        self.persist = persistent
        while True:
            hx = random.uniform(0, WIDTH)
            hy = random.uniform(0, HEIGHT)
            if (hx <= SPAWN_MARGIN or hx >= WIDTH-SPAWN_MARGIN
             or hy <= SPAWN_MARGIN or hy >= HEIGHT-SPAWN_MARGIN):
                break
        self.hx, self.hy = hx, hy
        vec = np.array([CENTER_OF_COUNTRY[0]-hx, CENTER_OF_COUNTRY[1]-hy])
        n = np.linalg.norm(vec)
        speed = 0.05 * random.uniform(2, 3.5)
        self.dx, self.dy = (vec/n * speed) if n else (0, 0)
        self.age = 0
        self.lifespan = LIFESPAN_REAL if persistent else LIFESPAN_FAKE
        self.xs, self.ys, self.sizes = [], [], []

    def phase(self):
        return min(self.age // STEP_FR, MAX_STEP)

    def emit_size(self):
        return 8 * (self.phase() + 1)

    def max_len(self):
        base = BASE_LEN * (self.phase() + 1)
        if not self.persist:
            base *= FALSE_LEN_FACTOR
        return int(base)

    def update(self):
        self.age += 1
        self.hx += self.dx
        self.hy += self.dy
        self.xs.append(self.hx)
        self.ys.append(self.hy)
        self.sizes.append(self.emit_size())
        keep = self.max_len()
        if len(self.xs) > keep:
            self.xs = self.xs[-keep:]
            self.ys = self.ys[-keep:]
            self.sizes = self.sizes[-keep:]

    def is_expired(self):
        return (self.age >= self.lifespan) or not (0 <= self.hx <= WIDTH and 0 <= self.hy <= HEIGHT)

# Spawn utilities

def spawn_bird_flock():
    if sum(1 for b in blips if b.lab == "bird") >= 24:
        return
    bx, by = random.uniform(10, 90), random.uniform(10, 90)
    dx, dy = random.uniform(0.06, 0.18), random.uniform(0.06, 0.18)
    for _ in range(12):
        ox, oy = random.uniform(-1, 1), random.uniform(-1, 1)
        blips.append(Blip(bx+ox, by+oy, dx, dy, "bird", random.randint(600,720), 2, True))

def spawn_noise(n=24):
    for _ in range(n*3):
        blips.append(Blip(
            random.uniform(0,WIDTH), random.uniform(0,HEIGHT),
            random.uniform(-.15,.15), random.uniform(-.15,.15),
            "noise", random.randint(5,15), size=random.uniform(1,3)
        ))

# Main function to launch simulation

def run_simulation(generator_path: str = "generate_fake_country.py",
                   python_exe: str = None):
    global CENTER_OF_COUNTRY, bg_img, im_obj, frame_counter
    CENTER_OF_COUNTRY, bg_img = regenerate_map(generator_path, python_exe)

    fig, ax = plt.subplots(figsize=(18,18))
    im_obj = ax.imshow(bg_img, extent=[0,100,0,100], interpolation="bilinear")
    ax.set_xlim(0,WIDTH); ax.set_ylim(0,HEIGHT); ax.axis('off')
    stream_scatter = ax.scatter([],[],s=[],c=SEMI,alpha=0.9,marker='.')
    blip_scatter   = ax.scatter([],[],s=[],c=SEMI,alpha=0.9,marker='.')

    def init():
        global frame_counter
        frame_counter = 0
        blips.clear(); streams.clear()
        spawn_noise(NOISE_BATCH_PER_FRAME)
        stream_scatter.set_offsets(np.empty((0,2)))
        stream_scatter.set_sizes([])
        blip_scatter.set_offsets(np.empty((0,2)))
        blip_scatter.set_sizes([])
        return stream_scatter, blip_scatter

    def update(_):
        global frame_counter, CENTER_OF_COUNTRY, bg_img
        frame_counter += 1
        if frame_counter >= RESET_INTERVAL_FRAMES:
            CENTER_OF_COUNTRY, bg_img = regenerate_map(generator_path, python_exe)
            im_obj.set_data(bg_img)
            return init()

        spawn_noise(NOISE_BATCH_PER_FRAME)
        if random.random() < 1/40:
            spawn_bird_flock()

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

        while len(streams) < 3:
            if random.random() < ((1/50)/(1/50+1/80)):
                streams.append(Stream(True))
            else:
                streams.append(Stream(False))

        xs = np.concatenate([s.xs for s in streams]) if streams else np.empty((0,))
        ys = np.concatenate([s.ys for s in streams]) if streams else np.empty((0,))
        ss = np.concatenate([s.sizes for s in streams]) if streams else np.empty((0,))
        stream_scatter.set_offsets(np.column_stack((xs,ys)))
        stream_scatter.set_sizes(ss)

        coords = np.array([[b.x,b.y] for b in blips]) if blips else np.empty((0,2))
        sizes  = np.array([b.size for b in blips])   if blips else np.empty((0,))
        blip_scatter.set_offsets(coords)
        blip_scatter.set_sizes(sizes)

        return stream_scatter, blip_scatter

    ani = animation.FuncAnimation(
        fig, update, init_func=init,
        interval=FRAME_MS, blit=True
    )
    plt.show()

# If running as script
if __name__ == "__main__":
    run_simulation()