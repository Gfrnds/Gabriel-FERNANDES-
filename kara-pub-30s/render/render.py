"""Rendu de la pub KARA 30 s « Trois gouttes » (version produit) à partir de la photo.

Usage : python3 render.py <photo.jpg> <sortie.mp4>
Aucun texte ajouté, aucune retouche du produit : seulement le cadrage, la lumière et le son.
"""
import subprocess
import sys
import wave

import numpy as np
from PIL import Image

SRC, OUT = sys.argv[1], sys.argv[2]
W, H, FPS, DUR = 1080, 1920, 30, 30.0
N = int(FPS * DUR)
SR = 48000
DROPS = (15.0, 17.2, 19.4)  # les trois gouttes

img = Image.open(SRC).convert("RGB")
IW, IH = img.size


def smooth(a, b, t):
    x = np.clip((t - a) / (b - a), 0.0, 1.0)
    return x * x * (3 - 2 * x)


# Images clés caméra : (temps, centre x, centre y, zoom)
KEYS = [
    (0.0, 690, 1000, 1.00),
    (6.5, 700, 1000, 1.08),
    (12.5, 975, 770, 2.55),
    (20.5, 975, 960, 2.25),
    (24.5, 975, 1190, 1.95),
    (29.0, 700, 1000, 1.06),
    (30.0, 700, 1000, 1.06),
]


def camera(t):
    for (t0, x0, y0, z0), (t1, x1, y1, z1) in zip(KEYS, KEYS[1:]):
        if t <= t1:
            k = float(smooth(t0, t1, t))
            return x0 + (x1 - x0) * k, y0 + (y1 - y0) * k, z0 + (z1 - z0) * k
    return KEYS[-1][1:]


uu, vv = np.meshgrid(np.linspace(0, 1, W, dtype=np.float32), np.linspace(0, 1, H, dtype=np.float32))
rng = np.random.default_rng(7)
COOL = np.array([0.50, 0.60, 0.92], np.float32)
WARM = np.array([1.02, 0.97, 0.87], np.float32)


def frame(i):
    t = i / FPS
    cx, cy, z = camera(t)
    ch = IH / z
    cw = ch * W / H
    x0 = min(max(cx - cw / 2, 0), 1290 - cw)  # évite la barre de défilement à droite
    y0 = min(max(cy - ch / 2, 0), IH - ch)
    crop = img.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch))
    f = np.asarray(crop, np.float32) / 255.0

    # Coordonnées scène (la lumière ne bouge pas avec la caméra)
    X = (x0 + uu * cw) / IW
    Y = (y0 + vv * ch) / IH

    day = float(smooth(2.0, 26.0, t))            # aube -> matin
    curtain = float(smooth(3.0, 7.5, t))         # le rideau s'ouvre de gauche à droite
    edge = curtain * 1.35 - 0.15
    reveal = np.clip((edge - X) / 0.18 + 0.5, 0, 1)

    # Faisceau de fenêtre diagonal, qui descend lentement avec le soleil
    d = X * 0.75 + Y * 0.55 - (0.55 + 0.25 * day)
    beam = np.exp(-(d / 0.42) ** 2)
    m = (X * 0.75 + Y * 0.55) * 7.0 + day * 0.8   # ombres des montants de la fenêtre
    mull = 1 - 0.22 * np.clip(np.cos(m * np.pi) * 3 - 2.2, 0, 1)
    sun = reveal * beam * mull * (0.35 + 0.65 * day)

    tint = COOL * (1 - day) + WARM * day
    ambient = 0.30 + 0.48 * day
    light = ambient + 0.32 * sun
    out = f * light[..., None] * tint

    # Reflets des trois gouttes à la surface du sérum
    for td in DROPS:
        a = np.exp(-((t - td) / 0.35) ** 2) * (t > td - 0.6)
        if a > 0.01:
            r2 = ((X * IW - 975) / 70) ** 2 + ((Y * IH - 935) / 18) ** 2
            out += (a * 0.45 * np.exp(-r2))[..., None] * np.array([1.0, 0.96, 0.88], np.float32)

    # Vignettage léger, grain fin, fondus d'entrée et de sortie
    vig = 1 - 0.28 * ((uu - 0.5) ** 2 + (vv - 0.5) ** 2) * 2
    out *= vig[..., None]
    out += rng.normal(0, 0.012, (H, W, 1)).astype(np.float32)
    out *= float(smooth(0.0, 1.8, t)) * (1 - float(smooth(28.8, 30.0, t)))
    return (np.clip(out, 0, 1) ** (1 / 1.04) * 255).astype(np.uint8)


def audio(path):
    n = int(SR * DUR)
    t = np.arange(n) / SR
    r = np.random.default_rng(3)
    # Ambiance de pièce : bruit brun très doux
    room = np.cumsum(r.normal(0, 1, n))
    room -= np.convolve(room, np.ones(4800) / 4800, "same")
    room = room / np.abs(room).max() * 0.035
    sig = room.copy()
    # Rideau : souffle filtré entre 3 et 7,5 s
    nz = r.normal(0, 1, n)
    nz = np.convolve(nz, np.ones(24) / 24, "same")
    env = np.exp(-((t - 5.0) / 1.4) ** 2)
    sig += nz * env * 0.09
    # Trois gouttes : petit « plic » qui glisse vers le grave
    for td in DROPS:
        k = t >= td
        tt = t[k] - td
        f = 1500 * np.exp(-tt * 9) + 650
        ph = 2 * np.pi * np.cumsum(f) / SR
        sig[k] += np.sin(ph) * np.exp(-tt * 22) * 0.28
        sig[k] += np.sin(ph * 0.5) * np.exp(-tt * 6) * 0.05 * (tt < 1.2)
    # Nappe très basse, qui se réchauffe avec le jour
    day = np.clip((t - 2) / 24, 0, 1)
    pad = (np.sin(2 * np.pi * 110 * t) + 0.5 * np.sin(2 * np.pi * 164.8 * t) * day
           + 0.35 * np.sin(2 * np.pi * 220 * t + 0.3 * np.sin(2 * np.pi * 0.1 * t)))
    sig += pad * 0.018 * np.clip(t / 4, 0, 1)
    fade = np.clip(t / 1.5, 0, 1) * np.clip((DUR - t) / 1.2, 0, 1)
    sig *= fade
    st = np.stack([sig, np.roll(sig, 180)], 1)
    st = (st / np.abs(st).max() * 0.5 * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(st.tobytes())


wav = OUT + ".wav"
audio(wav)
p = subprocess.Popen([
    "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
    "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "23",
    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", OUT,
], stdin=subprocess.PIPE)
for i in range(N):
    p.stdin.write(frame(i).tobytes())
p.stdin.close()
p.wait()
