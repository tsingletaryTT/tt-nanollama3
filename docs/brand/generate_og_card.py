#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Build the Open Graph / Twitter card at 1200x630.

Why this file exists rather than just pointing og:image at the logo: the card is
declared `summary_large_image`, and every scraper that honours that (X, Slack,
LinkedIn, Discord) centre-crops the image to roughly 1.91:1. Feeding it the
square logo therefore ships a band across the figure's torso with the head and
the feet cut off -- and, like the other failure the <head> comment warns about,
nothing anywhere reports it. So the card is composed at the ratio it is consumed
at, with the logo placed inside rather than cropped by someone else's server.

    python docs/brand/generate_og_card.py
"""
from PIL import Image, ImageDraw, ImageFont
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
W, H = 1200, 630

# Site palette (docs/index.html :root) -- light theme, matching the drawing's ground.
BG, INK, INK2, RULE, ACCENT = "#F1F8F8", "#092221", "#3A5452", "#C7D9D8", "#1B8EB1"

SANS  = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD  = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO  = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

card = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(card)

# The logo, square, on the right -- the side a 1.91:1 crop is least likely to touch
# if some consumer crops anyway.
side = 530
logo = Image.open(HERE / "tt-tnt-logo-full.jpg").convert("RGB").resize((side, side), Image.LANCZOS)
lx, ly = W - side - 50, (H - side) // 2
card.paste(logo, (lx, ly))
d.rectangle([lx, ly, lx + side - 1, ly + side - 1], outline=RULE, width=2)

x, y = 64, 96
d.text((x, y), "TENSTORRENT · TT-TNT", font=ImageFont.truetype(MONO, 21), fill=ACCENT)
y += 52

# Wrap to the column rather than hand-breaking: the text column ends where the
# logo begins, and a hand-broken line silently runs under the artwork the moment
# the wording or the face changes.
COL = lx - x - 34
def wrap(words, font):
    lines, cur = [], ""
    for w in words.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) <= COL: cur = t
        else: lines.append(cur); cur = w
    return lines + ([cur] if cur else [])

HEADLINE = "A small language model trained on Tenstorrent hardware."
for size in range(46, 25, -2):
    f_head = ImageFont.truetype(BOLD, size)
    lines = wrap(HEADLINE, f_head)
    if len(lines) <= 3: break
for line in lines:
    d.text((x, y), line, font=f_head, fill=INK); y += int(size * 1.26)

y += 26
d.line([(x, y), (x + 300, y)], fill=RULE, width=2); y += 30

f_sub = ImageFont.truetype(MONO, 20)
for line in ("~123M parameters · 8 blocks", "one epoch · 400M tokens",
             "trained, packaged and served", "on Tenstorrent tooling"):
    d.text((x, y), line, font=f_sub, fill=INK2); y += 30

# One accent edge, so the card has an anchor when it lands on a white surface.
d.rectangle([0, 0, 9, H], fill=ACCENT)

out = HERE / "og-card.jpg"
card.save(out, quality=90, optimize=True, progressive=True)
print(f"wrote {out} ({out.stat().st_size // 1024} KB, {W}x{H})")
