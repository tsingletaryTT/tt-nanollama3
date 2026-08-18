"""tt-tnt logo -- a Number-Muncher-ish creature drawn in marker on a burned CD-R.

The reference is Tortoise's TNT sleeve: a cartoon scrawled on a CD-R, an early
master of the record. This model is an early master. The creature carries TT-TNT
on its chest, and it is eyeing the disc's centre hole, which is the one piece of
the medium a muncher would obviously want to eat.

Marriage to Tenstorrent is structural rather than applied: the dye field is the TT
teal, the ink is the TT dark forest, and the creature's teeth are the harvested
grid's cores.
"""
import math, pathlib

_S = 20260818
def rnd():
    global _S; _S = (1103515245*_S + 12345) % (2**31); return _S/(2**31)
def jit(a=1.0): return (rnd()-.5)*2*a

OUT = pathlib.Path("/home/ttuser/code/tt-tnt/docs/brand")
INK, TEAL, DEEP, PAPER, SHEEN, HOT = "#08201f", "#4FD1C5", "#1B8EB1", "#EFE7D6", "#9BE3DA", "#FA512E"

def wpath(pts, close=False):
    d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
    for x,y in pts[1:]: d += f" L{x:.1f},{y:.1f}"
    return d + (" Z" if close else "")
def wline(x1,y1,x2,y2,amp=1.5,seg=8):
    return wpath([(x1+(x2-x1)*i/seg+jit(amp), y1+(y2-y1)*i/seg+jit(amp)) for i in range(seg+1)])
def wcircle(cx,cy,r,amp=1.2,seg=64,span=1.0):
    return wpath([(cx+(r+jit(amp))*math.cos(i/seg*math.tau*span),
                   cy+(r+jit(amp))*math.sin(i/seg*math.tau*span)) for i in range(seg+1)])
def wround(x,y,w,h,amp=1.6,seg=6):
    """A hand-drawn rounded box, corners cut like a marker would."""
    pts=[]
    for i in range(seg+1): pts.append((x+w*i/seg+jit(amp), y+jit(amp)))
    for i in range(seg+1): pts.append((x+w+jit(amp), y+h*i/seg+jit(amp)))
    for i in range(seg+1): pts.append((x+w-w*i/seg+jit(amp), y+h+jit(amp)))
    for i in range(seg+1): pts.append((x+jit(amp), y+h-h*i/seg+jit(amp)))
    return wpath(pts, close=True)


def logo(size=640, wordmark=True):
    C = size/2; K = size/640.0
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" '
       f'height="{size}" role="img">','<title>tt-tnt</title>',
       '<desc>A cartoon creature with TT-TNT on its chest, drawn in marker on a burned CD-R, '
       'eyeing the disc centre hole.</desc>',
       '<defs>'
       f'<radialGradient id="dye" cx="44%" cy="40%">'
       f'<stop offset="0" stop-color="{SHEEN}" stop-opacity=".5"/>'
       f'<stop offset=".55" stop-color="{TEAL}"/><stop offset="1" stop-color="{TEAL}"/>'
       '</radialGradient></defs>']

    # ---------- the disc ----------
    s.append(f'<circle cx="{C}" cy="{C}" r="{C-14*K}" fill="url(#dye)"/>')
    for rr,op,w in ((C-46*K,.18,9),(C-96*K,.12,7)):
        s.append(f'<circle cx="{C}" cy="{C}" r="{rr}" fill="none" stroke="#fff" '
                 f'stroke-width="{w*K}" stroke-opacity="{op}"/>')
    s.append(f'<circle cx="{C}" cy="{C}" r="{C*0.30}" fill="{PAPER}" fill-opacity=".20"/>')

    # ---------- the muncher ----------
    # sits above-left of centre; the hub hole is its target, lower right
    bw, bh = 286*K, 224*K
    bx, by = C-150*K, C-250*K
    g = f'<g transform="rotate(-3 {bx+bw/2:.0f} {by+bh/2:.0f})">'
    s.append(g)
    s.append(f'<path d="{wround(bx,by,bw,bh)}" fill="{PAPER}" fill-opacity=".92" '
             f'stroke="{INK}" stroke-width="{7.5*K}" stroke-linejoin="round"/>')

    # eyes -- big, close together, googly
    for ex, pupil_dx in ((bx+bw*0.30, 5*K), (bx+bw*0.63, 8*K)):
        ey = by+bh*0.235
        s.append(f'<path d="{wcircle(ex,ey,31*K,1.0)}" fill="#fff" stroke="{INK}" stroke-width="{5*K}"/>')
        s.append(f'<circle cx="{ex+pupil_dx:.0f}" cy="{ey+4*K:.0f}" r="{13*K:.0f}" fill="{INK}"/>')

    # mouth -- wide, open, with the cores as teeth
    mx, my, mw, mh = bx+bw*0.14, by+bh*0.455, bw*0.72, bh*0.245
    s.append(f'<path d="{wround(mx,my,mw,mh,1.4)}" fill="{INK}" fill-opacity=".92" '
             f'stroke="{INK}" stroke-width="{5*K}"/>')
    tooth = mw/5
    for i in range(5):
        if i in (1,3): continue          # gaps: harvested, like the die
        s.append(f'<path d="{wround(mx+tooth*i+tooth*0.14, my+2*K, tooth*0.72, mh*0.42, 0.8)}" '
                 f'fill="{PAPER}" stroke="none"/>')

    # chest badge
    s.append(f'<text x="{bx+bw/2:.0f}" y="{by+bh*0.885:.0f}" text-anchor="middle" '
             f'font-family="ui-monospace,Menlo,Consolas,monospace" font-size="{38*K:.0f}" '
             f'font-weight="700" fill="{INK}" transform="rotate(-1 {bx+bw/2:.0f} {by+bh*0.99:.0f})">'
             f'TT-TNT</text>')
    s.append('</g>')

    # stubby legs
    # legs: thick enough to survive a favicon, with feet that read as feet
    s.append(f'<g stroke="{INK}" stroke-width="{11*K}" stroke-linecap="round" fill="none">')
    for lx in (bx+bw*0.30, bx+bw*0.68):
        s.append(f'<path d="{wline(lx, by+bh-2*K, lx-5*K, by+bh+40*K, 1.1, 4)}"/>')
    s.append('</g>')
    s.append(f'<g fill="{INK}">')
    for lx in (bx+bw*0.30, bx+bw*0.68):
        s.append(f'<ellipse cx="{lx-9*K:.0f}" cy="{by+bh+45*K:.0f}" rx="{21*K:.0f}" ry="{9*K:.0f}" '
                 f'transform="rotate({jit(5):.0f} {lx-9*K:.0f} {by+bh+45*K:.0f})"/>')
    s.append('</g>')

    # ---------- hub: the thing it is about to eat ----------
    s.append(f'<circle cx="{C}" cy="{C+C*0.275:.0f}" r="{C*0.135}" fill="{PAPER}" fill-opacity=".96" '
             f'stroke="{INK}" stroke-width="{2*K}" stroke-opacity=".3"/>')
    s.append(f'<circle cx="{C}" cy="{C+C*0.275:.0f}" r="{C*0.062}" fill="#0b1918"/>')
    # crumbs
    s.append(f'<g fill="{INK}" fill-opacity=".5">')
    for dx,dy,r_ in ((-46,-30,4),(38,-40,3.5),(-24,-52,3)):
        s.append(f'<circle cx="{C+dx*K:.0f}" cy="{C+C*0.275+dy*K:.0f}" r="{r_*K:.1f}"/>')
    s.append('</g>')

    if wordmark:
        s.append(f'<text x="{C}" y="{C+C*0.66:.0f}" text-anchor="middle" '
                 f'font-family="ui-monospace,Menlo,Consolas,monospace" font-size="{23*K:.0f}" '
                 f'fill="{INK}" fill-opacity=".62" transform="rotate(1 {C} {C+C*0.66:.0f})">'
                 f'early master</text>')

    s.append(f'<path d="{wcircle(C,C,C-14*K,1.0)}" fill="none" stroke="{INK}" '
             f'stroke-width="{2.6*K}" stroke-opacity=".5" stroke-linecap="round"/>')
    s.append('</svg>')
    return "".join(s)

(OUT/"tt-tnt-logo.svg").write_text(logo(640, True))
(OUT/"tt-tnt-mark.svg").write_text(logo(256, False))
print("muncher generated")
