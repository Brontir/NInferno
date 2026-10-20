#!/usr/bin/env python3
"""Draw NInferno's icon (a white N on the page's accent color) into an .ico, no libraries.

    make_icon.py OUT.ico

Each size is rendered with 4x4 supersampling and stored as a PNG entry (Windows Vista and later
read those).
"""

import struct
import sys
import zlib

SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
TOP, BOTTOM = (0x5a, 0x4c, 0xff), (0x3a, 0x2b, 0xd8)  # the accent, lighter at the top


def inside_rounded(x, y, radius):
    """(x, y) in the unit square with corners rounded by `radius`."""
    if not (0 <= x <= 1 and 0 <= y <= 1):
        return False
    cx = min(max(x, radius), 1 - radius)
    cy = min(max(y, radius), 1 - radius)
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2


def inside_polygon(x, y, points):
    hit = False
    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            hit = not hit
        j = i
    return hit


# The letter N in unit coordinates: two stems and the diagonal between them.
LEFT, RIGHT, TOPY, BOTY, STEM = 0.25, 0.75, 0.22, 0.78, 0.135
GLYPH = [
    [(LEFT, TOPY), (LEFT + STEM, TOPY), (LEFT + STEM, BOTY), (LEFT, BOTY)],
    [(RIGHT - STEM, TOPY), (RIGHT, TOPY), (RIGHT, BOTY), (RIGHT - STEM, BOTY)],
    [(LEFT, TOPY), (LEFT + STEM * 1.15, TOPY), (RIGHT, BOTY), (RIGHT - STEM * 1.15, BOTY)],
]


def render(size):
    samples = 4
    rows = []
    for py in range(size):
        row = bytearray([0])  # PNG filter: none
        for px in range(size):
            bg = fg = 0
            for sy in range(samples):
                for sx in range(samples):
                    x = (px + (sx + 0.5) / samples) / size
                    y = (py + (sy + 0.5) / samples) / size
                    if inside_rounded(x, y, 0.22):
                        bg += 1
                        if any(inside_polygon(x, y, p) for p in GLYPH):
                            fg += 1
            n = samples * samples
            t = py / max(size - 1, 1)
            base = [round(TOP[c] + (BOTTOM[c] - TOP[c]) * t) for c in range(3)]
            if bg == 0:
                row += bytes((0, 0, 0, 0))
                continue
            share = fg / bg
            color = [round(base[c] * (1 - share) + 255 * share) for c in range(3)]
            row += bytes((*color, round(255 * bg / n)))
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def main():
    images = [render(size) for size in SIZES]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries = b""
    for size, png in zip(SIZES, images):
        side = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", side, side, 0, 0, 1, 32, len(png), offset)
        offset += len(png)
    with open(sys.argv[1], "wb") as out:
        out.write(header + entries + b"".join(images))


if __name__ == "__main__":
    main()
