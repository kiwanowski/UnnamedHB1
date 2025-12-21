#!/usr/bin/env python3
"""
aif2png.py — best-effort converter for the .aif texture format used in achrostel/UnnamedHB1

Usage:
    python aif2png.py input.aif output.png

This version computes width from the image data length so output images match the file's implied aspect ratio.
Requires Pillow: pip install pillow
"""

import sys
import struct
from PIL import Image

def read_header(data):
    if len(data) < 20:
        raise ValueError("File too small to contain header")
    name = data[0:8].decode('ascii', errors='replace')
    mode = data[8]
    transparency = data[9]
    size = data[10]
    clut_len_field = data[11]
    img_data_len = struct.unpack_from('<I', data, 12)[0]
    clut_data_len = struct.unpack_from('<I', data, 16)[0]
    return {
        'name': name,
        'mode': mode,
        'transparency': transparency,
        'size': size,
        'clut_len_field': clut_len_field,
        'img_data_len': img_data_len,
        'clut_data_len': clut_data_len
    }

def psx_15bit_to_rgb(val):
    # bits: 0-4 = R, 5-9 = G, 10-14 = B (typical PSX)
    r5 = val & 0x1F
    g5 = (val >> 5) & 0x1F
    b5 = (val >> 10) & 0x1F
    # expand 5-bit to 8-bit
    r = (r5 << 3) | (r5 >> 2)
    g = (g5 << 3) | (g5 >> 2)
    b = (b5 << 3) | (b5 >> 2)
    return (r, g, b, 255)

def parse_clut(clut_bytes):
    palette = []
    for i in range(0, len(clut_bytes), 2):
        if i+1 >= len(clut_bytes):
            break
        val = clut_bytes[i] | (clut_bytes[i+1] << 8)
        palette.append(psx_15bit_to_rgb(val))
    return palette

def compute_dimensions(header):
    mode = header['mode']
    size = header['size']
    height = 64 << size

    img_len = header['img_data_len']
    # compute number of pixels based on mode
    if mode == 2:
        # 2 bytes per pixel
        pixels = img_len // 2
    elif mode == 1:
        # 1 byte per pixel
        pixels = img_len
    elif mode == 0:
        # 4-bit: two pixels per byte
        pixels = img_len * 2
    else:
        raise ValueError("Unknown mode: {}".format(mode))

    # integer width if possible
    if height == 0:
        raise ValueError("Invalid height computed")
    width = pixels // height
    if width == 0:
        # fallback to engine formula used in repo:
        width = (64 << size) >> (2 - mode)
    return width, height

def convert_aif_to_rgba(header, clut_bytes, img_bytes):
    mode = header['mode']
    width, height = compute_dimensions(header)

    # parse palette
    palette = parse_clut(clut_bytes) if len(clut_bytes) >= 2 else []

    # Prepare output buffer (RGBA)
    out = bytearray(width * height * 4)

    if mode == 2:
        expected = width * height * 2
        if len(img_bytes) < expected:
            # partial fallback: continue but warn
            pass
        idx = 0
        for y in range(height):
            for x in range(width):
                if idx + 1 >= len(img_bytes):
                    val = 0
                else:
                    lo = img_bytes[idx]
                    hi = img_bytes[idx+1]
                    val = lo | (hi << 8)
                r,g,b,a = psx_15bit_to_rgb(val)
                pos = (y*width + x) * 4
                out[pos:pos+4] = bytes((r,g,b,255))
                idx += 2

    elif mode == 1:
        expected = width * height
        if len(img_bytes) < expected:
            pass
        idx = 0
        for y in range(height):
            for x in range(width):
                if idx >= len(img_bytes):
                    pi = 0
                else:
                    pi = img_bytes[idx]
                if pi < len(palette):
                    r,g,b,a = palette[pi]
                else:
                    r,g,b,a = (0,0,0,255)
                pos = (y*width + x) * 4
                out[pos:pos+4] = bytes((r,g,b,255))
                idx += 1

    elif mode == 0:
        expected = (width * height + 1) // 2
        if len(img_bytes) < expected:
            pass
        px = 0
        for b in img_bytes:
            lo = b & 0x0F
            hi = (b >> 4) & 0x0F
            for nib in (lo, hi):
                if px >= width * height:
                    break
                if nib < len(palette):
                    r,g,bcol,a = palette[nib]
                else:
                    r,g,bcol,a = (0,0,0,255)
                pos = px * 4
                out[pos:pos+4] = bytes((r,g,bcol,255))
                px += 1
    else:
        raise ValueError("Unknown mode: {}".format(mode))

    return (width, height, bytes(out))

def main():
    if len(sys.argv) < 3:
        print("Usage: python aif2png.py input.aif output.png")
        sys.exit(1)
    infile = sys.argv[1]
    outfile = sys.argv[2]

    with open(infile, 'rb') as f:
        data = f.read()

    header = read_header(data)
    clut_off = 20
    clut_len = header['clut_data_len']
    img_len = header['img_data_len']
    clut_bytes = data[clut_off:clut_off + clut_len]
    img_bytes = data[clut_off + clut_len: clut_off + clut_len + img_len]

    print("AIF name: {!r}".format(header['name'].strip()))
    print("mode={}, transparency={}, size={}, clut_len_field={}, img_data_len={}, clut_data_len={}".format(
        header['mode'], header['transparency'], header['size'], header['clut_len_field'],
        header['img_data_len'], header['clut_data_len']
    ))

    width, height, rgba = None, None, None
    try:
        width, height, rgba = (*convert_aif_to_rgba(header, clut_bytes, img_bytes)[:2], None)
    except Exception:
        # convert and return bytes together
        width, height, rgba = convert_aif_to_rgba(header, clut_bytes, img_bytes)

    # Note: convert_aif_to_rgba returns (width,height,bytes)
    if isinstance(rgba, type(None)):
        # second call returned differently above; ensure we have final bytes
        width, height, rgba = convert_aif_to_rgba(header, clut_bytes, img_bytes)

    img = Image.frombytes('RGBA', (width, height), rgba)
    img.save(outfile)
    print("Saved {}x{} PNG to {}".format(width, height, outfile))

if __name__ == '__main__':
    main()