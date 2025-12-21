#!/usr/bin/env python3
"""
png2aif.py — converter from PNG -> .aif (format used in achrostel/UnnamedHB1)

Usage:
    python png2aif.py input.png output.aif [--mode auto|4|8|16] [--size auto|64|128|256] [--name NAME]
                                           [--fit pad|resize|fail] [--transparency auto|0|1]

Behavior:
- Header layout (20 bytes):
    name[8] (ASCII, padded with NULs)
    mode (uint8): 0 = 4-bit paletted, 1 = 8-bit paletted, 2 = 16-bit direct color
    transparency (uint8): 0 or 1 (simple flag; engine uses bits, see repo)
    size (uint8): 0 -> 64, 1 -> 128, 2 -> 256 (height = 64 << size)
    clut_len (uint8): written as palette_count - 1 (engine uses clut_len+1 when uploading)
    img_data_len (uint32 LE)
    clut_data_len (uint32 LE)
  Followed by CLUT data then image data.

- Modes:
    16-bit: each pixel -> PSX 15-bit (2 bytes, little-endian)
    8-bit: 1 byte per pixel (index into CLUT)
    4-bit: 2 pixels per byte; low nibble = first pixel, high nibble = second pixel

- Palette/CLUT entries are 2-byte little-endian PSX 15-bit values (R,G,B 5 bits each).
- If mode is 'auto' the converter will try:
    * Quantize to <=16 colors -> use 4-bit
    * Else <=256 colors -> use 8-bit
    * Else use 16-bit

Notes / caveats:
- The AIF format in the repo expects heights of 64, 128 or 256. By default the script chooses the smallest supported height >= input height and pads the image at the bottom with transparent pixels. Use --fit resize to scale vertically to target height instead, or --fit fail to abort if the input height isn't one of the supported values.
- Transparency handling is simple: if any pixel has alpha < 255, the script sets the transparency flag to 1 and tries to make index 0 (if paletted) the fully transparent color when quantizing.
- You can force mode/size with CLI flags.
- Requires Pillow: pip install pillow
"""

import argparse
import struct
from PIL import Image

SUPPORTED_HEIGHTS = [64, 128, 256]

def rgb_to_psx15(r, g, b):
    """Pack 8-bit RGB into PSX 15-bit (5-5-5) format as uint16."""
    r5 = r >> 3
    g5 = g >> 3
    b5 = b >> 3
    val = (b5 << 10) | (g5 << 5) | (r5)
    return val

def psx15_to_bytes(val):
    return struct.pack('<H', val)

def pad_or_resize(image, target_height, fit):
    w, h = image.size
    if h == target_height:
        return image
    if fit == 'fail':
        raise SystemExit(f"Image height {h} not supported and --fit fail given.")
    if fit == 'resize':
        # preserve width, scale height to target_height
        return image.resize((w, target_height), Image.LANCZOS)
    # default: pad bottom with fully transparent pixels
    new = Image.new('RGBA', (w, target_height), (0,0,0,0))
    new.paste(image, (0,0))
    return new

def infer_mode_from_palette(img):
    # img must be an RGBA image
    # try quantize to 16 colors then 256 colors
    # we try to keep a transparent color if needed
    has_alpha = any(px[3] < 255 for px in img.getdata())
    if has_alpha:
        # To keep a transparent index, we will create an image with a matte color for transparent pixels
        # and then ensure that color is present in palette (PIL quantize can be unpredictable).
        # Simpler: use quantize and then check palette size and mapping of any transparent pixels to an index.
        pass
    # First try 16 colors
    pal16 = img.convert('RGB').quantize(colors=16, method=Image.MEDIANCUT)
    n16 = len([c for c in pal16.getpalette()[:16*3] if c is not None])  # not exact but palette length
    # Better approach: count unique colors after quantize
    unique16 = len(set(pal16.getdata()))
    if unique16 <= 16:
        return 0, pal16  # 4-bit
    pal256 = img.convert('RGB').quantize(colors=256, method=Image.MEDIANCUT)
    unique256 = len(set(pal256.getdata()))
    if unique256 <= 256:
        return 1, pal256  # 8-bit
    return 2, None  # 16-bit

def build_clut_bytes_from_palette(palette_rgb):
    """palette_rgb: list of (r,g,b) tuples in 0..255
       returns bytes of little-endian uint16 PSX entries"""
    out = bytearray()
    for (r,g,b) in palette_rgb:
        out += psx15_to_bytes(rgb_to_psx15(r,g,b))
    return bytes(out)

def image_to_aif_bytes(img, mode, size_field, name=b'', transparency_flag=0):
    """
    img: PIL RGBA image whose height must be one of 64/128/256
    mode: 0,1,2 corresponding to 4/8/16-bit
    size_field: 0/1/2 for header.size
    name: up to 8 bytes ASCII
    returns: bytes for full .aif file
    """
    w, h = img.size
    if h not in SUPPORTED_HEIGHTS or h != (64 << size_field):
        raise ValueError("Image height must equal 64<<size_field")

    # convert image pixels
    if mode == 2:
        # 16-bit direct color
        img_rgb = img.convert('RGB')
        pixels = list(img_rgb.getdata())
        img_bytes = bytearray()
        for (r,g,b) in pixels:
            img_bytes += psx15_to_bytes(rgb_to_psx15(r,g,b))
        clut_bytes = b''
        clut_len_field = 0
    else:
        # paletted
        # quantize preserving transparency: we map fully transparent pixels to a reserved color (0,0,0)
        # and ensure that color is present and placed at index 0 if any transparency exists.
        has_alpha = any(px[3] < 255 for px in img.getdata())
        if has_alpha:
            # produce an interim image where transparent pixels set to a unique color unlikely to appear
            # We'll use magenta (255,0,255) as placeholder and later ensure palette index 0 is that entry
            placeholder = (255, 0, 255)
            tmp = Image.new('RGBA', img.size)
            tmp_pixels = []
            for px in img.getdata():
                if px[3] < 128:
                    tmp_pixels.append(placeholder + (255,))
                else:
                    tmp_pixels.append(px[:3] + (255,))
            tmp.putdata(tmp_pixels)
            base = tmp.convert('RGB')
        else:
            base = img.convert('RGB')

        # quantize to target colors (16 or 256)
        target_colors = 16 if mode == 0 else 256
        pal_img = base.quantize(colors=target_colors, method=Image.MEDIANCUT)
        palette_data = pal_img.getpalette()[:target_colors*3]  # flat list
        # build palette tuples
        palette_rgb = []
        for i in range(0, len(palette_data), 3):
            palette_rgb.append((palette_data[i], palette_data[i+1], palette_data[i+2]))

        # If we had transparency, ensure placeholder color is at index 0:
        if has_alpha:
            placeholder_rgb = (255,0,255)
            try:
                idx = palette_rgb.index(placeholder_rgb)
            except ValueError:
                # placeholder not in palette: we must insert it as first entry (shift others)
                palette_rgb = [placeholder_rgb] + palette_rgb
                # chop to length target_colors if necessary
                palette_rgb = palette_rgb[:target_colors]
                # remap pal_img data indices accordingly by remapping any pixel that matched placeholder to 0
                # We'll rebuild index data from original base image to be safe
                # Build mapping color->index
                color_to_index = {c: i for i, c in enumerate(palette_rgb)}
                # fallback: re-quantize with adaptive palette forced might be complex; we'll continue with current palette
                pass
            else:
                # If placeholder is present but not at index 0, swap entries 0 and idx
                if idx != 0:
                    palette_rgb[0], palette_rgb[idx] = palette_rgb[idx], palette_rgb[0]
                    # swap palette entries in pal_img's palette structure won't change pixel indices,
                    # so we will remap indices when building img bytes below
        # Create CLUT bytes
        clut_bytes = build_clut_bytes_from_palette(palette_rgb)
        clut_len_field = max(0, len(palette_rgb) - 1)

        # Build pixel index data
        # We need the pixel indices for each pixel in the quantized image.
        # To avoid palette index vs actual color mismatches after manipulations above, regenerate per-pixel mapping:
        # Map each palette color to its index
        color_to_index = {c: i for i, c in enumerate(palette_rgb)}
        pixels_rgb = list(img.convert('RGBA').getdata())
        indices = []
        for px in pixels_rgb:
            if px[3] < 128:
                # transparent -> index 0 (we ensured placeholder at index 0)
                indices.append(0)
            else:
                # find the nearest palette color (euclidean in RGB) — fallback if exact color not found
                key = (px[0], px[1], px[2])
                if key in color_to_index:
                    indices.append(color_to_index[key])
                else:
                    # find nearest
                    best = 0
                    bestd = None
                    for i, c in enumerate(palette_rgb):
                        dr = c[0] - px[0]
                        dg = c[1] - px[1]
                        db = c[2] - px[2]
                        d = dr*dr + dg*dg + db*db
                        if bestd is None or d < bestd:
                            bestd = d
                            best = i
                    indices.append(best)

        if mode == 1:
            # 8-bit: one index per pixel
            img_bytes = bytes(indices)
        else:
            # 4-bit: pack two indices per byte, low nibble first
            b = bytearray()
            for i in range(0, len(indices), 2):
                low = indices[i] & 0x0F
                if i+1 < len(indices):
                    high = indices[i+1] & 0x0F
                else:
                    high = 0
                byte = (high << 4) | low
                b.append(byte)
            img_bytes = bytes(b)

    # Build header
    name_field = (name[:8].encode('ascii', 'ignore') if isinstance(name, str) else name)[:8]
    name_field = name_field.ljust(8, b'\x00')
    mode_byte = struct.pack('B', mode)
    transparency_byte = struct.pack('B', transparency_flag)
    size_byte = struct.pack('B', size_field)
    clut_len_byte = struct.pack('B', clut_len_field)
    img_len = len(img_bytes)
    clut_len_bytes = len(clut_bytes)
    img_len_bytes = struct.pack('<I', img_len)
    clut_len_le = struct.pack('<I', clut_len_bytes)

    header = bytearray()
    header += name_field
    header += mode_byte
    header += transparency_byte
    header += size_byte
    header += clut_len_byte
    header += img_len_bytes
    header += clut_len_le

    full = bytes(header) + clut_bytes + img_bytes
    return full

def choose_size_field(img_height, requested):
    if requested != 'auto':
        # map provided size to field
        if requested == '64':
            return 0
        if requested == '128':
            return 1
        if requested == '256':
            return 2
        raise ValueError("Unknown size option")
    # choose smallest supported height >= img_height
    for i, h in enumerate(SUPPORTED_HEIGHTS):
        if h >= img_height:
            return i
    # if bigger than largest, pick largest and caller should resize/pad
    return len(SUPPORTED_HEIGHTS) - 1

def main():
    parser = argparse.ArgumentParser(description="Convert PNG -> .aif (achrostel/UnnamedHB1 format)")
    parser.add_argument('input', help='input PNG file')
    parser.add_argument('output', help='output .aif file')
    parser.add_argument('--mode', default='auto', choices=['auto','4','8','16'],
                        help='target mode: 4=4-bit, 8=8-bit, 16=16-bit, auto tries to palette when possible')
    parser.add_argument('--size', default='auto', choices=['auto','64','128','256'],
                        help='target texture height (64/128/256) or auto')
    parser.add_argument('--name', default='tex', help='8-char name to put in header')
    parser.add_argument('--fit', default='pad', choices=['pad','resize','fail'],
                        help='what to do if input height not equal to target: pad, resize, or fail')
    parser.add_argument('--transparency', default='auto', choices=['auto','0','1'],
                        help='transparency flag in header (auto sets to 1 when input has alpha)')
    args = parser.parse_args()

    im = Image.open(args.input).convert('RGBA')
    w, h = im.size

    size_field = choose_size_field(h, args.size)
    target_height = 64 << size_field

    im2 = pad_or_resize(im, target_height, args.fit)

    # Decide mode
    chosen_mode = None
    pal_img = None
    if args.mode != 'auto':
        if args.mode == '4':
            chosen_mode = 0
        elif args.mode == '8':
            chosen_mode = 1
        elif args.mode == '16':
            chosen_mode = 2
    else:
        mode_infer, pal_img = infer_mode_from_palette(im2)
        chosen_mode = mode_infer

    transparency_flag = 0
    if args.transparency == '1':
        transparency_flag = 1
    elif args.transparency == '0':
        transparency_flag = 0
    else:
        # auto
        transparency_flag = 1 if any(px[3] < 255 for px in im2.getdata()) else 0

    aif_bytes = image_to_aif_bytes(im2, chosen_mode, size_field, name=args.name, transparency_flag=transparency_flag)

    with open(args.output, 'wb') as f:
        f.write(aif_bytes)

    print(f"Wrote {args.output}: mode={chosen_mode}, size_field={size_field} (height={target_height}), transparency={transparency_flag}, image size={im2.size}")

if __name__ == '__main__':
    main()