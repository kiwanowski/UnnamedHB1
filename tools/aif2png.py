#!/usr/bin/env python3
"""
AIF to PNG Converter

Converts AIF (Achrostel Image Format) texture files from the UnnamedHB1 project to PNG. 

AIF Format Specification:
- Header (20 bytes):
  - name[8]: 8-character texture name
  - mode (1 byte): 0=4-bit, 1=8-bit, 2=16-bit
  - transparency (1 byte): bit 0=transparent, bits 1-2=transparency mode
  - size (1 byte): 0=64x64, 1=128x128, 2=256x256
  - clut_len (1 byte): CLUT entries count
  - img_data_len (4 bytes): Image data length in bytes
  - clut_data_len (4 bytes): CLUT data length in bytes
- CLUT data (palette) follows header
- Image pixel data follows CLUT

PS1 uses BGR555 format:  5 bits each for B, G, R with 1 bit for semi-transparency
"""

import struct
import sys
import os
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow library required.  Install with: pip install Pillow")
    sys.exit(1)


def bgr555_to_rgba(color_16bit):
    """
    Convert PS1 BGR555 16-bit color to RGBA.
    PS1 format:  MBBBBBGGGGGRRRRR (M = semi-transparency bit)
    """
    r = (color_16bit & 0x1F) << 3
    g = ((color_16bit >> 5) & 0x1F) << 3
    b = ((color_16bit >> 10) & 0x1F) << 3
    
    # Semi-transparency bit (bit 15)
    # If color is 0x0000 (black with no STP bit), treat as fully transparent
    if color_16bit == 0x0000:
        a = 0
    else:
        a = 255
    
    return (r, g, b, a)


def read_aif_header(data):
    """Parse AIF header from binary data."""
    if len(data) < 20:
        raise ValueError("File too small to contain AIF header")
    
    # Parse header fields
    name = data[0:8]. decode('ascii', errors='replace').rstrip('\x00')
    mode = data[8]
    transparency = data[9]
    size = data[10]
    clut_len = data[11]
    img_data_len = struct.unpack('<I', data[12:16])[0]
    clut_data_len = struct.unpack('<I', data[16:20])[0]
    
    # Calculate dimensions from size field
    dimensions = {0: 64, 1: 128, 2: 256}
    width = height = dimensions. get(size, 64)
    
    return {
        'name': name,
        'mode': mode,
        'transparency':  transparency,
        'size': size,
        'clut_len': clut_len,
        'img_data_len': img_data_len,
        'clut_data_len': clut_data_len,
        'width': width,
        'height': height
    }


def read_clut(data, offset, clut_data_len):
    """Read CLUT (Color Look-Up Table) from binary data."""
    clut = []
    num_colors = clut_data_len // 2  # Each color is 2 bytes (16-bit)
    
    for i in range(num_colors):
        color_offset = offset + (i * 2)
        if color_offset + 2 <= len(data):
            color_16bit = struct.unpack('<H', data[color_offset: color_offset + 2])[0]
            clut.append(bgr555_to_rgba(color_16bit))
    
    return clut


def convert_4bit_image(data, offset, width, height, clut):
    """Convert 4-bit indexed image to RGBA."""
    pixels = []
    img_offset = offset
    
    for y in range(height):
        row = []
        for x in range(0, width, 2):  # 2 pixels per byte
            if img_offset < len(data):
                byte = data[img_offset]
                # Low nibble first, then high nibble
                pixel1_idx = byte & 0x0F
                pixel2_idx = (byte >> 4) & 0x0F
                
                row.append(clut[pixel1_idx] if pixel1_idx < len(clut) else (0, 0, 0, 255))
                if x + 1 < width:
                    row.append(clut[pixel2_idx] if pixel2_idx < len(clut) else (0, 0, 0, 255))
                
                img_offset += 1
            else:
                row.extend([(0, 0, 0, 255)] * 2)
        pixels.extend(row[: width])
    
    return pixels


def convert_8bit_image(data, offset, width, height, clut):
    """Convert 8-bit indexed image to RGBA."""
    pixels = []
    img_offset = offset
    
    for y in range(height):
        for x in range(width):
            if img_offset < len(data):
                pixel_idx = data[img_offset]
                if pixel_idx < len(clut):
                    pixels.append(clut[pixel_idx])
                else:
                    pixels.append((0, 0, 0, 255))
                img_offset += 1
            else:
                pixels.append((0, 0, 0, 255))
    
    return pixels


def convert_16bit_image(data, offset, width, height):
    """Convert 16-bit direct color image to RGBA."""
    pixels = []
    img_offset = offset
    
    for y in range(height):
        for x in range(width):
            if img_offset + 2 <= len(data):
                color_16bit = struct.unpack('<H', data[img_offset:img_offset + 2])[0]
                pixels.append(bgr555_to_rgba(color_16bit))
                img_offset += 2
            else:
                pixels.append((0, 0, 0, 255))
    
    return pixels


def convert_aif_to_png(input_path, output_path=None):
    """
    Convert an AIF file to PNG. 
    
    Args:
        input_path: Path to the AIF file
        output_path:  Optional output path for PNG (defaults to input_path with .png extension)
    
    Returns:
        Path to the created PNG file
    """
    input_path = Path(input_path)
    
    if output_path is None: 
        output_path = input_path. with_suffix('.png')
    else:
        output_path = Path(output_path)
    
    # Read the AIF file
    with open(input_path, 'rb') as f:
        data = f.read()
    
    # Parse header
    header = read_aif_header(data)
    
    print(f"Converting:  {input_path. name}")
    print(f"  Name: {header['name']}")
    print(f"  Mode: {['4-bit', '8-bit', '16-bit'][header['mode']]}")
    print(f"  Size: {header['width']}x{header['height']}")
    print(f"  Transparency: {header['transparency']}")
    print(f"  CLUT length:  {header['clut_len']}")
    print(f"  Image data length:  {header['img_data_len']} bytes")
    print(f"  CLUT data length: {header['clut_data_len']} bytes")
    
    # Calculate data offsets
    clut_offset = 20  # Header is 20 bytes
    img_offset = clut_offset + header['clut_data_len']
    
    # Read CLUT for indexed modes
    clut = []
    if header['mode'] in [0, 1]:  # 4-bit or 8-bit mode
        clut = read_clut(data, clut_offset, header['clut_data_len'])
        print(f"  Loaded {len(clut)} palette colors")
    
    # Convert image data to pixels
    width, height = header['width'], header['height']
    
    if header['mode'] == 0:  # 4-bit indexed
        pixels = convert_4bit_image(data, img_offset, width, height, clut)
    elif header['mode'] == 1:  # 8-bit indexed
        pixels = convert_8bit_image(data, img_offset, width, height, clut)
    elif header['mode'] == 2:  # 16-bit direct color
        pixels = convert_16bit_image(data, img_offset, width, height)
    else:
        raise ValueError(f"Unknown mode: {header['mode']}")
    
    # Create image
    img = Image.new('RGBA', (width, height))
    img. putdata(pixels)
    
    # Save as PNG
    img. save(output_path)
    print(f"  Saved:  {output_path}")
    
    return output_path


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 2:
        print("AIF to PNG Converter")
        print("Usage: python aif_to_png.py <input. aif> [output.png]")
        print("       python aif_to_png.py <directory>  # Convert all . aif files")
        sys.exit(1)
    
    input_path = Path(sys. argv[1])
    
    if input_path.is_dir():
        # Convert all AIF files in directory
        aif_files = list(input_path. glob('*.aif'))
        if not aif_files:
            print(f"No . aif files found in {input_path}")
            sys.exit(1)
        
        print(f"Found {len(aif_files)} AIF file(s)")
        for aif_file in aif_files: 
            try:
                convert_aif_to_png(aif_file)
            except Exception as e: 
                print(f"  Error converting {aif_file. name}: {e}")
    else: 
        # Convert single file
        output_path = Path(sys.argv[2]) if len(sys. argv) > 2 else None
        try:
            convert_aif_to_png(input_path, output_path)
        except Exception as e:
            print(f"Error:  {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()