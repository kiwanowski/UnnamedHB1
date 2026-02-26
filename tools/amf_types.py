"""Shared data structures for AMF/AAMF format used by UnnamedHB1 PS1 homebrew tools."""
import struct
from dataclasses import dataclass
from typing import Tuple

SVECTOR_SIZE = 8
CVECTOR_SIZE = 4
POINTER_SIZE = 4
FIXED_POINT_SCALE = 4096.0
POLY_G4_SIZE = 4 + 32   # tag + 4*(color + xy) = 36
PG4_SIZE = 4 * SVECTOR_SIZE + 4 * SVECTOR_SIZE + POLY_G4_SIZE  # 100


@dataclass
class SVECTOR:
    """Short vector (8 bytes) - used for vertices and normals"""
    vx: int  # int16_t
    vy: int  # int16_t
    vz: int  # int16_t
    pad: int  # int16_t (flags: depth check type, subdiv depth, culling mode)

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> 'SVECTOR':
        vx, vy, vz, pad = struct.unpack_from('<hhhh', data, offset)
        return cls(vx, vy, vz, pad)

    def to_float(self, scale: float = 1.0 / FIXED_POINT_SCALE) -> Tuple[float, float, float]:
        """Convert fixed-point to floating point coordinates"""
        return (self.vx * scale, self.vy * scale, self.vz * scale)

    def to_bytes(self) -> bytes:
        return struct.pack('<hhhh', self.vx, self.vy, self.vz, self.pad)

    @classmethod
    def from_float(cls, x: float, y: float, z: float, scale: float, pad: int = 0) -> 'SVECTOR':
        """Convert floating point coordinates to fixed-point"""
        vx = max(-32768, min(32767, int(round(x * scale))))
        vy = max(-32768, min(32767, int(round(y * scale))))
        vz = max(-32768, min(32767, int(round(z * scale))))
        return cls(vx, vy, vz, pad)

    def __str__(self) -> str:
        fx, fy, fz = self.to_float()
        return f"({self.vx}, {self.vy}, {self.vz}, pad={self.pad}) -> ({fx:.4f}, {fy:.4f}, {fz:.4f})"


@dataclass
class CVECTOR:
    """Color vector (4 bytes)"""
    r: int   # uint8_t
    g: int   # uint8_t
    b: int   # uint8_t
    cd: int  # uint8_t (code/alpha)

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> 'CVECTOR':
        r, g, b, cd = struct.unpack_from('<BBBB', data, offset)
        return cls(r, g, b, cd)

    def to_float(self) -> Tuple[float, float, float, float]:
        """Convert to normalized float RGBA"""
        return (self.r / 255.0, self.g / 255.0, self.b / 255.0, 1.0)

    def __str__(self) -> str:
        return f"RGBA({self.r}, {self.g}, {self.b}, {self.cd})"


@dataclass
class WorldBounds:
    """World bounding box (16 bytes)"""
    minX: int  # int32_t
    minZ: int  # int32_t
    maxX: int  # int32_t
    maxZ: int  # int32_t

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> 'WorldBounds':
        minX, minZ, maxX, maxZ = struct.unpack_from('<iiii', data, offset)
        return cls(minX, minZ, maxX, maxZ)


@dataclass
class AMFHeader:
    """AMF file header (24 bytes)"""
    used_textures: int  # uint32_t (highest bit for Texture* already set)
    x: int              # uint16_t (chunk count in X)
    z: int              # uint16_t (chunk count in Z)
    bounds: WorldBounds

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> 'AMFHeader':
        used_textures, x, z = struct.unpack_from('<IHH', data, offset)
        bounds = WorldBounds.from_bytes(data, offset + 8)
        return cls(used_textures & 0x7FFFFFFF, x, z, bounds)
