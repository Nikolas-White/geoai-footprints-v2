"""Orthogonal regularization of building footprints.

This module is the difference between a segmentation demo and something a GIS
professional would accept. A raw contour traced from a probability raster has
hundreds of vertices and corners at 87.3 degrees. Real buildings are
overwhelmingly rectilinear, and a hand-digitised footprint reflects that.

Method
------
1. Estimate the building's dominant axis from a length-weighted histogram of
   edge orientations, taken modulo 90 degrees. Long walls dominate, short
   noisy segments do not.
2. Rotate the polygon into that frame.
3. Snap: any edge within ``angle_snap_deg`` of horizontal or vertical is
   forced exactly axis-parallel by moving its two endpoints to a shared
   coordinate. Because vertices are shared between consecutive edges, the
   constraints conflict, so we accumulate all proposed positions per vertex
   and average, then repeat. Two or three passes converge.
4. Rotate back and repair topology.

Edges that are genuinely oblique (a diagonal wing, a curved apartment block)
fall outside the snap tolerance and are deliberately left alone. Forcing
everything to 90 degrees would be worse than not regularizing at all.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

try:  # shapely >= 2.0
    from shapely import make_valid
except ImportError:  # pragma: no cover
    make_valid = None


# --------------------------------------------------------------------------- #
def _ring_coords(ring) -> np.ndarray:
    """Closed ring -> open (N, 2) array with the duplicate last point dropped."""
    c = np.asarray(ring.coords, dtype=np.float64)
    if len(c) > 1 and np.allclose(c[0], c[-1]):
        c = c[:-1]
    return c


def dominant_angle(coords: np.ndarray) -> float:
    """Length-weighted modal edge orientation in radians, within [0, pi/2)."""
    if len(coords) < 3:
        return 0.0
    d = np.roll(coords, -1, axis=0) - coords
    lengths = np.hypot(d[:, 0], d[:, 1])
    keep = lengths > 1e-9
    if not keep.any():
        return 0.0
    d, lengths = d[keep], lengths[keep]

    ang = np.arctan2(d[:, 1], d[:, 0]) % (np.pi / 2)  # fold onto [0, 90)

    # Circular mean on the doubled-quadruple angle avoids the wrap-around bias
    # a plain histogram would suffer near 0 and 90 degrees.
    theta = 4.0 * ang
    x = float((lengths * np.cos(theta)).sum())
    y = float((lengths * np.sin(theta)).sum())
    if abs(x) < 1e-12 and abs(y) < 1e-12:
        return 0.0
    return (math.atan2(y, x) / 4.0) % (np.pi / 2)


def _rotate(coords: np.ndarray, theta: float, origin: np.ndarray) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    r = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (coords - origin) @ r.T + origin


def _remove_short_edges(coords: np.ndarray, min_edge: float) -> np.ndarray:
    """Collapse edges shorter than ``min_edge`` by keeping one endpoint.

    Douglas-Peucker leaves behind stubs a few centimetres long where a wall
    wobbles. Those stubs are what make orientation classification unstable, so
    they go before any snapping happens.
    """
    if min_edge <= 0 or len(coords) < 4:
        return coords
    kept = [coords[0]]
    for p in coords[1:]:
        if np.hypot(*(p - kept[-1])) >= min_edge:
            kept.append(p)
    # the closing edge back to the start must also be long enough
    while len(kept) > 3 and np.hypot(*(kept[-1] - kept[0])) < min_edge:
        kept.pop()
    return np.asarray(kept) if len(kept) >= 3 else coords


def _classify_edges(pts: np.ndarray, snap_rad: float) -> List[str]:
    """Label every edge H (near-horizontal), V (near-vertical) or O (oblique)."""
    n = len(pts)
    cls = []
    for i in range(n):
        d = pts[(i + 1) % n] - pts[i]
        if abs(d[0]) < 1e-12 and abs(d[1]) < 1e-12:
            cls.append("O")
            continue
        ang = math.atan2(d[1], d[0]) % math.pi
        horiz = min(ang, math.pi - ang)
        vert = abs(ang - math.pi / 2)
        if horiz <= snap_rad and horiz <= vert:
            cls.append("H")
        elif vert <= snap_rad:
            cls.append("V")
        else:
            cls.append("O")
    return cls


def _runs(cls: List[str]) -> List[Tuple[str, List[int]]]:
    """Group consecutive same-class edges, wrapping around the ring."""
    n = len(cls)
    if n == 0:
        return []
    if len(set(cls)) == 1:
        return [(cls[0], list(range(n)))]
    start = next(i for i in range(n) if cls[i] != cls[i - 1])
    out, cur, kind = [], [], cls[start]
    for k in range(n):
        i = (start + k) % n
        if cls[i] == kind:
            cur.append(i)
        else:
            out.append((kind, cur))
            kind, cur = cls[i], [i]
    out.append((kind, cur))
    return out


def _snap_ring(
    coords: np.ndarray, snap_rad: float, min_edge: float = 0.0, passes: int = 3
) -> np.ndarray:
    """Force near-axis edges exactly axis-parallel, in the rotated frame.

    Snapping works on *runs* of consecutive same-orientation edges rather than
    one edge at a time. A wall that Douglas-Peucker broke into three slightly
    different near-horizontal segments becomes a single straight line at one
    shared y, instead of a staircase. Because a horizontal run only rewrites y
    and a vertical run only rewrites x, the two never fight over the corner
    vertex they share, and one pass already converges.
    """
    pts = coords.copy()
    for _ in range(passes):
        pts = _remove_short_edges(pts, min_edge)
        n = len(pts)
        if n < 4:
            break

        cls = _classify_edges(pts, snap_rad)
        new = pts.copy()

        for kind, edges in _runs(cls):
            if kind == "O" or not edges:
                continue
            axis = 1 if kind == "H" else 0   # H fixes y, V fixes x
            weight, total = 0.0, 0.0
            verts = set()
            for i in edges:
                j = (i + 1) % n
                d = pts[j] - pts[i]
                length = float(np.hypot(d[0], d[1]))
                total += length * (pts[i, axis] + pts[j, axis]) / 2.0
                weight += length
                verts.update((i, j))
            if weight <= 0:
                continue
            value = total / weight
            for v in verts:
                new[v, axis] = value

        pts = _drop_collinear(new, tol_deg=5.0)
        if len(pts) < 4:
            break

    return pts


def _drop_collinear(coords: np.ndarray, tol_deg: float = 2.0) -> np.ndarray:
    """Remove vertices whose two adjacent edges are nearly parallel."""
    n = len(coords)
    if n < 4:
        return coords
    keep = []
    tol = math.radians(tol_deg)
    for i in range(n):
        p, c, q = coords[i - 1], coords[i], coords[(i + 1) % n]
        v1, v2 = c - p, q - c
        n1, n2 = np.hypot(*v1), np.hypot(*v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cosang = float(np.clip((v1 @ v2) / (n1 * n2), -1.0, 1.0))
        if math.acos(cosang) > tol:
            keep.append(i)
    return coords[keep] if len(keep) >= 3 else coords


# --------------------------------------------------------------------------- #
def regularize_polygon(
    poly: Polygon,
    angle_snap_deg: float = 10.0,
    hole_min_area: float = 0.0,
    min_edge_m: float = 1.0,
) -> Optional[Polygon]:
    """Return an orthogonally regularized copy of ``poly``, or None if degenerate."""
    if poly.is_empty or len(_ring_coords(poly.exterior)) < 4:
        return poly if not poly.is_empty else None

    ext = _ring_coords(poly.exterior)
    theta = dominant_angle(ext)
    origin = ext.mean(axis=0)
    snap_rad = math.radians(angle_snap_deg)

    def process(coords: np.ndarray) -> Optional[np.ndarray]:
        if len(coords) < 4:
            return None
        rot = _rotate(coords, -theta, origin)
        rot = _snap_ring(rot, snap_rad, min_edge_m)
        rot = _drop_collinear(rot)
        if len(rot) < 3:
            return None
        return _rotate(rot, theta, origin)

    new_ext = process(ext)
    if new_ext is None:
        return None

    new_holes: List[np.ndarray] = []
    for ring in poly.interiors:
        hc = _ring_coords(ring)
        if hole_min_area > 0 and Polygon(hc).area < hole_min_area:
            continue
        h = process(hc)
        if h is not None and len(h) >= 3:
            new_holes.append(h)

    try:
        out = Polygon(new_ext, new_holes)
    except Exception:
        return None

    if not out.is_valid and make_valid is not None:
        fixed = make_valid(out)
        # make_valid can return a collection; keep the largest polygonal part
        if fixed.geom_type == "Polygon":
            out = fixed
        elif hasattr(fixed, "geoms"):
            parts = [g for g in fixed.geoms if g.geom_type == "Polygon"]
            if not parts:
                return None
            out = max(parts, key=lambda g: g.area)
        else:
            return None

    if out.is_empty or out.area <= 0:
        return None
    return out


# --------------------------------------------------------------------------- #
def corner_angles(poly: Polygon) -> np.ndarray:
    """Interior angles at every exterior vertex, in degrees.

    Used as a quality metric: hand-digitised footprints show a sharp spike at
    90 degrees, blobby model output shows a broad smear.
    """
    c = _ring_coords(poly.exterior)
    if len(c) < 3:
        return np.array([])
    prev = np.roll(c, 1, axis=0)
    nxt = np.roll(c, -1, axis=0)
    v1, v2 = prev - c, nxt - c
    n1 = np.hypot(v1[:, 0], v1[:, 1])
    n2 = np.hypot(v2[:, 0], v2[:, 1])
    ok = (n1 > 1e-9) & (n2 > 1e-9)
    if not ok.any():
        return np.array([])
    cosang = np.clip(
        (v1[ok] * v2[ok]).sum(axis=1) / (n1[ok] * n2[ok]), -1.0, 1.0
    )
    return np.degrees(np.arccos(cosang))


def orthogonality_score(poly: Polygon, tol_deg: float = 10.0) -> float:
    """Fraction of corners within ``tol_deg`` of 90 or 180 degrees."""
    ang = corner_angles(poly)
    if len(ang) == 0:
        return 0.0
    near90 = np.abs(ang - 90.0) <= tol_deg
    near180 = np.abs(ang - 180.0) <= tol_deg
    return float((near90 | near180).mean())
