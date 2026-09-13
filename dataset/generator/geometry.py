"""Geometry: camera model, homographies and the exact annotation quad.

A plate is a plane.  Seen by a pinhole camera from any angle, a plane maps to
the image by a *homography*, so four corner correspondences define the whole
mapping exactly -- not just for the plate but for the vehicle panel it is
mounted on, which lies in the same plane.

The pipeline is:

1. rotate the plate (yaw, pitch, roll) in 3D and project it with a pinhole
   camera, giving the image positions of its four corners;
2. solve the homography taking the canvas's plate corners to those points;
3. warp the canvas with that homography and map the plate corners through the
   *same* matrix to get the annotation quad.

The quad is therefore computed from the transform itself, never estimated
from pixels.  Coordinates are continuous: pixel ``(i, j)`` covers
``[i, i+1) x [j, j+1)``, which is also how PIL's perspective transform
samples (verified empirically before this module was written).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Points = np.ndarray  # N x 2, float64


def homography(src: Points, dst: Points) -> np.ndarray:
    """The 3x3 homography mapping four ``src`` points onto ``dst``."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("homography needs exactly four point pairs")
    rows = []
    rhs = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.extend([u, v])
    solution = np.linalg.solve(np.asarray(rows), np.asarray(rhs))
    return np.append(solution, 1.0).reshape(3, 3)


def apply_homography(matrix: np.ndarray, points: Points) -> Points:
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.hstack([points, np.ones((len(points), 1))]) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def pil_perspective_coefficients(matrix: np.ndarray) -> tuple[float, ...]:
    """Coefficients for ``Image.transform(..., Image.PERSPECTIVE, ...)``.

    PIL wants the *inverse* mapping (output pixel -> input pixel), normalised
    so its last element is 1.
    """
    inverse = np.linalg.inv(matrix)
    inverse = inverse / inverse[2, 2]
    return tuple(float(value) for value in inverse.flatten()[:8])


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """``Rz(roll) @ Ry(yaw) @ Rx(pitch)`` -- x right, y down, z into the scene."""
    yaw, pitch, roll = (math.radians(a) for a in (yaw_deg, pitch_deg, roll_deg))
    rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    rz = np.array([[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
    return rz @ ry @ rx


def project_plate_corners(
    width_mm: float,
    height_mm: float,
    *,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    px_per_mm: float,
    distance_ratio: float,
) -> Points:
    """Image-plane corners (TL, TR, BR, BL) relative to the plate centre.

    ``px_per_mm`` is the scale a fronto-parallel plate would have; the focal
    length is chosen to give exactly that at the plate's distance, so plate
    *size* and *perspective strength* are controlled independently.
    """
    distance = distance_ratio * width_mm
    focal = px_per_mm * distance
    half_w, half_h = width_mm / 2.0, height_mm / 2.0
    corners = np.array(
        [[-half_w, -half_h, 0.0], [half_w, -half_h, 0.0], [half_w, half_h, 0.0], [-half_w, half_h, 0.0]]
    )
    camera = corners @ rotation_matrix(yaw_deg, pitch_deg, roll_deg).T
    camera[:, 2] += distance
    if np.any(camera[:, 2] <= 1e-6):
        raise ValueError("plate corner behind the camera")
    # The plate centre sits on the optical axis, so it projects to (0, 0).
    return focal * camera[:, :2] / camera[:, 2:3]


@dataclass(frozen=True)
class Placement:
    """Where the plate lands in the output image."""

    quad: Points  # TL, TR, BR, BL in output pixels
    matrix: np.ndarray  # canvas pixels -> output pixels


def place_plate(
    canvas_corners: Points,
    relative_quad: Points,
    image_size: tuple[int, int],
    rng: np.random.Generator,
    *,
    margin: float,
    min_centre_y: float | None = None,
) -> Placement:
    """Translate ``relative_quad`` to a random position fully inside the image.

    ``min_centre_y`` optionally keeps the plate centre at or below a line --
    used to stand the vehicle on the scene's ground rather than in the sky.
    It is ignored when it cannot be met with the plate still in frame.

    Raises ``ValueError`` if the quad cannot fit; the caller shrinks it.
    """
    width, height = image_size
    lo = relative_quad.min(axis=0)
    hi = relative_quad.max(axis=0)
    x_min, x_max = margin - lo[0], width - margin - hi[0]
    y_min, y_max = margin - lo[1], height - margin - hi[1]
    if x_min > x_max or y_min > y_max:
        raise ValueError("plate does not fit in the image")
    if min_centre_y is not None and min_centre_y <= y_max:
        y_min = max(y_min, min_centre_y)
    # Plates sit more often in the middle and lower part of a frame.
    centre = np.array(
        [
            x_min + (x_max - x_min) * float(rng.beta(2.0, 2.0)),
            y_min + (y_max - y_min) * float(rng.beta(2.2, 1.8)),
        ]
    )
    quad = relative_quad + centre
    return Placement(quad=quad, matrix=homography(canvas_corners, quad))


def quad_is_clockwise(quad: Points) -> bool:
    """Clockwise on screen (y down) <=> positive shoelace area."""
    area = 0.0
    for i in range(4):
        x1, y1 = quad[i]
        x2, y2 = quad[(i + 1) % 4]
        area += x1 * y2 - x2 * y1
    return area > 0


def quad_is_convex(quad: Points) -> bool:
    signs = []
    for i in range(4):
        a, b, c = quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(cross > 0)
    return all(signs) or not any(signs)


def local_scale(quad: Points, width_mm: float, height_mm: float) -> float:
    """Geometric-mean output px per mm along the quad's edges."""
    top = np.linalg.norm(quad[1] - quad[0])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])
    horizontal = max(top, bottom) / width_mm
    vertical = max(left, right) / height_mm
    return math.sqrt(horizontal * vertical)
