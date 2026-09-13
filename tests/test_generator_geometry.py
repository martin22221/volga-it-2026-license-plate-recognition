"""Homographies, the camera model and quad/bbox derivation."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from dataset.generator.annotations import bbox_from_quad, derive_conditions, masked_plate_number
from dataset.generator.geometry import (
    apply_homography,
    homography,
    pil_perspective_coefficients,
    place_plate,
    project_plate_corners,
    quad_is_clockwise,
    quad_is_convex,
)


def test_homography_maps_the_four_points_exactly() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        src = np.array([[0, 0], [100, 0], [100, 40], [0, 40]], dtype=float)
        dst = src + rng.uniform(-15, 15, size=(4, 2)) + rng.uniform(0, 300, size=2)
        matrix = homography(src, dst)
        np.testing.assert_allclose(apply_homography(matrix, src), dst, atol=1e-9)


def test_homography_round_trip() -> None:
    src = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=float)
    dst = np.array([[3, 4], [20, 2], [22, 14], [1, 12]], dtype=float)
    matrix = homography(src, dst)
    points = np.array([[2.5, 1.0], [7.0, 4.0], [5.0, 2.5]])
    back = apply_homography(np.linalg.inv(matrix), apply_homography(matrix, points))
    np.testing.assert_allclose(back, points, atol=1e-9)


def test_pil_warp_agrees_with_the_matrix() -> None:
    """The annotation quad and the pixels must use the same convention."""
    src_image = Image.new("F", (200, 100), 0.0)
    src_image.paste(1.0, (50, 30, 150, 70))  # block [50,150) x [30,70)
    corners = np.array([[50, 30], [150, 30], [150, 70], [50, 70]], dtype=float)
    target = np.array([[112.0, 83.0], [305.0, 91.0], [296.0, 170.0], [120.0, 160.0]])
    matrix = homography(corners, target)
    warped = src_image.transform((400, 250), Image.PERSPECTIVE, pil_perspective_coefficients(matrix), Image.BILINEAR)
    mask = np.asarray(warped) > 0.5
    ys, xs = np.nonzero(mask)
    assert abs(xs.min() - target[:, 0].min()) <= 1.0
    assert abs(xs.max() + 1 - target[:, 0].max()) <= 1.0
    assert abs(ys.min() - target[:, 1].min()) <= 1.0
    assert abs(ys.max() + 1 - target[:, 1].max()) <= 1.0
    # Area agrees with the quad's shoelace area to within 1 %.
    x, y = target[:, 0], target[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert abs(float(np.asarray(warped).sum()) - area) / area < 0.01


def test_frontal_projection_is_the_scaled_rectangle() -> None:
    quad = project_plate_corners(520, 112, yaw_deg=0, pitch_deg=0, roll_deg=0, px_per_mm=0.5, distance_ratio=10)
    np.testing.assert_allclose(quad, [[-130, -28], [130, -28], [130, 28], [-130, 28]], atol=1e-9)


def test_roll_rotates_without_changing_size() -> None:
    quad = project_plate_corners(520, 112, yaw_deg=0, pitch_deg=0, roll_deg=30, px_per_mm=0.5, distance_ratio=10)
    assert np.linalg.norm(quad[1] - quad[0]) == pytest.approx(260)
    assert np.linalg.norm(quad[3] - quad[0]) == pytest.approx(56)


def test_yaw_foreshortens_horizontally() -> None:
    quad = project_plate_corners(520, 112, yaw_deg=45, pitch_deg=0, roll_deg=0, px_per_mm=0.5, distance_ratio=10)
    width = quad[:, 0].max() - quad[:, 0].min()
    assert width < 260 * 0.8
    # Perspective: the nearer edge is taller than the farther one.
    left, right = np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])
    assert left != pytest.approx(right)


@pytest.mark.parametrize("seed", range(20))
def test_random_poses_keep_the_quad_clockwise_and_convex(seed: int) -> None:
    rng = np.random.default_rng(seed)
    quad = project_plate_corners(
        520, 112,
        yaw_deg=rng.uniform(-60, 60), pitch_deg=rng.uniform(-30, 30), roll_deg=rng.uniform(-20, 20),
        px_per_mm=rng.uniform(0.1, 0.6), distance_ratio=rng.uniform(3, 20),
    )
    assert quad_is_clockwise(quad)
    assert quad_is_convex(quad)


def test_place_plate_keeps_the_quad_inside_the_image() -> None:
    rng = np.random.default_rng(3)
    relative = project_plate_corners(520, 112, yaw_deg=20, pitch_deg=5, roll_deg=3, px_per_mm=0.4, distance_ratio=8)
    canvas = np.array([[10, 10], [218, 10], [218, 55], [10, 55]], dtype=float)
    for _ in range(50):
        placement = place_plate(canvas, relative, (640, 480), rng, margin=5)
        assert placement.quad[:, 0].min() >= 5 and placement.quad[:, 0].max() <= 635
        assert placement.quad[:, 1].min() >= 5 and placement.quad[:, 1].max() <= 475
        np.testing.assert_allclose(apply_homography(placement.matrix, canvas), placement.quad, atol=1e-6)


def test_place_plate_refuses_an_oversized_plate() -> None:
    relative = project_plate_corners(520, 112, yaw_deg=0, pitch_deg=0, roll_deg=0, px_per_mm=2.0, distance_ratio=8)
    canvas = np.array([[0, 0], [1040, 0], [1040, 224], [0, 224]], dtype=float)
    with pytest.raises(ValueError):
        place_plate(canvas, relative, (640, 480), np.random.default_rng(0), margin=5)


def test_bbox_is_the_tight_box_of_the_quad() -> None:
    quad = ((10.0, 20.0), (110.5, 15.25), (112.0, 40.0), (8.75, 44.5))
    assert bbox_from_quad(quad) == (8.75, 15.25, 103.25, 29.25)


def test_masked_plate_number() -> None:
    assert masked_plate_number("A123BC77", {1, 7}) == "A#23BC7#"
    assert masked_plate_number("A123BC77", set()) == "A123BC77"


def test_condition_tags_follow_the_vocabulary() -> None:
    tags = derive_conditions(
        night=True, yaw_deg=20, pitch_deg=0, roll_deg=0,
        effects={
            "dirt": {"strength": 0.5},
            "glare": {"on_plate": True},
            "motion_blur": {"length_px": 5.0},
            "rain": {"density": 0.4},
        },
    )
    assert tags == ("night", "rain", "dirt", "glare", "motion_blur", "angle")
    calm = derive_conditions(night=False, yaw_deg=5, pitch_deg=-3, roll_deg=2, effects={
        "dirt": {"strength": 0.1}, "glare": {"on_plate": False}, "motion_blur": {"length_px": 1.5},
    })
    assert calm == ("day",)


def test_place_plate_honours_a_ground_line_when_it_can() -> None:
    rng = np.random.default_rng(8)
    relative = project_plate_corners(520, 112, yaw_deg=0, pitch_deg=0, roll_deg=0, px_per_mm=0.3, distance_ratio=8)
    canvas = np.array([[0, 0], [156, 0], [156, 34], [0, 34]], dtype=float)
    for _ in range(50):
        placement = place_plate(canvas, relative, (640, 480), rng, margin=5, min_centre_y=300.0)
        assert placement.quad.mean(axis=0)[1] >= 300.0 - 1e-9
    # An unreachable line is ignored rather than pushing the plate out of frame.
    placement = place_plate(canvas, relative, (640, 480), rng, margin=5, min_centre_y=10_000.0)
    assert placement.quad[:, 1].max() <= 475
