from __future__ import annotations

from tests.test_route_roadblock_repair import FakePoint, FakePolygon, _install_extract_single_frame_import_stubs


_install_extract_single_frame_import_stubs()

from src.platform.nuplan.features import extract_single_frame as esf


class FakeNode:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class FakePath:
    def __init__(self, coords: list[tuple[float, float]]):
        self.discrete_path = [FakeNode(x, y) for x, y in coords]


class FakeRouteLane:
    def __init__(self, lane_id: str, roadblock_id: str, centerline: list[tuple[float, float]]):
        self.id = lane_id
        self._roadblock_id = roadblock_id
        self.baseline_path = FakePath(centerline)
        self.left_boundary = FakePath([(x, y + 1.0) for x, y in centerline])
        self.right_boundary = FakePath([(x, y - 1.0) for x, y in centerline])
        cx = sum(x for x, _ in centerline) / len(centerline)
        cy = sum(y for _, y in centerline) / len(centerline)
        self.polygon = FakePolygon(cx, cy)
        self.speed_limit_mps = None

    def get_roadblock_id(self) -> str:
        return self._roadblock_id


class FakeRouteMap:
    def __init__(self, lanes: list[FakeRouteLane]):
        self._lanes = lanes

    def get_proximal_map_objects(self, point, radius, layers):
        return {
            esf.SemanticMapLayer.LANE: self._lanes,
            esf.SemanticMapLayer.LANE_CONNECTOR: [],
        }


def _first_valid_x(route_lanes, route_lanes_avails, slot_idx: int) -> float:
    valid = route_lanes_avails[slot_idx] > 0
    return float(route_lanes[slot_idx, valid, 0][0])


def _slot_is_valid(route_lanes_avails, slot_idx: int) -> bool:
    return bool((route_lanes_avails[slot_idx] > 0).any())


def test_route_lanes_follow_route_roadblock_order_before_distance() -> None:
    lanes = [
        FakeRouteLane("lane_d", "D", [(1.0, 0.0), (2.0, 0.0)]),
        FakeRouteLane("lane_b", "B", [(20.0, 0.0), (30.0, 0.0)]),
        FakeRouteLane("lane_c", "C", [(30.0, 0.0), (40.0, 0.0)]),
    ]

    route_lanes, route_lanes_avails, _, _ = esf.extract_route_lanes(
        FakePoint(),
        FakeRouteMap(lanes),
        max_route_lanes=3,
        route_roadblock_ids=["B", "C", "D"],
    )

    assert _first_valid_x(route_lanes, route_lanes_avails, 0) == 20.0
    assert _first_valid_x(route_lanes, route_lanes_avails, 1) == 30.0
    assert _first_valid_x(route_lanes, route_lanes_avails, 2) == 1.0


def test_route_lanes_use_distance_within_same_roadblock() -> None:
    lanes = [
        FakeRouteLane("lane_b_near", "B", [(10.0, 0.0), (20.0, 0.0)]),
        FakeRouteLane("lane_b_far", "B", [(20.0, 0.0), (30.0, 0.0)]),
        FakeRouteLane("lane_c", "C", [(30.0, 0.0), (40.0, 0.0)]),
    ]

    route_lanes, route_lanes_avails, _, _ = esf.extract_route_lanes(
        FakePoint(),
        FakeRouteMap(lanes),
        max_route_lanes=3,
        route_roadblock_ids=["B", "C"],
    )

    assert _first_valid_x(route_lanes, route_lanes_avails, 0) == 10.0
    assert _first_valid_x(route_lanes, route_lanes_avails, 1) == 20.0
    assert _first_valid_x(route_lanes, route_lanes_avails, 2) == 30.0


def test_route_lanes_drop_route_islands_not_connected_to_ego_corridor() -> None:
    lanes = [
        FakeRouteLane("lane_a", "A", [(0.0, 0.0), (8.0, 0.0)]),
        FakeRouteLane("lane_b", "B", [(8.0, 0.0), (16.0, 0.0)]),
        FakeRouteLane("lane_c_far", "C", [(100.0, 0.0), (108.0, 0.0)]),
    ]

    route_lanes, route_lanes_avails, _, _ = esf.extract_route_lanes(
        FakePoint(),
        FakeRouteMap(lanes),
        max_route_lanes=3,
        route_roadblock_ids=["A", "B", "C"],
    )

    assert _first_valid_x(route_lanes, route_lanes_avails, 0) == 0.0
    assert _first_valid_x(route_lanes, route_lanes_avails, 1) == 8.0
    assert _slot_is_valid(route_lanes_avails, 2) is False


def test_route_lanes_stop_when_forward_corridor_reverses() -> None:
    lanes = [
        FakeRouteLane("lane_a", "A", [(0.0, 0.0), (10.0, 0.0)]),
        FakeRouteLane("lane_b", "B", [(10.0, 0.0), (80.0, 0.0)]),
        FakeRouteLane("lane_c_reverse", "C", [(80.0, 0.0), (20.0, 0.0)]),
    ]

    route_lanes, route_lanes_avails, _, _ = esf.extract_route_lanes(
        FakePoint(),
        FakeRouteMap(lanes),
        max_route_lanes=3,
        ego_heading=0.0,
        route_roadblock_ids=["A", "B", "C"],
    )

    assert _first_valid_x(route_lanes, route_lanes_avails, 0) == 0.0
    assert _first_valid_x(route_lanes, route_lanes_avails, 1) == 10.0
    assert _slot_is_valid(route_lanes_avails, 2) is False


def test_route_lanes_keep_all_when_no_ego_connected_component_exists() -> None:
    lanes = [
        FakeRouteLane("lane_a_far", "A", [(80.0, 0.0), (88.0, 0.0)]),
        FakeRouteLane("lane_b_far", "B", [(88.0, 0.0), (96.0, 0.0)]),
    ]

    route_lanes, route_lanes_avails, _, _ = esf.extract_route_lanes(
        FakePoint(),
        FakeRouteMap(lanes),
        max_route_lanes=2,
        route_roadblock_ids=["A", "B"],
    )

    assert _slot_is_valid(route_lanes_avails, 0) is True
    assert _slot_is_valid(route_lanes_avails, 1) is True


def test_route_lanes_empty_route_ids_return_no_route_lanes() -> None:
    lanes = [
        FakeRouteLane("lane_a", "A", [(0.0, 0.0), (8.0, 0.0)]),
        FakeRouteLane("lane_b", "B", [(8.0, 0.0), (16.0, 0.0)]),
    ]

    route_lanes, route_lanes_avails, _, _ = esf.extract_route_lanes(
        FakePoint(),
        FakeRouteMap(lanes),
        max_route_lanes=2,
        route_roadblock_ids=[],
    )

    assert int(route_lanes_avails.sum()) == 0
    assert float(route_lanes.sum()) == 0.0
