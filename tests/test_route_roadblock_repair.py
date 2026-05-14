from __future__ import annotations

import sys
import types

import numpy as np


def _install_extract_single_frame_import_stubs() -> None:
    def module(name: str) -> types.ModuleType:
        stub = types.ModuleType(name)
        sys.modules.setdefault(name, stub)
        return sys.modules[name]

    class Point2D:
        def __init__(self, x: float = 0.0, y: float = 0.0):
            self.x = x
            self.y = y

    state_representation = module("nuplan.common.actor_state.state_representation")
    state_representation.Point2D = Point2D
    ego_state = module("nuplan.common.actor_state.ego_state")
    ego_state.EgoState = object
    vehicle_parameters = module("nuplan.common.actor_state.vehicle_parameters")
    vehicle_parameters.get_pacifica_parameters = lambda: None

    maps_datatypes = module("nuplan.common.maps.maps_datatypes")

    class SemanticMapLayer:
        LANE = "lane"
        LANE_CONNECTOR = "lane_connector"

    maps_datatypes.SemanticMapLayer = SemanticMapLayer

    nuplan_scenario = module("nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario")
    nuplan_scenario.NuPlanScenario = object

    roadblock_utils = module("nuplan.diffusion_planner.data_process.roadblock_utils")
    roadblock_utils.BreadthFirstSearchRoadBlock = object
    roadblock_utils.route_roadblock_correction = lambda *args, **kwargs: None

    abstract_scenario = module("nuplan.planning.scenario_builder.abstract_scenario")
    abstract_scenario.AbstractScenario = object
    observation_type = module("nuplan.planning.simulation.observation.observation_type")
    observation_type.DetectionsTracks = object
    observation_type.Observation = object
    abstract_planner = module("nuplan.planning.simulation.planner.abstract_planner")
    abstract_planner.AbstractPlanner = object
    abstract_planner.PlannerInitialization = object
    abstract_planner.PlannerInput = object
    transform_utils = module("nuplan.planning.simulation.planner.ml_planner.transform_utils")
    transform_utils.transform_predictions_to_states = lambda *args, **kwargs: []
    abstract_trajectory = module("nuplan.planning.simulation.trajectory.abstract_trajectory")
    abstract_trajectory.AbstractTrajectory = object
    interpolated_trajectory = module("nuplan.planning.simulation.trajectory.interpolated_trajectory")
    interpolated_trajectory.InterpolatedTrajectory = object
    trajectory_sampling = module("nuplan.planning.simulation.trajectory.trajectory_sampling")
    trajectory_sampling.TrajectorySampling = object


_install_extract_single_frame_import_stubs()

from src.platform.nuplan.features import extract_single_frame as esf


class FakePoint:
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y


class FakeCentroid(FakePoint):
    pass


class FakePolygon:
    def __init__(self, x: float, y: float):
        self.centroid = FakeCentroid(x, y)


class FakeLane:
    def __init__(self, roadblock_id: str, x: float, y: float, *, method: bool = True):
        self._roadblock_id = roadblock_id
        self.polygon = FakePolygon(x, y)
        if not method:
            self.roadblock_id = roadblock_id

    def get_roadblock_id(self) -> str:
        return self._roadblock_id


class FakeLaneAttrOnly:
    def __init__(self, roadblock_id: str, x: float, y: float):
        self.roadblock_id = roadblock_id
        self.polygon = FakePolygon(x, y)


class FakeMap:
    def __init__(self, lanes: list[object], connectors: list[object] | None = None):
        self.lanes = lanes
        self.connectors = connectors or []
        self.proximal_calls = 0

    def get_proximal_map_objects(self, ego_point, radius, layers):
        self.proximal_calls += 1
        assert radius == 150.0
        assert esf.SemanticMapLayer.LANE in layers
        assert esf.SemanticMapLayer.LANE_CONNECTOR in layers
        return {
            esf.SemanticMapLayer.LANE: self.lanes,
            esf.SemanticMapLayer.LANE_CONNECTOR: self.connectors,
        }


def test_ego_roadblock_b_realigns_route_prefix() -> None:
    result = esf.repair_route_roadblock_ids(
        FakeMap([FakeLane("B", 0.0, 0.0)]),
        FakePoint(),
        ["A", "B", "C", "D"],
    )

    assert result.route_ids == ["B", "C", "D"]
    assert result.ego_roadblock_id == "B"
    assert result.overlap_index == 1
    assert result.bridge_found is False
    assert result.bridge_len == 0
    assert result.bfs_called is False
    assert "realign_from_overlap_idx=1" in result.reason


def test_earliest_proximal_overlap_c_realigns_even_when_not_nearest() -> None:
    result = esf.repair_route_roadblock_ids(
        FakeMap([FakeLane("X", 0.0, 0.0), FakeLaneAttrOnly("C", 5.0, 0.0)]),
        FakePoint(),
        ["A", "B", "C", "D"],
        mode="realign",
    )

    assert result.route_ids == ["C", "D"]
    assert result.ego_roadblock_id == "X"
    assert result.overlap_index == 2
    assert result.bfs_called is False
    assert "realign_from_overlap_idx=2" in result.reason


def test_overlap_at_index_zero_leaves_route_unchanged() -> None:
    result = esf.repair_route_roadblock_ids(
        FakeMap([FakeLane("A", 0.0, 0.0)]),
        FakePoint(),
        ["A", "B", "C"],
    )

    assert result.route_ids == ["A", "B", "C"]
    assert result.overlap_index == 0
    assert result.bfs_called is False
    assert "overlap_idx=0" in result.reason


def test_closest_proximal_overlap_wins_over_earliest_route_overlap() -> None:
    result = esf.repair_route_roadblock_ids(
        FakeMap([FakeLane("A", 40.0, 0.0), FakeLane("C", 0.0, 0.0)]),
        FakePoint(),
        ["A", "B", "C", "D"],
    )

    assert result.route_ids == ["C", "D"]
    assert result.ego_roadblock_id == "C"
    assert result.overlap_index == 2
    assert result.bfs_called is False
    assert "realign_from_overlap_idx=2" in result.reason


def test_far_proximal_route_overlap_is_not_treated_as_current_route(monkeypatch) -> None:
    class FakeBFS:
        def __init__(self, ego_roadblock_id, map_api):
            assert ego_roadblock_id == "X"

        def search(self, *, target_roadblock_id, max_depth, max_time_s):
            assert target_roadblock_id == ["A", "B"]
            return ([], ["X", "Y", "A"]), True

    monkeypatch.setattr(esf, "BreadthFirstSearchRoadBlock", FakeBFS)

    result = esf.repair_route_roadblock_ids(
        FakeMap([FakeLane("X", 0.0, 0.0), FakeLane("A", 60.0, 0.0)]),
        FakePoint(),
        ["A", "B"],
    )

    assert result.route_ids == ["X", "Y", "A", "B"]
    assert result.bfs_called is True
    assert result.bridge_found is True
    assert result.reason == "bfs_bridge_found"


def test_empty_route_returns_empty_without_bfs() -> None:
    fake_map = FakeMap([FakeLane("A", 0.0, 0.0)])

    result = esf.repair_route_roadblock_ids(fake_map, FakePoint(), [])

    assert result.route_ids == []
    assert result.reason == "empty_route"
    assert result.ego_roadblock_id is None
    assert result.overlap_index is None
    assert result.bfs_called is False
    assert fake_map.proximal_calls == 0


def test_nofix_returns_deduped_route_without_bfs() -> None:
    fake_map = FakeMap([FakeLane("B", 0.0, 0.0)])

    result = esf.repair_route_roadblock_ids(fake_map, FakePoint(), ["A", "A", "B"], mode="nofix")

    assert result.route_ids == ["A", "B"]
    assert result.reason == "nofix"
    assert result.bfs_called is False
    assert fake_map.proximal_calls == 0


def test_realign_mode_with_no_overlap_does_not_call_bfs() -> None:
    result = esf.repair_route_roadblock_ids(
        FakeMap([FakeLane("X", 0.0, 0.0)]),
        FakePoint(),
        ["C", "D"],
        mode="realign",
    )

    assert result.route_ids == ["C", "D"]
    assert result.ego_roadblock_id == "X"
    assert result.overlap_index is None
    assert result.bfs_called is False
    assert "no_overlap" in result.reason


def test_bfs_success_bridges_from_nearest_ego_roadblock(monkeypatch) -> None:
    calls = []

    class FakeBFS:
        def __init__(self, ego_roadblock_id, map_api):
            calls.append(("init", ego_roadblock_id, map_api))

        def search(self, *, target_roadblock_id, max_depth, max_time_s):
            calls.append(("search", target_roadblock_id, max_depth, max_time_s))
            return ([], ["X", "Y", "C"]), True

    fake_map = FakeMap([FakeLane("X", 0.0, 0.0)])
    monkeypatch.setattr(esf, "BreadthFirstSearchRoadBlock", FakeBFS)
    monkeypatch.setenv("BFS_MAX_TIME_S", "0.25")

    result = esf.repair_route_roadblock_ids(fake_map, FakePoint(), ["C", "D"])

    assert result.route_ids == ["X", "Y", "C", "D"]
    assert result.ego_roadblock_id == "X"
    assert result.overlap_index is None
    assert result.bridge_found is True
    assert result.bridge_len == 3
    assert result.bfs_called is True
    assert result.reason == "bfs_bridge_found"
    assert calls == [
        ("init", "X", fake_map),
        ("search", ["C", "D"], 80, 0.25),
    ]


def test_bfs_default_timeout_allows_long_real_map_bridge(monkeypatch) -> None:
    calls = []

    class FakeBFS:
        def __init__(self, ego_roadblock_id, map_api):
            pass

        def search(self, *, target_roadblock_id, max_depth, max_time_s):
            calls.append(max_time_s)
            return ([], ["X", "Y", "C"]), True

    monkeypatch.delenv("BFS_MAX_TIME_S", raising=False)
    monkeypatch.setattr(esf, "BreadthFirstSearchRoadBlock", FakeBFS)

    result = esf.repair_route_roadblock_ids(
        FakeMap([FakeLane("X", 0.0, 0.0)]),
        FakePoint(),
        ["C", "D"],
    )

    assert result.reason == "bfs_bridge_found"
    assert calls == [8.0]


def test_bfs_failure_keeps_route_and_records_reason(monkeypatch) -> None:
    class FakeBFS:
        def __init__(self, ego_roadblock_id, map_api):
            pass

        def search(self, *, target_roadblock_id, max_depth, max_time_s):
            return ([], []), False

    monkeypatch.setattr(esf, "BreadthFirstSearchRoadBlock", FakeBFS)

    result = esf.repair_route_roadblock_ids(
        FakeMap([FakeLane("X", 0.0, 0.0)]),
        FakePoint(),
        ["C", "D"],
    )

    assert result.route_ids == ["C", "D"]
    assert result.bridge_found is False
    assert result.bridge_len == 0
    assert result.bfs_called is True
    assert "bfs_not_found" in result.reason


def test_extract_features_uses_repaired_route_ids_for_new_route_lanes(monkeypatch) -> None:
    route_lanes_calls = []

    class FakeScenario:
        def get_route_roadblock_ids(self):
            return ["A", "B", "C"]

    class FakeMapNoProx:
        map_name = "fake_map"

        def get_proximal_map_objects(self, *args, **kwargs):
            return {
                esf.SemanticMapLayer.LANE: [],
                esf.SemanticMapLayer.LANE_CONNECTOR: [],
            }

    def fake_route_lanes(point, map_api, *, max_route_lanes, route_roadblock_ids=None, **kwargs):
        route_lanes_calls.append(list(route_roadblock_ids or []))
        lanes = np.zeros((max_route_lanes, 20, 12), dtype=np.float32)
        avails = np.zeros((max_route_lanes, 20), dtype=np.float32)
        avails[0, 0] = 1.0
        return lanes, avails, np.zeros((max_route_lanes,), dtype=np.float32), np.zeros((max_route_lanes,), dtype=np.float32)

    def fake_repair(map_api, ego_point, route_roadblock_ids, **kwargs):
        assert list(route_roadblock_ids) == ["A", "B", "C"]
        return esf.RouteRoadblockRepairResult(
            route_ids=["B", "C"],
            reason="test_repaired",
            ego_roadblock_id="B",
            overlap_index=1,
            bridge_found=False,
            bridge_len=0,
            bfs_called=False,
        )

    monkeypatch.setenv("EXTRACT_PROFILE", "1")
    monkeypatch.setattr(esf, "get_target_frame", lambda *args: ("center", 123, None))
    monkeypatch.setattr(
        esf,
        "extract_ego_data",
        lambda *args: (
            np.zeros((10,), dtype=np.float32),
            np.zeros((esf.EGO_FUTURE_LEN, 3), dtype=np.float32),
            np.zeros((esf.MAX_NEIGHBORS, esf.NEIGHBOR_HISTORY_LEN, 11), dtype=np.float32),
            0.0,
            0.0,
            0.0,
            None,
            None,
        ),
    )
    monkeypatch.setattr(esf, "get_traffic_lights_at_timestamp", lambda *args: {})
    monkeypatch.setattr(
        esf,
        "extract_neighbor_agents",
        lambda *args: (
            np.zeros((esf.MAX_NEIGHBORS, esf.NEIGHBOR_HISTORY_LEN, 11), dtype=np.float32),
            np.zeros((esf.MAX_NEIGHBORS, esf.NEIGHBOR_FUTURE_LEN, 3), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(esf, "extract_static_objects", lambda *args: np.zeros((esf.MAX_STATIC_OBJECTS, 10), dtype=np.float32))
    monkeypatch.setattr(esf, "build_nuplan_scenario_from_db", lambda *args: FakeScenario())
    monkeypatch.setattr(esf, "get_pruned_route_roadblock_ids", lambda *args: ["A", "B", "C"])
    monkeypatch.setattr(
        esf,
        "extract_lanes",
        lambda *args, **kwargs: (
            np.zeros((esf.MAX_LANES, 20, 12), dtype=np.float32),
            np.zeros((esf.MAX_LANES, 20), dtype=np.float32),
            np.zeros((esf.MAX_LANES,), dtype=np.float32),
            np.zeros((esf.MAX_LANES,), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(esf, "extract_route_lanes", fake_route_lanes)
    monkeypatch.setattr(esf, "repair_route_roadblock_ids", fake_repair)

    features = esf.extract_features(object(), FakeMapNoProx(), "scene-token", 0, debug_log=False)

    assert route_lanes_calls[-1] == ["B", "C"]
    assert features["_profile_flags"]["bridge_reason"] == "test_repaired"
    assert features["_profile_flags"]["bridge_found"] is True
    assert features["_profile_flags"]["bfs_bridge_found"] is False
    assert features["_profile_flags"]["realign_from_overlap"] is True
    assert features["_profile_flags"]["overlap_idx"] == 1
    assert features["_profile_flags"]["route_len_old"] == 3
    assert features["_profile_flags"]["route_len_new"] == 2


def test_extract_features_profiles_helper_bfs_without_bad_route_geom(monkeypatch) -> None:
    class FakeScenario:
        def get_route_roadblock_ids(self):
            return ["A", "B", "C"]

    class FakeMapNoProx:
        map_name = "fake_map"

        def get_proximal_map_objects(self, *args, **kwargs):
            return {
                esf.SemanticMapLayer.LANE: [],
                esf.SemanticMapLayer.LANE_CONNECTOR: [],
            }

    def fake_route_lanes(point, map_api, *, max_route_lanes, route_roadblock_ids=None, **kwargs):
        lanes = np.zeros((max_route_lanes, 20, 12), dtype=np.float32)
        avails = np.zeros((max_route_lanes, 20), dtype=np.float32)
        avails[0, 0] = 1.0
        return lanes, avails, np.zeros((max_route_lanes,), dtype=np.float32), np.zeros((max_route_lanes,), dtype=np.float32)

    def fake_repair(map_api, ego_point, route_roadblock_ids, **kwargs):
        assert list(route_roadblock_ids) == ["A", "B", "C"]
        return esf.RouteRoadblockRepairResult(
            route_ids=["X", "Y", "A", "B", "C"],
            reason="bfs_bridge_found",
            ego_roadblock_id="X",
            overlap_index=None,
            bridge_found=True,
            bridge_len=2,
            bfs_called=True,
        )

    monkeypatch.setenv("EXTRACT_PROFILE", "1")
    monkeypatch.setattr(esf, "get_target_frame", lambda *args: ("center", 123, None))
    monkeypatch.setattr(
        esf,
        "extract_ego_data",
        lambda *args: (
            np.zeros((10,), dtype=np.float32),
            np.zeros((esf.EGO_FUTURE_LEN, 3), dtype=np.float32),
            np.zeros((esf.MAX_NEIGHBORS, esf.NEIGHBOR_HISTORY_LEN, 11), dtype=np.float32),
            0.0,
            0.0,
            0.0,
            None,
            None,
        ),
    )
    monkeypatch.setattr(esf, "get_traffic_lights_at_timestamp", lambda *args: {})
    monkeypatch.setattr(
        esf,
        "extract_neighbor_agents",
        lambda *args: (
            np.zeros((esf.MAX_NEIGHBORS, esf.NEIGHBOR_HISTORY_LEN, 11), dtype=np.float32),
            np.zeros((esf.MAX_NEIGHBORS, esf.NEIGHBOR_FUTURE_LEN, 3), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(esf, "extract_static_objects", lambda *args: np.zeros((esf.MAX_STATIC_OBJECTS, 10), dtype=np.float32))
    monkeypatch.setattr(esf, "build_nuplan_scenario_from_db", lambda *args: FakeScenario())
    monkeypatch.setattr(esf, "get_pruned_route_roadblock_ids", lambda *args: ["A", "B", "C"])
    monkeypatch.setattr(
        esf,
        "extract_lanes",
        lambda *args, **kwargs: (
            np.zeros((esf.MAX_LANES, 20, 12), dtype=np.float32),
            np.zeros((esf.MAX_LANES, 20), dtype=np.float32),
            np.zeros((esf.MAX_LANES,), dtype=np.float32),
            np.zeros((esf.MAX_LANES,), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(esf, "extract_route_lanes", fake_route_lanes)
    monkeypatch.setattr(esf, "repair_route_roadblock_ids", fake_repair)

    features = esf.extract_features(object(), FakeMapNoProx(), "scene-token", 0, debug_log=False)
    flags = features["_profile_flags"]

    assert flags["bad_route_geom"] is False
    assert flags["bfs_called"] is True
    assert flags["need_bridge"] is True
    assert flags["bfs_bridge_found"] is True
    assert flags["bridge_found"] is True
    assert flags["bfs_triggered_by"] == "helper_bfs_no_overlap"


def test_planner_runtime_features_use_repaired_route_ids_for_route_lanes(monkeypatch) -> None:
    from src.platform.nuplan.planners import diffusion_planner_ckpt_planner as planner_mod

    route_lanes_calls = []
    cfg = planner_mod.PaperModelConfig(
        agent_num=1,
        static_objects_num=1,
        lane_num=1,
        route_num=1,
        time_len=2,
        future_len=1,
        lane_len=2,
    )

    def fake_repair(map_api, ego_point, route_roadblock_ids, **kwargs):
        assert list(route_roadblock_ids) == ["A", "B", "C"]
        return esf.RouteRoadblockRepairResult(
            route_ids=["B", "C"],
            reason="test_runtime_repaired",
            ego_roadblock_id="B",
            overlap_index=1,
            bridge_found=False,
            bridge_len=0,
            bfs_called=False,
        )

    def fake_route_lanes(point, map_api, *, max_route_lanes, route_roadblock_ids=None, **kwargs):
        route_lanes_calls.append(list(route_roadblock_ids or []))
        lanes = np.zeros((max_route_lanes, cfg.lane_len, 12), dtype=np.float32)
        avails = np.zeros((max_route_lanes, cfg.lane_len), dtype=np.float32)
        return lanes, avails, np.zeros((max_route_lanes,), dtype=np.float32), np.zeros((max_route_lanes,), dtype=np.float32)

    monkeypatch.setattr(planner_mod, "repair_route_roadblock_ids", fake_repair, raising=False)
    monkeypatch.setattr(planner_mod, "Point2D", FakePoint)
    monkeypatch.setattr(
        planner_mod,
        "extract_lanes",
        lambda *args, **kwargs: (
            np.zeros((cfg.lane_num, cfg.lane_len, 12), dtype=np.float32),
            np.zeros((cfg.lane_num, cfg.lane_len), dtype=np.float32),
            np.zeros((cfg.lane_num,), dtype=np.float32),
            np.zeros((cfg.lane_num,), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(planner_mod, "extract_route_lanes", fake_route_lanes)
    monkeypatch.setattr(planner_mod, "extract_static_objects", lambda *args: np.zeros((cfg.static_objects_num, 10), dtype=np.float32))
    monkeypatch.setattr(
        planner_mod,
        "extract_neighbor_agents",
        lambda *args: (
            np.zeros((cfg.agent_num, cfg.time_len, 11), dtype=np.float32),
            np.zeros((cfg.agent_num, cfg.future_len, 3), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(planner_mod, "get_traffic_lights_at_timestamp", lambda *args: {})

    pose = types.SimpleNamespace(x=0.0, y=0.0, heading=0.0)
    ego_state = types.SimpleNamespace(
        rear_axle=pose,
        center=pose,
        time_us=123,
        dynamic_car_state=types.SimpleNamespace(
            rear_axle_velocity_2d=types.SimpleNamespace(x=0.0, y=0.0),
            rear_axle_acceleration_2d=types.SimpleNamespace(x=0.0, y=0.0),
            center_velocity_2d=types.SimpleNamespace(x=0.0, y=0.0),
            center_acceleration_2d=types.SimpleNamespace(x=0.0, y=0.0),
        ),
        car_footprint=types.SimpleNamespace(vehicle_parameters=types.SimpleNamespace(width=1.8, length=4.5)),
    )
    current_input = types.SimpleNamespace(
        history=types.SimpleNamespace(ego_states=[ego_state, ego_state]),
        iteration=types.SimpleNamespace(index=0, time_us=123),
    )

    planner = planner_mod.DiffusionPlannerCkpt.__new__(planner_mod.DiffusionPlannerCkpt)
    planner._map_api = object()
    planner._route_roadblock_ids = ["A", "B", "C"]
    planner._feature_conn = object()
    planner._runtime_debug = False
    planner._feature_sanity_debug = False
    planner._feature_sanity_ticks = 0
    planner._feature_sanity_k = 0

    planner._build_paper_runtime_features(current_input, cfg)

    assert route_lanes_calls[-1] == ["B", "C"]
