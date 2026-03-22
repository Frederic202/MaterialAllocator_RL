from ma_rl.baselines import solve_greedy
from ma_rl.domain import FeasibleMatch


def test_solve_greedy_selects_best_non_conflicting_matches():
    feasible_matches = [
        FeasibleMatch(
            material_id="M1",
            order_step_id="S1",
            allocatable=True,
            score=10.0,
        ),
        FeasibleMatch(
            material_id="M1",
            order_step_id="S2",
            allocatable=True,
            score=9.0,
        ),
        FeasibleMatch(
            material_id="M2",
            order_step_id="S1",
            allocatable=True,
            score=8.0,
        ),
        FeasibleMatch(
            material_id="M2",
            order_step_id="S2",
            allocatable=True,
            score=7.0,
        ),
    ]

    result = solve_greedy(feasible_matches, penalty_threshold=0.0)

    assert len(result.assignment_set.assignments) == 2
    assert result.assignment_set.total_score == 17.0

    assigned_pairs = {
        (a.material_id, a.order_step_id)
        for a in result.assignment_set.assignments
    }

    assert ("M1", "S1") in assigned_pairs
    assert ("M2", "S2") in assigned_pairs


def test_solve_greedy_respects_threshold():
    feasible_matches = [
        FeasibleMatch(
            material_id="M1",
            order_step_id="S1",
            allocatable=True,
            score=5.0,
        ),
        FeasibleMatch(
            material_id="M2",
            order_step_id="S2",
            allocatable=True,
            score=-1.0,
        ),
    ]

    result = solve_greedy(feasible_matches, penalty_threshold=0.0)

    assert len(result.assignment_set.assignments) == 1
    assert result.assignment_set.assignments[0].material_id == "M1"
    assert result.assignment_set.assignments[0].order_step_id == "S1"


def test_solve_greedy_ignores_non_allocatable_or_unscored_matches():
    feasible_matches = [
        FeasibleMatch(
            material_id="M1",
            order_step_id="S1",
            allocatable=False,
            score=10.0,
        ),
        FeasibleMatch(
            material_id="M2",
            order_step_id="S2",
            allocatable=True,
            score=None,
        ),
        FeasibleMatch(
            material_id="M3",
            order_step_id="S3",
            allocatable=True,
            score=4.0,
        ),
    ]

    result = solve_greedy(feasible_matches, penalty_threshold=0.0)

    assert len(result.assignment_set.assignments) == 1
    assert result.assignment_set.assignments[0].material_id == "M3"
    assert result.assignment_set.assignments[0].order_step_id == "S3"