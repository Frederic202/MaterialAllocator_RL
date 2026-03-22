from __future__ import annotations

from dataclasses import dataclass, field

from ma_rl.domain.models import Assignment, AssignmentSet, FeasibleMatch


@dataclass(slots=True)
class SolveGreedyResult:
    assignment_set: AssignmentSet
    selected_matches: list[FeasibleMatch] = field(default_factory=list)
    skipped_matches: list[FeasibleMatch] = field(default_factory=list)
    threshold: float = 0.0
    candidate_count: int = 0
    filtered_count: int = 0


def solve_greedy(
    feasible_matches: list[FeasibleMatch],
    penalty_threshold: float | None = None,
) -> SolveGreedyResult:
    """
    1. nur allocatable matches berücksichtigen
    2. score darf nicht None sein
    3. score muss >= penalty_threshold sein
    4. nach score absteigend sortieren
    5. ein Match nur übernehmen, wenn Material und OrderStep noch nicht vergeben sind
    """

    candidates = [
        match
        for match in feasible_matches
        if match.allocatable
        and match.score is not None
        and (
            penalty_threshold is None
            or match.score >= penalty_threshold
        )
    ]

    candidates.sort(key=lambda match: match.score, reverse=True)

    used_material_ids: set[str] = set()
    used_order_step_ids: set[str] = set()

    selected_matches: list[FeasibleMatch] = []
    skipped_matches: list[FeasibleMatch] = []

    assignment_set = AssignmentSet()

    for match in candidates:
        if match.material_id in used_material_ids:
            skipped_matches.append(match)
            continue

        if match.order_step_id in used_order_step_ids:
            skipped_matches.append(match)
            continue

        selected_matches.append(match)
        used_material_ids.add(match.material_id)
        used_order_step_ids.add(match.order_step_id)

        assignment = Assignment(
            material_id=match.material_id,
            order_step_id=match.order_step_id,
            score=float(match.score),
            source_rule_set_name=match.rule_set_name,
        )
        assignment_set.add_assignment(assignment)

    assignment_set.total_match_score = sum(a.score for a in assignment_set.assignments)
    assignment_set.total_penalty = 0.0
    assignment_set.total_bonus = 0.0
    assignment_set.total_score = (
        assignment_set.total_match_score
        - assignment_set.total_penalty
        + assignment_set.total_bonus
    )

    return SolveGreedyResult(
        assignment_set=assignment_set,
        selected_matches=selected_matches,
        skipped_matches=skipped_matches,
        threshold=penalty_threshold,
        candidate_count=len(feasible_matches),
        filtered_count=len(candidates),
    )