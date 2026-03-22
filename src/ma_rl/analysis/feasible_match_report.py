from __future__ import annotations

from collections import Counter, defaultdict

from ma_rl.domain import FeasibleMatch, Scenario


def summarize_input_types(scenario: Scenario) -> dict[str, Counter]:
    material_type_counts = Counter(material.mat_type_code for material in scenario.materials)
    order_step_type_counts = Counter(step.prod_step_type_code for step in scenario.order_steps)

    return {
        "material_type_counts": material_type_counts,
        "order_step_type_counts": order_step_type_counts,
    }


def summarize_feasible_matches_by_type_pair(
    feasible_matches: list[FeasibleMatch],
    scenario: Scenario,
    penalty_threshold: float = 0.0,
) -> list[dict]:
    material_by_id = {material.material_id: material for material in scenario.materials}
    order_step_by_id = {step.order_step_id: step for step in scenario.order_steps}

    grouped: dict[tuple[str, str], dict] = defaultdict(
        lambda: {
            "total_matches": 0,
            "allocatable_matches": 0,
            "non_allocatable_matches": 0,
            "scored_matches": 0,
            "threshold_matches": 0,
            "scores": [],
            "failed_rule_names": Counter(),
        }
    )

    for match in feasible_matches:
        material = material_by_id.get(match.material_id)
        order_step = order_step_by_id.get(match.order_step_id)

        if material is None or order_step is None:
            continue

        key = (material.mat_type_code, order_step.prod_step_type_code)
        stats = grouped[key]

        stats["total_matches"] += 1

        if match.allocatable:
            stats["allocatable_matches"] += 1
        else:
            stats["non_allocatable_matches"] += 1

        if match.score is not None:
            stats["scored_matches"] += 1
            stats["scores"].append(match.score)

            if match.score >= penalty_threshold:
                stats["threshold_matches"] += 1

        for failed_rule_name in match.failed_rule_names:
            stats["failed_rule_names"][failed_rule_name] += 1

    rows: list[dict] = []
    for (mat_type_code, prod_step_type_code), stats in sorted(grouped.items()):
        scores = stats["scores"]
        avg_score = sum(scores) / len(scores) if scores else None
        min_score = min(scores) if scores else None
        max_score = max(scores) if scores else None

        rows.append(
            {
                "mat_type_code": mat_type_code,
                "prod_step_type_code": prod_step_type_code,
                "total_matches": stats["total_matches"],
                "allocatable_matches": stats["allocatable_matches"],
                "non_allocatable_matches": stats["non_allocatable_matches"],
                "scored_matches": stats["scored_matches"],
                "threshold_matches": stats["threshold_matches"],
                "avg_score": avg_score,
                "min_score": min_score,
                "max_score": max_score,
                "failed_rule_names": dict(stats["failed_rule_names"]),
            }
        )

    return rows


def print_input_type_summary(scenario: Scenario) -> None:
    summary = summarize_input_types(scenario)

    print("\nMaterial types in scenario:")
    for mat_type_code, count in sorted(summary["material_type_counts"].items()):
        print(f"  {mat_type_code}: {count}")

    print("\nOrder step types in scenario:")
    for step_type_code, count in sorted(summary["order_step_type_counts"].items()):
        print(f"  {step_type_code}: {count}")


def print_feasible_match_type_pair_summary(
    feasible_matches: list[FeasibleMatch],
    scenario: Scenario,
    penalty_threshold: float = 0.0,
) -> None:
    rows = summarize_feasible_matches_by_type_pair(
        feasible_matches=feasible_matches,
        scenario=scenario,
        penalty_threshold=penalty_threshold,
    )

    print("\nFeasible matches by type pair:")
    for row in rows:
        avg_score = "None" if row["avg_score"] is None else f"{row['avg_score']:.4f}"
        min_score = "None" if row["min_score"] is None else f"{row['min_score']:.4f}"
        max_score = "None" if row["max_score"] is None else f"{row['max_score']:.4f}"

        print(
            f"  ({row['mat_type_code']} -> {row['prod_step_type_code']}): "
            f"total={row['total_matches']}, "
            f"allocatable={row['allocatable_matches']}, "
            f"non_allocatable={row['non_allocatable_matches']}, "
            f"scored={row['scored_matches']}, "
            f"above_threshold={row['threshold_matches']}, "
            f"avg={avg_score}, min={min_score}, max={max_score}"
        )

        if row["failed_rule_names"]:
            print(f"    failed_rules={row['failed_rule_names']}")