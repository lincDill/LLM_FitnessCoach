from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import ollama
except Exception:  # pragma: no cover - fallback when dependency is unavailable
    ollama = None

OLLAMA_MODEL = "llama3.2:3b"

WORKOUT_TYPES = ["run", "swim", "lift", "bike"]
WORKOUT_TYPE_BONUS = {
    "run": 1.1,
    "swim": 1.0,
    "bike": 0.9,
    "lift": 1.2,
}


def normalize_workouts(workouts: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    normalized: list[dict[str, float | str]] = []
    for workout in workouts:
        workout_type = str(workout.get("type", "")).strip().lower()
        if workout_type not in WORKOUT_TYPES:
            continue
        normalized.append(
            {
                "type": workout_type,
                "duration": float(workout.get("duration", 0) or 0),
                "intensity": float(workout.get("intensity", 0) or 0),
                "sets": float(workout.get("sets", 1) or 1),
            }
        )
    return normalized


def workout_load(workout: dict[str, float | str]) -> float:
    duration = float(workout.get("duration", 0) or 0)
    intensity = float(workout.get("intensity", 0) or 0)
    sets = float(workout.get("sets", 1) or 1)
    return duration * intensity * sets * WORKOUT_TYPE_BONUS.get(str(workout.get("type", "")).lower(), 1.0)


def calculate_recovery_score(history: list[dict[str, Any]], nutrition: dict[str, Any], current_weight: float) -> float:
    recent = normalize_workouts(history[-4:])
    if not recent:
        return 80.0

    recent_loads = [workout_load(w) for w in recent]
    average_load = sum(recent_loads) / len(recent_loads)
    avg_intensity = sum(float(w["intensity"]) for w in recent) / len(recent)
    duration_pressure = sum(float(w["duration"]) for w in recent) / len(recent)

    total_calories = (
        float(nutrition.get("P", 0) or 0) * 4
        + float(nutrition.get("C", 0) or 0) * 4
        + float(nutrition.get("F", 0) or 0) * 9
        + float(nutrition.get("Other", 0) or 0)
    )

    protein_g = float(nutrition.get("P", 0) or 0)
    carb_g = float(nutrition.get("C", 0) or 0)
    fat_g = float(nutrition.get("F", 0) or 0)

    nutrition_score = 100.0
    if total_calories > 0:
        if protein_g < max(100.0, current_weight * 1.5):
            nutrition_score -= 10
        if carb_g < 150:
            nutrition_score -= 8
        if fat_g < 45:
            nutrition_score -= 6

    trend_penalty = 0.0
    if len(recent) > 1:
        deltas = [recent_loads[i] - recent_loads[i - 1] for i in range(1, len(recent_loads))]
        if sum(deltas) > 0:
            trend_penalty += 8
        if any(load < previous for load, previous in zip(recent_loads[1:], recent_loads[:-1])):
            trend_penalty += 6

    fatigue = (avg_intensity * 5.5) + (duration_pressure / 8) + (average_load / 18) + trend_penalty
    recovery = 100 - fatigue + nutrition_score * 0.35
    recovery = max(0.0, min(100.0, recovery))
    return round(recovery, 1)


def infer_training_balance(history: list[dict[str, Any]]) -> str:
    recent = normalize_workouts(history[-4:])
    if not recent:
        return "balanced"
    counts = Counter(str(workout["type"]) for workout in recent)
    most_common = counts.most_common(1)[0][0]
    return most_common


def recommend_workout(recovery_score: float, history: list[dict[str, Any]], goal: str) -> str:
    recent = normalize_workouts(history[-4:])
    dominant_type = infer_training_balance(history)
    goal = str(goal).lower()

    if recovery_score >= 80:
        intensity_label = "high-intensity"
        if dominant_type == "run":
            suggested_type = "lift" if goal in {"gain", "maintain"} else "bike"
        elif dominant_type == "lift":
            suggested_type = "run" if goal == "maintain" else "bike"
        elif dominant_type == "bike":
            suggested_type = "lift" if goal == "gain" else "run"
        else:
            suggested_type = "run"
        detail = "40-60 min session with hard intervals and a brief cooldown."
    elif recovery_score >= 60:
        intensity_label = "moderate"
        if dominant_type == "run":
            suggested_type = "swim" if goal == "maintain" else "bike"
        elif dominant_type == "lift":
            suggested_type = "bike" if goal == "maintain" else "run"
        elif dominant_type == "bike":
            suggested_type = "lift" if goal == "gain" else "swim"
        else:
            suggested_type = "lift"
        detail = "30-45 min steady effort with controlled volume."
    elif recovery_score >= 40:
        intensity_label = "low"
        suggested_type = "bike" if dominant_type == "run" else "swim"
        detail = "20-35 min easy aerobic work with mobility and light technique work."
    else:
        intensity_label = "deload"
        suggested_type = "bike"
        detail = "20-30 min easy spin or recovery walk with minimal strain."

    if recent and len(recent) >= 3:
        last_three = [w["type"] for w in recent[-3:]]
        if last_three.count(dominant_type) >= 2 and recovery_score < 55:
            suggested_type = "bike"
            intensity_label = "deload"
            detail = "Recovery-focused session to reduce accumulated fatigue before the next hard block."

    if goal == "lose":
        if suggested_type in {"lift", "run"}:
            suggested_type = "bike" if recovery_score < 70 else "run"

    return f"Recommended workout: {intensity_label} {suggested_type} session — {detail}"


def call_llm(summary_prompt: str, model: str = OLLAMA_MODEL) -> str:
    if ollama is None:
        return "Ollama is not available in this environment. Using rule-based fitness analysis instead."

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": summary_prompt}],
        )
        return str(response["message"]["content"]).strip()
    except Exception as e:
        return f"Ollama was not reachable. Using rule-based fitness analysis instead. Error: ({e.__class__.__name__}"


def analyze_trends(history: list[dict[str, Any]], recovery_score: float, goal: str, current_weight: float) -> tuple[str, str]:
    recent = normalize_workouts(history[-4:])
    if not recent:
        return (
            "No workout history available yet. Begin logging workouts to establish your trend baseline.",
            "Add a structured training history and aim for balanced recovery before increasing intensity.",
        )

    loads = [workout_load(w) for w in recent]
    trend_message = "Training load is stable and recovery is manageable."
    if recovery_score < 50:
        trend_message = "Recovery is trending down and fatigue appears elevated relative to recent workload."
    elif recovery_score >= 80:
        trend_message = "Recovery is strong and recent training load is supporting performance gains."
    elif any(load < previous for load, previous in zip(loads[1:], loads[:-1])):
        trend_message = "Performance has started to drift downward and the recent volume trend is not fully supported by recovery."

    improvement = "Keep protein and carbs higher on training days to support performance and recovery."
    if goal == "lose":
        improvement = "Maintain a consistent calorie deficit without cutting carbs too aggressively on high-load days."
    elif goal == "gain":
        improvement = "Add a little more total volume to the next quality session and protect sleep quality."

    if recovery_score < 45:
        improvement = "Use a low-intensity training day and reduce total set count before the next hard session."

    summary = (
        f"Current trends: {trend_message} Recent workload over the last four workouts is {sum(loads):.0f} "
        f"load points, and the current recovery estimate is {recovery_score:.1f}/100 for a {goal} goal at "
        f"{current_weight:.1f} lb."
    )
    return summary, improvement


def build_daily_report(date_value: str, data: dict[str, Any]) -> dict[str, Any]:
    history = data.get("workouts", [])
    goal = str(data.get("goal", "maintain")).lower()
    weight = float(data.get("current_weight", 0) or 0)
    nutrition = data.get("calories", {})
    recovery_score = calculate_recovery_score(history, nutrition, weight)

    summary, improvement = analyze_trends(history, recovery_score, goal, weight)
    llm_summary = call_llm(
        f"You are a fitness coach. Based on the following data, provide a concise summary current trends and one action item. "
        f"Goal: {goal}. Recovery score: {recovery_score}. Weight: {weight}. Prior workouts: {history}. "
        f"Calories: {nutrition}. Do not mention missing data if not necessary."
    )
    llm_improvement = call_llm(
        f"You are a fitness coach. Give one specific thing the user should do better next to improve training quality. "
        f"Goal: {goal}. Recovery score: {recovery_score}. Recent workouts: {history}."
    )
    summary_text = llm_summary if llm_summary else summary
    improvement_text = llm_improvement if llm_improvement else improvement
    recommendation = recommend_workout(recovery_score, history, goal)

    return {
        "date": date_value,
        "goal": goal,
        "current_weight": weight,
        "calories_P": float(nutrition.get("P", 0) or 0),
        "calories_C": float(nutrition.get("C", 0) or 0),
        "calories_F": float(nutrition.get("F", 0) or 0),
        "calories_Other": float(nutrition.get("Other", 0) or 0),
        "summary_current_trends": summary_text,
        "one_thing_to_do_better": improvement_text,
        "recovery_score": recovery_score,
        "recommended_workout": recommendation,
    }


def append_daily_csv(output_path: str | Path, row: dict[str, Any]) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date",
        "goal",
        "current_weight",
        "calories_P",
        "calories_C",
        "calories_F",
        "calories_Other",
        "summary_current_trends",
        "one_thing_to_do_better",
        "recovery_score",
        "recommended_workout",
    ]

    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})

    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI fitness coach for daily workout recommendations.")
    parser.add_argument("--input", default="daily_input.json", help="Path to the JSON input file.")
    parser.add_argument("--output", default="fitness_output.csv", help="CSV path for the dated output.")
    parser.add_argument("--demo", action="store_true", help="Use a built-in example dataset instead of a JSON file.")
    return parser.parse_args()


def load_demo_payload() -> dict[str, Any]:
    return {
        "goal": "maintain",
        "current_weight": 170,
        "calories": {"P": 180, "C": 220, "F": 60, "Other": 350},
        "workouts": [
            {"type": "run", "duration": 45, "intensity": 7, "sets": 1},
            {"type": "lift", "duration": 60, "intensity": 8, "sets": 4},
            {"type": "bike", "duration": 40, "intensity": 6, "sets": 1},
            {"type": "run", "duration": 50, "intensity": 8, "sets": 1},
        ],
    }


def main() -> None:
    args = parse_args()
    if args.demo:
        payload = load_demo_payload()
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        payload = json.loads(input_path.read_text(encoding="utf-8"))

    date_value = datetime.utcnow().strftime("%Y-%m-%d")
    row = build_daily_report(date_value, payload)
    out = append_daily_csv(args.output, row)
    print(f"Saved daily recommendations to {out}")
    print(f"Recovery score: {row['recovery_score']}")
    print(f"Summary: {row['summary_current_trends']}")
    print(f"One thing to do better: {row['one_thing_to_do_better']}")
    print(f"Recommended workout: {row['recommended_workout']}")


if __name__ == "__main__":
    main()
