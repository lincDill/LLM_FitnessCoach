import math

from main import calculate_recovery_score, recommend_workout


def test_recovery_score_is_in_range():
    history = [
        {"type": "run", "duration": 45, "intensity": 7, "sets": 1},
        {"type": "lift", "duration": 60, "intensity": 8, "sets": 4},
        {"type": "bike", "duration": 40, "intensity": 6, "sets": 1},
        {"type": "swim", "duration": 35, "intensity": 7, "sets": 1},
    ]
    score = calculate_recovery_score(history, {"goal": "maintain", "calories": {"P": 150, "C": 220, "F": 60, "Other": 300}}, 170)
    assert 0 <= score <= 100
    assert math.isfinite(score)


def test_recommendation_uses_recent_training_bias():
    history = [
        {"type": "run", "duration": 60, "intensity": 8, "sets": 1},
        {"type": "run", "duration": 55, "intensity": 8, "sets": 1},
        {"type": "run", "duration": 52, "intensity": 8, "sets": 1},
        {"type": "run", "duration": 50, "intensity": 7, "sets": 1},
    ]
    rec = recommend_workout(76, history, "maintain")
    assert "bike" in rec.lower() or "swim" in rec.lower() or "lift" in rec.lower() or "run" in rec.lower()
