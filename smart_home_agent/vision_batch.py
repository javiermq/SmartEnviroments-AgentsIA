"""Decisiones de lote independientes de DeepFace y del transporte."""
from statistics import median
import math


def choose_identity(candidates: list[dict], margin: float) -> tuple[str, str]:
    if any(not math.isfinite(item[key]) for item in candidates for key in ("distance", "threshold")):
        return "unknow", "invalid_scores"
    ordered = sorted(candidates, key=lambda item: item["distance"])
    if not ordered:
        return "unknow", "empty_gallery"
    best = ordered[0]
    if best["distance"] > best["threshold"]:
        return "unknow", "no_match"
    if len(ordered) > 1 and ordered[1]["distance"] - best["distance"] < margin:
        return "unknow", "ambiguous"
    return best["user"], "matched"



def aggregate_batch(frames: list[dict], margin: float = 0.05, emotion_margin: float = 10.0) -> dict:
    if len(frames) != 3:
        raise ValueError("Se requieren exactamente tres resultados")

    candidates = []
    names = sorted({item["user"] for frame in frames for item in frame.get("candidates", [])})
    for name in names:
        rows = [item for frame in frames for item in frame.get("candidates", []) if item["user"] == name]
        valid = [row for row in rows if all(math.isfinite(row[key]) for key in ("distance", "threshold"))]
        if len(valid) < 2:
            continue
        candidates.append({"user": name, "distance": float(median(row["distance"] for row in valid)),
                           "threshold": min(row["threshold"] for row in valid),
                           "valid_frames": len(valid),
                           "votes": sum(frame.get("user") == name and frame.get("status") == "matched" for frame in frames)})
    user, status = choose_identity(candidates, margin)
    if user != "unknow" and next(item for item in candidates if item["user"] == user)["votes"] < 2:
        user, status = "unknow", "insufficient_agreement"
    if not candidates:
        status = "insufficient_valid_frames"
    # No mezclar la expresión de una captura que votó por otra identidad.
    expression_frames = [frame for frame in frames if user == "unknow" or frame.get("user") == user]
    scores = [frame.get("emotion_scores", {}) for frame in expression_frames]
    scores = [row for row in scores if row and all(math.isfinite(v) and 0 <= v <= 100.001 for v in row.values())]
    combined = {}
    emotion = "uncertain"
    if len(scores) >= 2:
        labels = set.intersection(*(set(row) for row in scores))
        combined = {label: float(median(row[label] for row in scores)) for label in sorted(labels)}
        total = sum(combined.values())
        if total > 0:
            combined = {label: value * 100 / total for label, value in combined.items()}
            ranked = sorted(combined, key=combined.get, reverse=True)
            if len(ranked) >= 2 and combined[ranked[0]] - combined[ranked[1]] >= emotion_margin:
                emotion = ranked[0]
    return {"user": user, "status": status, "candidates": candidates,
            "emotion": emotion, "emotion_scores": combined, "emotion_valid_frames": len(scores),
            "aggregation": "median_and_2_of_3", "frames": frames}
