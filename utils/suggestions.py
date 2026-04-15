"""
Rule-based supportive suggestions mapped to predicted emotion labels.
These are non-clinical, general wellness prompts suitable for a demo API.
"""
from utils.config import LABELS

# Primary suggestion per label (extend or localize as needed)
SUGGESTIONS: dict[str, str] = {
    "Stress": (
        "Your words suggest elevated stress. Consider short breaks, breathing exercises, "
        "and prioritizing one task at a time. If stress persists, talking to someone you trust can help."
    ),
    "Anxiety": (
        "Your text hints at anxiety. Grounding techniques (5-4-3-2-1), limiting caffeine, "
        "and gentle movement may help. Consider reaching out to a counselor if worry feels overwhelming."
    ),
    "Depression": (
        "What you shared may reflect low mood. Small steps matter—try sunlight, a brief walk, "
        "or connecting with one supportive person. If you have thoughts of self-harm, seek immediate help "
        "from local emergency services or a crisis line."
    ),
    "Neutral": (
        "Your message sounds fairly neutral. Keep checking in with yourself—brief self-reflection "
        "or journaling can help you notice patterns early."
    ),
}

# Optional secondary tips (could be rotated or A/B tested)
EXTRA_TIPS: dict[str, list[str]] = {
    "Stress": [
        "Try the Pomodoro technique to chunk work.",
        "Schedule a non-negotiable 10-minute unwind before sleep.",
    ],
    "Anxiety": [
        "Label worries as 'thoughts' rather than facts.",
        "Limit news scrolling before bed.",
    ],
    "Depression": [
        "Celebrate micro-wins (e.g., brushing teeth, one message to a friend).",
        "Maintain a simple routine even on hard days.",
    ],
    "Neutral": [
        "Gratitude notes—even one line—can boost perspective.",
        "Stay hydrated and move a little each hour.",
    ],
}


def suggestion_for_label(label: str) -> str:
    """Return the main suggestion string for a predicted label."""
    if label not in LABELS:
        return SUGGESTIONS["Neutral"]
    return SUGGESTIONS.get(label, SUGGESTIONS["Neutral"])


def extra_tips_for_label(label: str, max_tips: int = 2) -> list[str]:
    """Return up to max_tips additional bullet ideas."""
    tips = EXTRA_TIPS.get(label, EXTRA_TIPS["Neutral"])
    return tips[:max_tips]
