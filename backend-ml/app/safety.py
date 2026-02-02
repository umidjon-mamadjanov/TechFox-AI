BANNED_KEYWORDS = [
    "hack", "exploit", "illegal", "crime",
    "kill", "weapon", "drugs", "suicide", "porn"
]

def is_dangerous(text: str) -> bool:
    text = text.lower()
    return any(word in text for word in BANNED_KEYWORDS)
