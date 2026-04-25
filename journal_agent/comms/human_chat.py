from enum import StrEnum


class Speaker(StrEnum):
    AI = "ai"
    SYSTEM = "system"


_ANSI = {
    Speaker.AI:     "\033[36m",   # cyan
    Speaker.SYSTEM: "\033[33m",   # yellow
}
_RESET = "\033[0m"
_PREFIX = {
    Speaker.AI:     "AI",
    Speaker.SYSTEM: "System",
}


def get_human_input() -> str:
    while True:
        print("You (blank line to send):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        text = "\n".join(lines).strip()
        if text:
            return text


def talk_to_human(message: str, speaker: Speaker = Speaker.SYSTEM) -> None:
    color = _ANSI[speaker]
    prefix = _PREFIX[speaker]
    print(f"{color}{prefix}: {message}{_RESET}")
