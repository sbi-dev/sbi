from .two_moons import TwoMoons


def get_task(name: str):
    if name == "two_moons":
        return TwoMoons()
    else:
        raise ValueError(f"Unknown task {name}")
