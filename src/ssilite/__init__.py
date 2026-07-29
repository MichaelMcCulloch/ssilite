"""Support acquisition and compute-aware robust-learning prototype."""

__all__ = ["main"]


def main() -> None:
    from .experiment import main as experiment_main

    experiment_main()
