"""Utility to run all experiments."""

from __future__ import annotations

from examples.run_nasa_pnn_lstm import main as run_nasa_pnn_lstm


def run_all_experiments() -> None:
    """Run all configured experiments."""
    run_nasa_pnn_lstm()


if __name__ == "__main__":
    run_all_experiments()
