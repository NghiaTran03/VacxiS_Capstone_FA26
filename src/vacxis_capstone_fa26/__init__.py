"""VacxiS Capstone: Vaccine Information RAG System."""

__all__ = ["main"]


def main() -> None:
    """Load and run the generation CLI when invoked."""
    from vacxis_capstone_fa26.generation.cli import main as generation_main

    generation_main()
