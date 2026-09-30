"""Run the default JAX-Noah simulation and write the standard txt output."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FORCING = PROJECT_DIR / "wudaoliang-forcing_cmfd.txt"
DEFAULT_OUTPUT = PROJECT_DIR / "Noah_output_jax.txt"


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the default Noah JAX result file in txt format."
    )
    parser.add_argument(
        "--forcing",
        default=str(DEFAULT_FORCING),
        help=f"Forcing file path. Default: {DEFAULT_FORCING}",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output txt path. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    forcing_path = resolve_project_path(args.forcing)
    output_path = resolve_project_path(args.output)

    if not forcing_path.exists():
        raise FileNotFoundError(f"Forcing file not found: {forcing_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # noah_main reads parameter tables through project-relative paths.
    os.chdir(PROJECT_DIR)

    from Noah_jax_simple import noah_main

    print(f"Running Noah JAX with forcing: {forcing_path}")
    print(f"Writing txt output to: {output_path}")
    noah_main(str(forcing_path), output_flag=True, output_filename=str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
