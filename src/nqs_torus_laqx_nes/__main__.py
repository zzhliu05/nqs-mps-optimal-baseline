"""Module entry point: python -m nqs_torus_laqx_nes."""

from __future__ import annotations

import jax

from .cli import parse_config
from .workflow import run


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    args, config = parse_config()
    print("Arguments:", vars(args))
    run(config)


if __name__ == "__main__":
    main()

