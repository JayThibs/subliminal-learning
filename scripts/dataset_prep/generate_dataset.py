#!/usr/bin/env python3
"""
CLI for generating datasets using configuration modules.

Usage:
    python scripts/generate_dataset.py cfgs/animal_number_preferences/dataset_cfg.py control_cfg
"""

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path
from loguru import logger
from sl.datasets.services import Cfg
from sl.datasets.services import generate_dataset


def load_config_from_module(module_path: str, cfg_var_name: str) -> Cfg:
    """Load a configuration instance from a Python module."""
    spec = importlib.util.spec_from_file_location("config_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Look for the specified cfg variable in the module
    if not hasattr(module, cfg_var_name):
        raise AttributeError(
            f"Module {module_path} must contain a '{cfg_var_name}' variable"
        )

    cfg = getattr(module, cfg_var_name)
    assert isinstance(cfg, Cfg)
    return cfg


async def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset using a configuration module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/generate_dataset.py cfgs/animal_number_preferences/dataset_cfg.py control_cfg
        """,
    )

    parser.add_argument(
        "config_module", help="Path to Python module containing dataset configuration"
    )

    parser.add_argument(
        "cfg_var_name",
        nargs="?",
        help="Name of the configuration variable in the module",
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config file {args.config_module} does not exist")
        sys.exit(1)

    try:
        # Load configuration from module
        logger.info(
            f"Loading configuration from {args.config_module} (variable: {args.cfg_var_name})..."
        )
        cfg = load_config_from_module(args.config_module, args.cfg_var_name)

        # Import and run dataset generation
        logger.info("Starting dataset generation...")
        await generate_dataset(cfg)
        logger.success("Dataset generation completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
