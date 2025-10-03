#!/usr/bin/env python3

"""
===============================================================================
JSON Loader Utilities
===============================================================================

Purpose
-------
This script provides utility functions for loading JSON-based configuration
files used to parameterize runs of other scripts. It uses argparse for
command-line integration and performs sanity checks on required keys.

Functions
---------
- load_json(path): Reads a JSON config file, converts "start"/"end" ISO strings
  to datetime objects, and returns the parsed dict.
- load_config():  Parses CLI arguments (-p/--params), loads config via
  load_json(), and returns both config and argparse Namespace.

Dependencies
------------
- argparse: for command-line parsing
- sys: for exiting gracefully with error codes
- json: for parsing config files
- datetime: for ISO string → datetime conversion
- logging: for structured log messages

Authors & Contributors
----------------------
Diego F. Sanchez (@kd2rlm)          
- 10/02/2025

===============================================================================
"""

import argparse, sys, json
from datetime import datetime
import logging

# Setup module-level logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def load_json(path: str) -> dict:
    """
    Load and parse a JSON configuration file.

    Parameters
    ----------
    path : str
        Path to the JSON file containing configuration parameters.
        Required keys: "start", "end" (ISO datetime strings).

    Returns
    -------
    cfg : dict
        Parsed configuration with:
          - "sDate" : datetime object converted from "start"
          - "eDate" : datetime object converted from "end"
          - all other keys as loaded from JSON
    """
    with open(path) as f:
        cfg = json.load(f)
    cfg["sDate"] = datetime.fromisoformat(cfg.pop("start"))
    cfg["eDate"] = datetime.fromisoformat(cfg.pop("end"))
    log.info(f"Loading JSON file {path}...")
    return cfg


def load_config() -> tuple[dict, argparse.Namespace]:
    """
    Parse command-line arguments and load the JSON config.

    Command-line Arguments
    ----------------------
    -p, --params : str
        Path to the JSON configuration file.

    Returns
    -------
    cfg : dict
        Parsed configuration dictionary (see load_json()).
    args : argparse.Namespace
        Namespace object returned by argparse, containing parsed arguments.
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--params", required=True, help="Path to config JSON")
    try:
        args = ap.parse_args()
        cfg  = load_json(args.params)
        return cfg, args
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {args.params}"); sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {args.params}: {e}"); sys.exit(1)
    except KeyError as e:
        print(f"ERROR: Missing required key in {args.params}: {e}"); sys.exit(1)
    except SystemExit:
        print("ERROR: You must pass -p/--params pointing to a JSON config file."); sys.exit(2)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}"); sys.exit(1)
