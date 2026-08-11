"""
download.py

Downloads and extracts the PlantVillage dataset from Kaggle.

Author: Manahil Ishfaq
Project: Plant Disease Detection
"""

from __future__ import annotations

import logging
import random
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np

# =============================================================================
# Configuration
# =============================================================================

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"

ZIP_NAME = "plantvillage.zip"

DATASET = "mohitsingh1804/plantvillage"

# =============================================================================
# Logging
# =============================================================================
 
logger = logging.getLogger(__name__)

# =============================================================================
# Reproducibility
# =============================================================================

random.seed(SEED)
np.random.seed(SEED)


def create_directories() -> None:
    """Create required directories."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset() -> Path:
    """
    Download dataset only if zip file does not already exist.

    Returns
    -------
    Path
        Path to dataset zip file.
    """

    zip_path = RAW_DIR / ZIP_NAME

    # ---------------------------------------------------------
    # Skip download if zip already exists
    # ---------------------------------------------------------
    if zip_path.exists():

        logger.info("Dataset zip already exists.")
        logger.info(f"Using existing file: {zip_path}")

        return zip_path

    # ---------------------------------------------------------
    # Otherwise download from Kaggle
    # ---------------------------------------------------------
    logger.info("Downloading PlantVillage dataset from Kaggle...")

    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET,
            "-p",
            str(RAW_DIR),
        ],
        check=True,
    )

    if not zip_path.exists():

        raise FileNotFoundError(zip_path)

    logger.info("Download completed.")

    return zip_path


def extract_dataset(zip_path: Path) -> None:
    """
    Extract dataset.

    Parameters
    ----------
    zip_path : Path
        Dataset zip path.
    """

    logger.info("Extracting dataset...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:

        zip_ref.extractall(RAW_DIR)

    logger.info("Extraction completed.")


def remove_zip(zip_path: Path) -> None:
    """
    Delete zip file.

    Parameters
    ----------
    zip_path : Path
    """

    if zip_path.exists():

        zip_path.unlink()

        logger.info("Zip file removed.")


def clean_mac_files() -> None:
    """
    Remove macOS metadata folders if present.
    """

    mac_folder = RAW_DIR / "__MACOSX"

    if mac_folder.exists():

        shutil.rmtree(mac_folder)

        logger.info("Removed __MACOSX folder.")


def main() -> None:

    logger.info("========== Dataset Download ==========")

    create_directories()

    zip_path = download_dataset()

    extract_dataset(zip_path)

    clean_mac_files()

    remove_zip(zip_path)

    logger.info("Dataset ready inside data/raw/")


if __name__ == "__main__":

    main()