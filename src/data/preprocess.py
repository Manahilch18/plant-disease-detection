"""
src/data/preprocess.py
======================
Single source of truth for all image preprocessing in the Plant Disease
Detection System.

Every downstream component — training scripts, FastAPI, Streamlit, Grad-CAM,
notebooks — imports from this module.  Nothing else should contain image
preprocessing logic.

Supported models
----------------
* custom_cnn   → pixel / 255.0
* efficientnet → tf.keras.applications.efficientnet.preprocess_input
* resnet50     → tf.keras.applications.resnet.preprocess_input

Author: Plant Disease Detection Project
Python: 3.11+
Style : PEP 8, Google-style docstrings
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

 
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_NUM_CLASSES: int = 38
ModelType = Literal["custom_cnn", "efficientnet", "resnet50"]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PreprocessingConfig:
    """All hyperparameters and paths for the preprocessing pipeline.

    Attributes:
        dataset_root: Root directory that contains train/, val/, test/.
        train_dir: Path to the training split directory.
        val_dir: Path to the validation split directory.
        test_dir: Path to the test split directory.
        image_size: Target spatial resolution as (height, width).
        batch_size: Number of samples per training batch.
        seed: Global random seed for reproducibility.
        shuffle_buffer: Size of the shuffle buffer for tf.data.
        cache_dataset: Whether to cache the dataset in memory after first epoch.
        prefetch: Number of batches to prefetch (``tf.data.AUTOTUNE`` = -1).
        model_type: Selects the normalisation strategy.
        artifact_directory: Directory where JSON artefacts are persisted.
    """

    dataset_root: Path = Path("resplit_dataset")
    train_dir: Path = field(init=False)
    val_dir: Path = field(init=False)
    test_dir: Path = field(init=False)
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 32
    seed: int = 42
    shuffle_buffer: int = 1000
    cache_dataset: bool = True
    prefetch: int = tf.data.AUTOTUNE
    model_type: ModelType = "custom_cnn"
    artifact_directory: Path = Path("artifacts/preprocessing")

    def __post_init__(self) -> None:
        self.dataset_root = Path(self.dataset_root)
        self.train_dir = self.dataset_root / "train"
        self.val_dir = self.dataset_root / "val"
        self.test_dir = self.dataset_root / "test"
        self.artifact_directory = Path(self.artifact_directory)
        self.artifact_directory.mkdir(parents=True, exist_ok=True)
        logger.info("PreprocessingConfig initialised: model_type=%s, image_size=%s, batch_size=%d",
                    self.model_type, self.image_size, self.batch_size)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_random_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and TensorFlow.

    Args:
        seed: Integer seed value used across all RNG sources.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info("Global random seed set to %d.", seed)


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------


def validate_dataset_structure(config: PreprocessingConfig) -> None:
    """Verify that the dataset on disk matches project expectations.

    Checks that train, val, and test directories exist and that each split
    contains exactly :data:`EXPECTED_NUM_CLASSES` class sub-directories.

    Args:
        config: Preprocessing configuration holding split directory paths.

    Raises:
        FileNotFoundError: If any split directory does not exist.
        ValueError: If the number of classes in any split differs from
            :data:`EXPECTED_NUM_CLASSES`.
    """
    for split_name, split_dir in [
        ("train", config.train_dir),
        ("val", config.val_dir),
        ("test", config.test_dir),
    ]:
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory '{split_name}' not found at: {split_dir}. "
                "Ensure the dataset has been downloaded and split correctly."
            )
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        num_classes = len(class_dirs)
        if num_classes != EXPECTED_NUM_CLASSES:
            raise ValueError(
                f"Expected {EXPECTED_NUM_CLASSES} classes in '{split_name}' split "
                f"at {split_dir}, but found {num_classes}. "
                "Verify dataset integrity."
            )
        logger.info("Split '%s' validated: %d classes found at %s.", split_name, num_classes, split_dir)

    logger.info("Dataset structure validation passed.")


# ---------------------------------------------------------------------------
# Class mapping
# ---------------------------------------------------------------------------


def build_class_mapping(
    train_dir: Path,
) -> tuple[dict[str, int], dict[int, str]]:
    """Build a deterministic class-to-index mapping from the training directory.

    Class directories are sorted lexicographically so that the mapping is
    identical across all runs regardless of file-system ordering.

    Args:
        train_dir: Path to the training split directory.

    Returns:
        A 2-tuple of:
            - class_to_index: ``{class_name: integer_label}``
            - index_to_class: ``{integer_label: class_name}``

    Raises:
        FileNotFoundError: If *train_dir* does not exist.
    """
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    class_names: list[str] = sorted(
        [d.name for d in train_dir.iterdir() if d.is_dir()]
    )
    class_to_index: dict[str, int] = {name: idx for idx, name in enumerate(class_names)}
    index_to_class: dict[int, str] = {idx: name for name, idx in class_to_index.items()}

    logger.info("Class mapping built: %d classes.", len(class_to_index))
    return class_to_index, index_to_class


def save_class_mapping(
    class_to_index: dict[str, int],
    index_to_class: dict[int, str],
    artifact_directory: Path,
) -> Path:
    """Persist the class mapping to disk as JSON.

    Args:
        class_to_index: Mapping from class name to integer label.
        index_to_class: Mapping from integer label to class name.
        artifact_directory: Directory in which to write ``class_mapping.json``.

    Returns:
        Path to the written JSON file.
    """
    artifact_directory.mkdir(parents=True, exist_ok=True)
    output_path = artifact_directory / "class_mapping.json"
    payload = {
        "class_to_index": class_to_index,
        "index_to_class": {str(k): v for k, v in index_to_class.items()},
    }
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    logger.info("Class mapping saved to %s.", output_path)
    return output_path


def load_class_mapping(
    artifact_directory: Path,
) -> tuple[dict[str, int], dict[int, str]]:
    """Load a previously saved class mapping from disk.

    Args:
        artifact_directory: Directory containing ``class_mapping.json``.

    Returns:
        A 2-tuple of:
            - class_to_index: ``{class_name: integer_label}``
            - index_to_class: ``{integer_label: class_name}``

    Raises:
        FileNotFoundError: If ``class_mapping.json`` does not exist.
    """
    mapping_path = artifact_directory / "class_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"class_mapping.json not found at {mapping_path}. "
            "Run the preprocessing pipeline first."
        )
    with mapping_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    class_to_index: dict[str, int] = payload["class_to_index"]
    index_to_class: dict[int, str] = {int(k): v for k, v in payload["index_to_class"].items()}
    logger.info("Class mapping loaded from %s (%d classes).", mapping_path, len(class_to_index))
    return class_to_index, index_to_class


# ---------------------------------------------------------------------------
# Image-path collection
# ---------------------------------------------------------------------------


def collect_image_paths(
    split_dir: Path,
    class_to_index: dict[str, int],
) -> tuple[list[str], list[int], list[str]]:
    """Collect all image file paths and their corresponding integer labels.

    Supports JPEG and PNG images (case-insensitive extension matching).

    Args:
        split_dir: Root directory of the dataset split (train/val/test).
        class_to_index: Mapping from class name to integer label.

    Returns:
        A 3-tuple of:
            - paths: Absolute image file paths as strings.
            - labels: Integer class labels aligned with *paths*.
            - class_names: Sorted list of class names (mirrors *class_to_index*).

    Raises:
        FileNotFoundError: If *split_dir* does not exist.
        ValueError: If no images are found.
    """
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    valid_extensions: set[str] = {".jpg", ".jpeg", ".png"}
    paths: list[str] = []
    labels: list[int] = []

    for class_name, label in sorted(class_to_index.items()):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            logger.warning("Class directory missing: %s — skipping.", class_dir)
            continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in valid_extensions:
                paths.append(str(img_path))
                labels.append(label)

    if not paths:
        raise ValueError(
            f"No images found in {split_dir}. "
            "Check dataset structure and file extensions."
        )

    class_names: list[str] = sorted(class_to_index.keys())
    logger.info(
        "Collected %d images from '%s' across %d classes.",
        len(paths), split_dir.name, len(class_names),
    )
    return paths, labels, class_names


# ---------------------------------------------------------------------------
# Low-level image ops (TensorFlow graph-compatible)
# ---------------------------------------------------------------------------


def decode_image(raw_bytes: tf.Tensor) -> tf.Tensor:
    """Decode a raw JPEG/PNG byte string into a float32 RGB tensor.

    Args:
        raw_bytes: 1-D ``tf.string`` tensor containing the raw image bytes.

    Returns:
        3-D ``tf.float32`` tensor with shape ``[H, W, 3]`` and values in
        ``[0.0, 255.0]``.
    """
    image = tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    return image


def resize_image(
    image: tf.Tensor,
    target_size: tuple[int, int] = (224, 224),
) -> tf.Tensor:
    """Resize an image tensor to the target spatial resolution.

    Uses bilinear interpolation.  The image is NOT saved to disk.

    Args:
        image: 3-D ``tf.float32`` tensor with shape ``[H, W, C]``.
        target_size: ``(height, width)`` in pixels.

    Returns:
        3-D ``tf.float32`` tensor with shape ``[target_size[0], target_size[1], C]``.
    """
    return tf.image.resize(image, size=target_size, method="bilinear")


# ---------------------------------------------------------------------------
# Normalisation functions
# ---------------------------------------------------------------------------


def normalize_cnn(image: tf.Tensor) -> tf.Tensor:
    """Scale pixel values to [0, 1] for a custom CNN.

    Args:
        image: ``tf.float32`` tensor with pixel values in ``[0.0, 255.0]``.

    Returns:
        ``tf.float32`` tensor with values in ``[0.0, 1.0]``.
    """
    return image / 255.0


def normalize_efficientnet(image: tf.Tensor) -> tf.Tensor:
    """Apply EfficientNet-specific preprocessing.

    Applies ``tf.keras.applications.efficientnet.preprocess_input``, which
    scales pixels to ``[-1, 1]``.

    Args:
        image: ``tf.float32`` tensor with pixel values in ``[0.0, 255.0]``.

    Returns:
        Preprocessed ``tf.float32`` tensor.
    """
    return tf.keras.applications.efficientnet.preprocess_input(image)


def normalize_resnet(image: tf.Tensor) -> tf.Tensor:
    """Apply ResNet50-specific preprocessing.

    Applies ``tf.keras.applications.resnet.preprocess_input``, which performs
    mean-subtraction in BGR channel order.

    Args:
        image: ``tf.float32`` tensor with pixel values in ``[0.0, 255.0]``.

    Returns:
        Preprocessed ``tf.float32`` tensor.
    """
    return tf.keras.applications.resnet.preprocess_input(image)


def get_preprocessing_function(
    model_type: ModelType,
) -> Callable[[tf.Tensor], tf.Tensor]:
    """Return the normalisation function corresponding to *model_type*.

    Args:
        model_type: One of ``"custom_cnn"``, ``"efficientnet"``, ``"resnet50"``.

    Returns:
        A callable that accepts and returns a ``tf.Tensor``.

    Raises:
        ValueError: If *model_type* is not one of the supported values.
    """
    registry: dict[str, Callable[[tf.Tensor], tf.Tensor]] = {
        "custom_cnn": normalize_cnn,
        "efficientnet": normalize_efficientnet,
        "resnet50": normalize_resnet,
    }
    if model_type not in registry:
        raise ValueError(
            f"Unsupported model_type='{model_type}'. "
            f"Choose from: {sorted(registry.keys())}."
        )
    logger.info("Preprocessing function selected for model_type='%s'.", model_type)
    return registry[model_type]


# ---------------------------------------------------------------------------
# Composite image loader
# ---------------------------------------------------------------------------


def load_and_preprocess_image(
    path: tf.Tensor,
    label: tf.Tensor,
    target_size: tuple[int, int],
    preprocess_fn: Callable[[tf.Tensor], tf.Tensor],
) -> tuple[tf.Tensor, tf.Tensor]:
    """Read, decode, resize, and normalise a single image.

    Designed to be used inside a ``tf.data`` pipeline via
    ``dataset.map(fn, num_parallel_calls=tf.data.AUTOTUNE)``.

    Pipeline:
        file path → raw bytes → float32 tensor → resize → normalise

    Args:
        path: Scalar ``tf.string`` tensor containing the image file path.
        label: Scalar integer label tensor.
        target_size: ``(height, width)`` to resize to.
        preprocess_fn: Normalisation callable returned by
            :func:`get_preprocessing_function`.

    Returns:
        A 2-tuple of ``(preprocessed_image, label)`` where the image has
        shape ``[H, W, 3]``.
    """
    raw = tf.io.read_file(path)
    image = decode_image(raw)
    image = resize_image(image, target_size)
    image = preprocess_fn(image)
    return image, label


# ---------------------------------------------------------------------------
# tf.data pipeline
# ---------------------------------------------------------------------------


def build_tf_dataset(
    paths: list[str],
    labels: list[int],
) -> tf.data.Dataset:
    """Create a ``tf.data.Dataset`` from lists of file paths and integer labels.

    Args:
        paths: Image file paths.
        labels: Integer class labels aligned with *paths*.

    Returns:
        An un-batched ``tf.data.Dataset`` of ``(path_tensor, label_tensor)``
        pairs.
    """
    path_tensor = tf.constant(paths, dtype=tf.string)
    label_tensor = tf.constant(labels, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((path_tensor, label_tensor))
    logger.info("Built tf.data.Dataset with %d samples.", len(paths))
    return dataset


def optimize_dataset(
    dataset: tf.data.Dataset,
    config: PreprocessingConfig,
    is_training: bool,
    preprocess_fn: Callable[[tf.Tensor], tf.Tensor],
) -> tf.data.Dataset:
    """Apply caching, shuffling, mapping, batching, and prefetching.

    Args:
        dataset: Raw ``tf.data.Dataset`` of ``(path, label)`` pairs.
        config: Preprocessing configuration.
        is_training: When ``True`` the dataset is shuffled after caching.
        preprocess_fn: Image normalisation function.

    Returns:
        An optimised, batched ``tf.data.Dataset``.
    """
    # Partial-apply fixed arguments to make the map callable signature-compatible.
    target_size = config.image_size
    _load = lambda path, label: load_and_preprocess_image(
        path, label, target_size, preprocess_fn
    )

    # Cache raw file paths before decoding to avoid repeated I/O on small datasets.
    if config.cache_dataset:
        dataset = dataset.cache()
        logger.debug("Dataset caching enabled.")

    if is_training:
        dataset = dataset.shuffle(
            buffer_size=config.shuffle_buffer,
            seed=config.seed,
            reshuffle_each_iteration=True,
        )
        logger.debug("Dataset shuffling enabled (buffer=%d).", config.shuffle_buffer)

    dataset = dataset.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(config.batch_size, drop_remainder=False)
    dataset = dataset.prefetch(config.prefetch)

    logger.info(
        "Dataset optimised: training=%s, batch_size=%d, cache=%s.",
        is_training, config.batch_size, config.cache_dataset,
    )
    return dataset


# ---------------------------------------------------------------------------
# High-level dataset factory
# ---------------------------------------------------------------------------


def create_datasets(
    config: PreprocessingConfig,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Build train, validation, and test ``tf.data.Dataset`` objects.

    This is the primary entry point for all downstream consumers.

    Args:
        config: Preprocessing configuration.

    Returns:
        A 3-tuple of ``(train_ds, val_ds, test_ds)``.
    """
    logger.info("Creating datasets for model_type='%s'.", config.model_type)

    set_random_seed(config.seed)
    validate_dataset_structure(config)

    class_to_index, index_to_class = build_class_mapping(config.train_dir)
    save_class_mapping(class_to_index, index_to_class, config.artifact_directory)

    preprocess_fn = get_preprocessing_function(config.model_type)

    splits: dict[str, tuple[tf.data.Dataset, bool]] = {}
    split_sizes: dict[str, int] = {}

    for split_name, split_dir, is_training in [
        ("train", config.train_dir, True),
        ("val", config.val_dir, False),
        ("test", config.test_dir, False),
    ]:
        paths, labels, class_names = collect_image_paths(split_dir, class_to_index)
        split_sizes[split_name] = len(paths)

        raw_ds = build_tf_dataset(paths, labels)
        optimised_ds = optimize_dataset(raw_ds, config, is_training, preprocess_fn)
        splits[split_name] = (optimised_ds, labels)

    train_ds = splits["train"][0]
    val_ds = splits["val"][0]
    test_ds = splits["test"][0]

    # Compute and persist class weights using training labels.
    train_labels = splits["train"][1]
    class_weights = compute_class_weights(train_labels, list(class_to_index.values()))
    save_class_weights(class_weights, config.artifact_directory)

    save_dataset_statistics(
        num_classes=len(class_to_index),
        split_sizes=split_sizes,
        config=config,
    )
    save_preprocessing_config(config)

    logger.info(
        "Datasets ready — train: %d, val: %d, test: %d samples.",
        split_sizes["train"], split_sizes["val"], split_sizes["test"],
    )
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------


def compute_class_weights(
    labels: list[int],
    classes: list[int],
) -> dict[int, float]:
    """Compute per-class weights to compensate for label imbalance.

    Uses ``sklearn.utils.class_weight.compute_class_weight`` with the
    ``"balanced"`` strategy.

    Args:
        labels: Full list of integer labels from the training split.
        classes: Sorted list of unique integer class indices.

    Returns:
        Mapping from integer class index to float weight.
    """
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(classes),
        y=np.array(labels),
    )
    class_weight_dict: dict[int, float] = {
        int(cls): float(w) for cls, w in zip(classes, weights)
    }
    logger.info(
        "Class weights computed. Min=%.4f, Max=%.4f.",
        min(class_weight_dict.values()),
        max(class_weight_dict.values()),
    )
    return class_weight_dict


def save_class_weights(
    class_weights: dict[int, float],
    artifact_directory: Path,
) -> Path:
    """Persist class weights to disk as JSON.

    Args:
        class_weights: Mapping from integer class index to float weight.
        artifact_directory: Directory in which to write ``class_weights.json``.

    Returns:
        Path to the written JSON file.
    """
    artifact_directory.mkdir(parents=True, exist_ok=True)
    output_path = artifact_directory / "class_weights.json"
    serialisable = {str(k): v for k, v in class_weights.items()}
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2, sort_keys=True)
    logger.info("Class weights saved to %s.", output_path)
    return output_path


def load_class_weights(artifact_directory: Path) -> dict[int, float]:
    """Load previously saved class weights from disk.

    Args:
        artifact_directory: Directory containing ``class_weights.json``.

    Returns:
        Mapping from integer class index to float weight.

    Raises:
        FileNotFoundError: If ``class_weights.json`` does not exist.
    """
    weights_path = artifact_directory / "class_weights.json"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"class_weights.json not found at {weights_path}. "
            "Run the preprocessing pipeline first."
        )
    with weights_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    class_weights: dict[int, float] = {int(k): float(v) for k, v in raw.items()}
    logger.info("Class weights loaded from %s.", weights_path)
    return class_weights


# ---------------------------------------------------------------------------
# Artefact persistence
# ---------------------------------------------------------------------------


def save_dataset_statistics(
    num_classes: int,
    split_sizes: dict[str, int],
    config: PreprocessingConfig,
) -> Path:
    """Save high-level dataset statistics to disk.

    Args:
        num_classes: Number of unique classes.
        split_sizes: Mapping from split name to sample count.
        config: Preprocessing configuration.

    Returns:
        Path to the written JSON file.
    """
    total_images = sum(split_sizes.values())
    stats = {
        "num_classes": num_classes,
        "total_images": total_images,
        "split_sizes": split_sizes,
        "batch_size": config.batch_size,
        "image_size": list(config.image_size),
        "seed": config.seed,
        "model_type": config.model_type,
    }
    output_path = config.artifact_directory / "dataset_statistics.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("Dataset statistics saved to %s.", output_path)
    return output_path


def save_preprocessing_config(config: PreprocessingConfig) -> Path:
    """Serialise the full :class:`PreprocessingConfig` to disk.

    Args:
        config: Preprocessing configuration to persist.

    Returns:
        Path to the written JSON file.
    """
    output_path = config.artifact_directory / "preprocessing_config.json"
    payload = asdict(config)
    # Convert Path objects to strings for JSON serialisation.
    payload = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in payload.items()
    }
    # tuple → list for JSON.
    payload["image_size"] = list(config.image_size)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Preprocessing config saved to %s.", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Dataset verification
# ---------------------------------------------------------------------------


def verify_dataset(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    config: PreprocessingConfig,
    num_classes: int,
) -> None:
    """Run sanity checks on the built datasets.

    Inspects the first batch of each split to confirm:

    * Batch shapes are consistent with ``config.image_size`` and
      ``config.batch_size``.
    * Labels are non-negative integers within ``[0, num_classes)``.
    * Image pixel values fall within the expected normalised range for the
      selected ``model_type``.

    Args:
        train_ds: Training ``tf.data.Dataset``.
        val_ds: Validation ``tf.data.Dataset``.
        test_ds: Test ``tf.data.Dataset``.
        config: Preprocessing configuration.
        num_classes: Total number of unique classes.

    Raises:
        ValueError: If any verification check fails.
    """
    h, w = config.image_size

    for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        images, labels = next(iter(ds))

        # Shape check.
        expected_image_shape = (None, h, w, 3)
        actual_shape = images.shape
        if actual_shape[1] != h or actual_shape[2] != w or actual_shape[3] != 3:
            raise ValueError(
                f"[{split_name}] Unexpected image shape {actual_shape}. "
                f"Expected (batch, {h}, {w}, 3)."
            )

        # Label range check.
        min_label = int(tf.reduce_min(labels).numpy())
        max_label = int(tf.reduce_max(labels).numpy())
        if min_label < 0 or max_label >= num_classes:
            raise ValueError(
                f"[{split_name}] Labels out of range [{min_label}, {max_label}]. "
                f"Expected [0, {num_classes - 1}]."
            )

        # Pixel value range check (model-type specific).
        img_min = float(tf.reduce_min(images).numpy())
        img_max = float(tf.reduce_max(images).numpy())

        if config.model_type == "custom_cnn":
            if not (0.0 <= img_min and img_max <= 1.0):
                raise ValueError(
                    f"[{split_name}] custom_cnn: pixel values out of [0, 1]: "
                    f"min={img_min:.4f}, max={img_max:.4f}."
                )
        elif config.model_type in {"efficientnet", "resnet50"}:
            if not (-200.0 <= img_min and img_max <= 200.0):
                raise ValueError(
                    f"[{split_name}] {config.model_type}: unexpected pixel range: "
                    f"min={img_min:.4f}, max={img_max:.4f}."
                )

        logger.info(
            "[verify] split='%s' | shape=%s | labels=[%d, %d] | pixels=[%.4f, %.4f]",
            split_name, tuple(actual_shape), min_label, max_label, img_min, img_max,
        )

    logger.info("All dataset verification checks passed.")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def print_dataset_summary(
    config: PreprocessingConfig,
    class_to_index: dict[str, int],
    split_sizes: dict[str, int],
    class_weights: dict[int, float],
) -> None:
    """Print a professionally formatted preprocessing summary to stdout.

    Args:
        config: Preprocessing configuration.
        class_to_index: Class-name to integer-label mapping.
        split_sizes: Sample counts per split.
        class_weights: Per-class weight values.
    """
    total = sum(split_sizes.values())
    divider = "=" * 60
    thin_divider = "-" * 60

    print(f"\n{divider}")
    print("  PLANT DISEASE DETECTION — PREPROCESSING SUMMARY")
    print(divider)

    print(f"\n  {'Model type':<25}: {config.model_type}")
    print(f"  {'Image size':<25}: {config.image_size[0]} × {config.image_size[1]} px")
    print(f"  {'Batch size':<25}: {config.batch_size}")
    print(f"  {'Random seed':<25}: {config.seed}")
    print(f"  {'Cache dataset':<25}: {config.cache_dataset}")
    print(f"  {'Shuffle buffer':<25}: {config.shuffle_buffer}")
    print(f"  {'Prefetch':<25}: {'AUTOTUNE' if config.prefetch == tf.data.AUTOTUNE else config.prefetch}")
    print(f"  {'Artifact directory':<25}: {config.artifact_directory}")

    print(f"\n{thin_divider}")
    print("  DATASET SPLITS")
    print(thin_divider)
    for split_name, count in split_sizes.items():
        pct = 100 * count / total if total else 0.0
        print(f"  {split_name:<10} : {count:>6,} samples  ({pct:>5.1f}%)")
    print(f"  {'TOTAL':<10} : {total:>6,} samples")

    print(f"\n{thin_divider}")
    print(f"  CLASSES  (total: {len(class_to_index)})")
    print(thin_divider)
    for class_name, idx in sorted(class_to_index.items(), key=lambda kv: kv[1]):
        weight = class_weights.get(idx, float("nan"))
        print(f"  [{idx:>2}] {class_name:<45} weight={weight:.4f}")

    print(f"\n{thin_divider}")
    print("  CLASS WEIGHT STATISTICS")
    print(thin_divider)
    weights = list(class_weights.values())
    print(f"  Min weight  : {min(weights):.4f}")
    print(f"  Max weight  : {max(weights):.4f}")
    print(f"  Mean weight : {sum(weights) / len(weights):.4f}")

    print(f"\n{thin_divider}")
    print("  ARTIFACT FILES")
    print(thin_divider)
    for fname in [
        "class_mapping.json",
        "class_weights.json",
        "dataset_statistics.json",
        "preprocessing_config.json",
    ]:
        fpath = config.artifact_directory / fname
        status = "✓  exists" if fpath.exists() else "✗  missing"
        print(f"  {status}  {fpath}")

    print(f"\n{divider}\n")


# ---------------------------------------------------------------------------
# Module-level convenience entry point
# ---------------------------------------------------------------------------


def run_preprocessing_pipeline(
    config: Optional[PreprocessingConfig] = None,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, dict[str, int], dict[int, float]]:
    """Execute the full preprocessing pipeline end-to-end.

    This function is the single call a training script or notebook needs to
    make to obtain production-ready datasets plus supporting artefacts.

    Args:
        config: Optional :class:`PreprocessingConfig`.  A default config is
            used when ``None`` is passed.

    Returns:
        A 5-tuple of:
            - train_ds: Training ``tf.data.Dataset``.
            - val_ds: Validation ``tf.data.Dataset``.
            - test_ds: Test ``tf.data.Dataset``.
            - class_to_index: Class-name to integer-label mapping.
            - class_weights: Per-class weight dict for use with ``model.fit``.
    """
    if config is None:
        config = PreprocessingConfig()

    logger.info("Starting preprocessing pipeline.")

    train_ds, val_ds, test_ds = create_datasets(config)

    class_to_index, index_to_class = load_class_mapping(config.artifact_directory)
    class_weights = load_class_weights(config.artifact_directory)

    num_classes = len(class_to_index)

    verify_dataset(train_ds, val_ds, test_ds, config, num_classes)

    # Collect split sizes for the summary.
    stats_path = config.artifact_directory / "dataset_statistics.json"
    with stats_path.open("r", encoding="utf-8") as fh:
        stats = json.load(fh)
    split_sizes: dict[str, int] = stats["split_sizes"]

    print_dataset_summary(config, class_to_index, split_sizes, class_weights)

    logger.info("Preprocessing pipeline complete.")
    return train_ds, val_ds, test_ds, class_to_index, class_weights
