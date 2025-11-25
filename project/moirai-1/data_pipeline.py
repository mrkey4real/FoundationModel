"""Utilities for preparing MOIRAI training data with the all-in-target policy.

This module keeps every dynamic variable inside ``target`` so that masking and
variate embeddings cover the full signal space. ``feat_dynamic_real`` is kept
empty by construction, and missing values are preserved as ``NaN`` so the
downstream dataloaders can mask them automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
import yaml


def load_super_schema(schema_path: str | Path) -> dict:
    """Load the super schema without reordering existing IDs.

    The schema is stored as a YAML mapping with string IDs. Existing IDs are
    respected verbatim and new variables should always be appended via
    :func:`append_variables` instead of inserting into the middle.
    """

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    if not isinstance(schema, MutableMapping) or "variables" not in schema:
        raise ValueError("Schema must contain a top-level 'variables' mapping")

    # Preserve author-specified ordering while ensuring the IDs are treated as
    # opaque strings instead of sorted numerically.
    variables = schema["variables"]
    schema["variables"] = {str(k): v for k, v in variables.items()}
    return schema


def append_variables(schema: dict, new_variables: Mapping[str, dict]) -> dict:
    """Append new variables to the super schema.

    Existing IDs stay untouched and new IDs are allocated by walking forward
    from the current maximum. This enforces the "only append, never reorder"
    policy required by the super schema.
    """

    existing_ids: list[int] = [int(k) for k in schema["variables"].keys()]
    next_id = max(existing_ids) + 1 if existing_ids else 0

    for _, payload in new_variables.items():
        schema["variables"][str(next_id)] = payload
        next_id += 1

    return schema


def _lookup_variable_id(schema: dict, variable_name: str) -> str:
    for vid, vinfo in schema["variables"].items():
        if vinfo.get("name") == variable_name:
            return str(vid)
    raise KeyError(f"Variable name '{variable_name}' not found in schema")


def _apply_physical_range(values: np.ndarray, variable_info: Mapping) -> np.ndarray:
    cleaned = values.astype(np.float32, copy=True)
    if "physical_range" in variable_info:
        low, high = variable_info["physical_range"]
        if low is not None:
            cleaned[cleaned < low] = np.nan
        if high is not None:
            cleaned[cleaned > high] = np.nan
    return cleaned


def dataframe_to_target(
    df: pd.DataFrame,
    schema: dict,
    column_mapping: Mapping[str, str],
    *,
    start: pd.Timestamp,
    freq: str,
    item_id: str | int,
    feat_static_cat: Sequence[int] | None = None,
) -> dict:
    """Convert a dataframe into a MOIRAI sample using the all-in-target policy.

    Parameters
    ----------
    df:
        Time-indexed dataframe containing the raw signals.
    schema:
        Super schema dictionary loaded via :func:`load_super_schema`.
    column_mapping:
        Mapping from dataframe column name to schema ``variables[].name``.
    start / freq / item_id / feat_static_cat:
        Metadata stored on the resulting sample.
    """

    max_id = max(int(k) for k in schema["variables"].keys()) if schema["variables"] else -1
    target = np.full((max_id + 1, len(df)), np.nan, dtype=np.float32)

    for column, schema_name in column_mapping.items():
        if column not in df.columns:
            continue

        variable_id = _lookup_variable_id(schema, schema_name)
        variable_info = schema["variables"][variable_id]

        values = _apply_physical_range(df[column].to_numpy(dtype=np.float32), variable_info)
        target[int(variable_id), :] = values

    sample: dict[str, object] = {
        "item_id": str(item_id),
        "start": pd.Timestamp(start),
        "freq": freq,
        "target": target,
        # Explicitly empty to honor the all-in-target rule.
        "feat_dynamic_real": np.empty((0, len(df)), dtype=np.float32),
    }

    if feat_static_cat is not None:
        sample["feat_static_cat"] = np.asarray(feat_static_cat, dtype=np.int64)

    return sample


def build_samples(
    frames: Iterable[pd.DataFrame],
    schema: dict,
    column_mapping: Mapping[str, str],
    *,
    start: pd.Timestamp,
    freq: str,
    item_ids: Iterable[str | int],
    feat_static_cat: Sequence[int] | None = None,
) -> list[dict]:
    """Batch helper around :func:`dataframe_to_target` for multiple frames."""

    samples = []
    for item_id, frame in zip(item_ids, frames):
        samples.append(
            dataframe_to_target(
                frame,
                schema,
                column_mapping,
                start=start,
                freq=freq,
                item_id=item_id,
                feat_static_cat=feat_static_cat,
            )
        )
    return samples
