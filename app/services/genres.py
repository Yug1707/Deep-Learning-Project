"""Genre label loading for AST predictions and feedback."""

from __future__ import annotations

import ast
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd

from app.schemas import GenreInfo


def _read_genre_names(metadata_dir: Path) -> Dict[str, str]:
    genres_csv = metadata_dir / "genres.csv"
    if not genres_csv.exists():
        return {}

    names: Dict[str, str] = {}
    with open(genres_csv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            genre_id = str(row.get("genre_id", "")).strip()
            title = str(row.get("title", "")).strip()
            if genre_id:
                names[genre_id] = title or f"Genre {genre_id}"
    return names


def _derive_index_to_genre(ast_config: Dict) -> Dict[str, str]:
    """Derive the same top-k genre mapping used by AST build_dataset."""
    dataset_cfg = ast_config["dataset"]
    metadata_csv = Path(ast_config["paths"]["metadata_csv"])
    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"Missing AST class mapping and metadata CSV: {metadata_csv}"
        )

    tracks = pd.read_csv(metadata_csv, index_col=0, header=[0, 1])
    subset = tracks[tracks[("set", "subset")] == dataset_cfg["subset"]]
    genres_raw = subset[("track", "genres_all")].dropna()
    genres_parsed = genres_raw.apply(ast.literal_eval)

    counter: Counter[int] = Counter()
    for genre_list in genres_parsed:
        counter.update(int(item) for item in genre_list)

    top_k = int(dataset_cfg["top_k_genres"])
    top_genres = [str(genre_id) for genre_id, _ in counter.most_common(top_k)]
    return {str(index): genre_id for index, genre_id in enumerate(top_genres)}


def load_genres(ast_config: Dict) -> List[GenreInfo]:
    """Load genre labels from AST class_mapping.json or derive them from metadata."""
    manifests_dir = Path(ast_config["paths"]["manifests_dir"])
    mapping_path = manifests_dir / "class_mapping.json"

    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        index_to_genre = payload.get("index_to_genre", {})
    else:
        index_to_genre = _derive_index_to_genre(ast_config)

    metadata_dir = Path(ast_config["paths"]["metadata_csv"]).parent
    names = _read_genre_names(metadata_dir)

    genres: List[GenreInfo] = []
    for raw_index, genre_id in sorted(index_to_genre.items(), key=lambda item: int(item[0])):
        genre_id_str = str(genre_id)
        genres.append(
            GenreInfo(
                class_index=int(raw_index),
                genre_id=genre_id_str,
                name=names.get(genre_id_str, f"Genre {genre_id_str}"),
            )
        )

    return genres


def genre_lookup(genres: List[GenreInfo]) -> Dict[str, GenreInfo]:
    """Return genre_id -> GenreInfo lookup."""
    return {genre.genre_id: genre for genre in genres}


def index_lookup(genres: List[GenreInfo]) -> Dict[int, GenreInfo]:
    """Return class_index -> GenreInfo lookup."""
    return {genre.class_index: genre for genre in genres}

