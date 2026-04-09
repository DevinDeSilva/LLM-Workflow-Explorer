from __future__ import annotations

import csv
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class SyntheticQuestionCategory(str, Enum):
    OBJECT_LEVEL_FROM_OBJECT = "object-level|from-object"
    OBJECT_LEVEL_FROM_PROP = "object-level|from-prop"
    PATH_LEVEL = "path-level"

    @property
    def description(self) -> str:
        descriptions = {
            self.OBJECT_LEVEL_FROM_OBJECT: (
                "Questions about a single ontology object and the facts, attributes, "
                "or directly connected entities that can be retrieved from that object."
            ),
            self.OBJECT_LEVEL_FROM_PROP: (
                "Questions centered on a relation or property and the ontology objects "
                "that should be found by filtering or traversing through that property."
            ),
            self.PATH_LEVEL: (
                "Questions about chains, routes, or multi-hop connections between "
                "classes or entities in the workflow graph."
            ),
        }
        return descriptions[self]

    @classmethod
    def values(cls) -> tuple[str, ...]:
        return tuple(category.value for category in cls)


class SQRetriver:
    DEFAULT_LOCATION = "evaluations/calibration/ques_creation/SyntheticQuestionKG.csv"
    CATEGORY_ALIASES = {
        SyntheticQuestionCategory.PATH_LEVEL.value: (
            SyntheticQuestionCategory.PATH_LEVEL.value
        ),
        "path_level": SyntheticQuestionCategory.PATH_LEVEL.value,
        "path": SyntheticQuestionCategory.PATH_LEVEL.value,
        SyntheticQuestionCategory.OBJECT_LEVEL_FROM_OBJECT.value: (
            SyntheticQuestionCategory.OBJECT_LEVEL_FROM_OBJECT.value
        ),
        "object_level_from_object": (
            SyntheticQuestionCategory.OBJECT_LEVEL_FROM_OBJECT.value
        ),
        "from_object": SyntheticQuestionCategory.OBJECT_LEVEL_FROM_OBJECT.value,
        SyntheticQuestionCategory.OBJECT_LEVEL_FROM_PROP.value: (
            SyntheticQuestionCategory.OBJECT_LEVEL_FROM_PROP.value
        ),
        "object_level_from_prop": (
            SyntheticQuestionCategory.OBJECT_LEVEL_FROM_PROP.value
        ),
        "from_prop": SyntheticQuestionCategory.OBJECT_LEVEL_FROM_PROP.value,
    }

    def __init__(self, location: Optional[str | Path] = None) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.file_path = self._resolve_location(location or self.DEFAULT_LOCATION)
        self.rows = self._load_rows(self.file_path)
        self.rows_by_program_id = {
            row["program_id"]: row
            for row in self.rows
            if row.get("program_id")
        }
        self.columns = tuple(self.rows[0].keys()) if self.rows else tuple()
        category_values = [
            category
            for row in self.rows
            for category in [row.get("category")]
            if category is not None
        ]
        self.categories = tuple(
            sorted(set(category_values))
        )
        self.category_specific_filters = self._infer_category_specific_filters()

    def retrieve(
        self,
        category: str,
        **filters: Any,
    ) -> list[dict[str, Optional[str]]]:
        normalized_category = self._normalize_category(category)
        return self._filter_category(normalized_category, **filters)

    def retrieve_path_level(
        self,
        start_node: Optional[str] = None,
        end_node: Optional[str] = None,
        **filters: Any,
    ) -> list[dict[str, Optional[str]]]:
        return self._filter_category(
            "path-level",
            start_node=start_node,
            end_node=end_node,
            **filters,
        )

    def retrieve_object_level_from_object(
        self,
        focal_node: Optional[str] = None,
        focal_relation: Optional[str] = None,
        **filters: Any,
    ) -> list[dict[str, Optional[str]]]:
        return self._filter_category(
            "object-level|from-object",
            focal_node=focal_node,
            focal_relation=focal_relation,
            **filters,
        )

    def retrieve_object_level_from_prop(
        self,
        focal_node: Optional[str] = None,
        focal_relation: Optional[str] = None,
        **filters: Any,
    ) -> list[dict[str, Optional[str]]]:
        return self._filter_category(
            "object-level|from-prop",
            focal_node=focal_node,
            focal_relation=focal_relation,
            **filters,
        )

    def get_category_specific_filters(self, category: str) -> tuple[str, ...]:
        normalized_category = self._normalize_category(category)
        return self.category_specific_filters.get(normalized_category, tuple())

    def get_available_categories(self) -> tuple[str, ...]:
        known_categories = [
            category
            for category in SyntheticQuestionCategory.values()
            if category in self.categories
        ]
        if known_categories:
            return tuple(known_categories)
        return self.categories

    def get_category_description(self, category: str) -> str:
        normalized_category = self._normalize_category(category)
        try:
            return SyntheticQuestionCategory(normalized_category).description
        except ValueError:
            return normalized_category

    def normalize_category(self, category: str) -> str:
        return self._normalize_category(category)

    def get_program_row(
        self,
        program_id: str,
    ) -> Optional[dict[str, Optional[str]]]:
        return self.rows_by_program_id.get(program_id)

    def get_generic_class_explorer(self) -> Optional[dict[str, Optional[str]]]:
        return self.get_program_row("explore_object_of_class")

    def get_generic_object_explorer(self) -> Optional[dict[str, Optional[str]]]:
        return self.get_program_row("explore_attr_of_object")

    def _filter_category(
        self,
        category: str,
        **filters: Any,
    ) -> list[dict[str, Optional[str]]]:
        frame = [row for row in self.rows if row.get("category") == category]
        unknown_filters = sorted(
            filter(
                lambda column: column not in self.columns,
                filters.keys(),
            )
        )

        if unknown_filters:
            raise ValueError(
                f"Unknown filters for category '{category}': {', '.join(unknown_filters)}"
            )

        for column, value in filters.items():
            if value is None:
                continue

            if isinstance(value, (list, tuple, set, frozenset)):
                allowed_values = set(value)
                frame = [
                    row
                    for row in frame
                    if row.get(column) in allowed_values
                ]
            else:
                frame = [
                    row
                    for row in frame
                    if row.get(column) == value
                ]

        return frame

    def _infer_category_specific_filters(self) -> Dict[str, tuple[str, ...]]:
        sparse_columns = [
            column
            for column in self.columns
            if column != "category" and any(row.get(column) is None for row in self.rows)
        ]
        category_filters: Dict[str, tuple[str, ...]] = {}

        for category in self.categories:
            category_frame = [row for row in self.rows if row.get("category") == category]
            filters = tuple(
                column
                for column in sparse_columns
                if any(row.get(column) is not None for row in category_frame)
            )
            category_filters[category] = filters

        return category_filters

    def _normalize_category(self, category: str) -> str:
        normalized_category = self.CATEGORY_ALIASES.get(category, category)
        if normalized_category not in self.categories:
            available_categories = ", ".join(self.categories)
            raise ValueError(
                f"Unknown category '{category}'. Available categories: {available_categories}"
            )
        return normalized_category

    def _resolve_location(self, location: str | Path) -> Path:
        candidate = Path(location).expanduser()
        if candidate.is_file():
            return candidate.resolve()

        if not candidate.is_absolute():
            repo_relative = (self.repo_root / candidate).resolve()
            if repo_relative.is_file():
                return repo_relative

        raise FileNotFoundError(
            f"Could not resolve synthetic question CSV from location '{location}'. "
            "Provide a valid file path."
        )

    @classmethod
    def _load_rows(cls, file_path: Path) -> list[dict[str, Optional[str]]]:
        with open(file_path, newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            return [
                {
                    key: cls._normalize_cell(value)
                    for key, value in row.items()
                }
                for row in reader
            ]

    @staticmethod
    def _normalize_cell(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        normalized_value = value.strip()
        if not normalized_value or normalized_value.lower() in {"nan", "none"}:
            return None

        return normalized_value
