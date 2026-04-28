from __future__ import annotations

import pandas as pd
from typing import Any, Dict, List, Optional


class SQRetriver:

    def __init__(self, location: str) -> None:
        self.rows = pd.read_csv(location)

    @staticmethod
    def _as_list(value: Optional[Any]) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value if str(item).strip()]

    @staticmethod
    def _first_record(select_row: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if select_row.empty:
            return None
        return select_row.iloc[0].where(pd.notna(select_row.iloc[0])).to_dict()

    def get_programs_from_prop(
        self,
        relation: Optional[List[str]] = None,
        class_candidates: Optional[List[str]] = None,
    ):
        select_row = self.rows[self.rows["category"] == "object-level|from-prop"]
        relation = self._as_list(relation)
        class_candidates = self._as_list(class_candidates)

        if relation:
            select_row = select_row[select_row["focal_relation"].isin(relation)]

        if class_candidates:
            select_row = select_row[select_row["focal_node"].isin(class_candidates)]

        return select_row[["program_id", "solves"]].to_dict(orient="records")

    def get_programs_from_prop_metadata(self):
        select_row = self.rows[self.rows["category"] == "object-level|from-prop"]
        select_row = select_row[["focal_node", "focal_relation"]]
        select_row.columns = ["class", "relation"]
        return select_row.to_dict(orient="records")
    
    def get_programs_path_metadata(self):
        select_row = self.rows[self.rows["category"] == "path-level"]
        select_row = select_row[["start_node", "end_node"]]
        return select_row.to_dict(orient="records")


    def get_objects_of_a_class(self, class_uri: Optional[str] = None):
        select_row = self.rows[self.rows["program_id"] == "explore_object_of_class"]
        return self._first_record(select_row)

    def get_attr_of_object(self):
        select_row = self.rows[self.rows["program_id"] == "explore_attr_of_object"]
        return self._first_record(select_row)


    def get_path_questions(
        self,
        start_nodes: Optional[List[str]] = None,
        end_nodes: Optional[List[str]] = None,
    ):
        select_row = self.rows[self.rows["category"] == "path-level"]
        start_nodes = self._as_list(start_nodes)
        end_nodes = self._as_list(end_nodes)
        
        focal_nodes = list(set(start_nodes+end_nodes))

        if focal_nodes:
            select_row = select_row[select_row["start_node"].isin(focal_nodes) | select_row["end_node"].isin(focal_nodes)]

        return select_row[["program_id", "solves"]].to_dict(orient="records")

    def get_program_by_id(self, _id:str, solves_only:bool = False):
        select_row = self.rows[self.rows["program_id"] == _id]
        rec = self._first_record(select_row)
        
        if rec and solves_only:
            return rec['solves'] 
        
        return rec
    
