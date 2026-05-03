import pandas as pd
from icecream import ic
import dycomutils as common_utils
from typing import Dict, Iterable, List, Optional, Tuple

class OntologyInfoRetriever:
    LABEL_PREDICATES: Tuple[str, ...] = ("rdfs:label", "skos:prefLabel")
    ALIAS_PREDICATES: Tuple[str, ...] = ("http://purl.org/dc/terms/alternative",)
    DESCRIPTION_PREDICATES: Tuple[str, ...] = (
        "prov:definition",
        "skos:definition",
        "dc:description",
        "rdfs:comment",
    )

    def __init__(self, 
                 file_path:str,
                 schema_path:Optional[str] = None
                 ) -> None:
        self.df = pd.read_csv(
            file_path
        )
        self.schema = None
        
        if schema_path:
            self.schema = common_utils.serialization.load_json(
                schema_path
            )

    def get_schema_components(
        self,
    ) -> Tuple[List[str], Dict[str, Dict[str, List[Dict[str, str]]]]]:
        schema = self.schema or {}
        return schema.get("classes", []), schema.get("object_properties", {})

    def get_schema_terms(self) -> List[str]:
        classes, object_properties = self.get_schema_components()
        return classes + list(object_properties.keys())

    def build_metadata_index(
        self,
        subjects: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, List[str]]]:
        if subjects is None:
            subjects = self.get_schema_terms()

        triples_df = self.filter_triples(subjects, str_fmt=False)
        metadata_by_subject: Dict[str, Dict[str, List[str]]] = {}

        if triples_df.empty:
            return metadata_by_subject

        for triple in triples_df.itertuples(index=False):
            subject = str(triple.s)
            predicate = str(triple.p)
            obj = self._compact_text(triple.o)

            if not obj:
                continue

            subject_metadata = metadata_by_subject.setdefault(subject, {})
            predicate_values = subject_metadata.setdefault(predicate, [])
            if obj not in predicate_values:
                predicate_values.append(obj)

        return metadata_by_subject

    def format_schema_prompt(
        self,
        description_limit: int = 180,
    ) -> str:
        classes, object_properties = self.get_schema_components()
        metadata_index = self.build_metadata_index(classes + list(object_properties.keys()))

        if not classes and not object_properties:
            return "Schema context unavailable."

        prompt_lines = [
            "Workflow schema",
            "Classes:",
        ]

        prompt_lines.extend(
            f"- {self.summarize_schema_term(class_name, metadata_index, description_limit)}"
            for class_name in classes
        )

        prompt_lines.append("Object properties:")
        for property_name, property_info in object_properties.items():
            property_summary = self.summarize_schema_term(
                property_name,
                metadata_index,
                description_limit,
            )
            connection_text = self._format_connections(property_info.get("connections", []))

            if connection_text:
                prompt_lines.append(f"- {property_summary}; {connection_text}")
            else:
                prompt_lines.append(f"- {property_summary}")

        return "\n".join(prompt_lines)

    def summarize_schema_term(
        self,
        term: str,
        metadata_index: Optional[Dict[str, Dict[str, List[str]]]] = None,
        description_limit: int = 180,
    ) -> str:
        if metadata_index is None:
            metadata_index = self.build_metadata_index([term])

        label = self._first_value(term, self.LABEL_PREDICATES, metadata_index)
        description = self._first_value(
            term,
            self.DESCRIPTION_PREDICATES,
            metadata_index,
        )

        summary = term
        details: List[str] = []

        if label and label != term:
            details.append(label)
        # if alias:
        #     details.append(f"alias={alias}")

        if details:
            summary = f"{summary} ({'; '.join(details)})"

        if description:
            summary = f"{summary}: {self._truncate_text(description, description_limit)}"

        return summary

    def _first_value(
        self,
        subject: str,
        predicates: Iterable[str],
        metadata_index: Dict[str, Dict[str, List[str]]],
    ) -> Optional[str]:
        for predicate in predicates:
            values = metadata_index.get(subject, {}).get(predicate, [])
            if values:
                return values[0]
        return None

    @staticmethod
    def _compact_text(value: object) -> str:
        return " ".join(str(value).split()).strip()

    @staticmethod
    def _truncate_text(value: str, limit: int) -> str:
        if limit <= 0 or len(value) <= limit:
            return value

        cutoff = max(limit - 3, 0)
        truncated = value[:cutoff].rstrip()
        if " " in truncated:
            truncated = truncated.rsplit(" ", 1)[0]
        return f"{truncated}..."

    @staticmethod
    def _format_connections(connections: List[Dict[str, str]]) -> str:
        if not connections:
            return ""

        unique_connections: List[str] = []
        for connection in connections:
            domain = connection.get("domain", "Unknown")
            range_ = connection.get("range", "Unknown")
            connection_text = f"{domain} -> {range_}"
            if connection_text not in unique_connections:
                unique_connections.append(connection_text)

        return f"links {', '.join(unique_connections)}"
            
    
    def filter_triples(
        self,
        objects: Optional[list] = None,
        columns: Optional[list[str]] = None,
        _exclude: Dict[str, List[str]] = {
            "p":[
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "rdfs:subClassOf"
                ]
        },
        str_fmt:bool = True,
        debug:bool = False
    ):
        if not objects:
            return pd.DataFrame()

        if columns is None:
            columns = ["s"]

        _index: pd.Series = pd.Series(True, index=self.df.index)
        for col in self.df.columns:
            if col in columns:
                _index = _index & self.df[col].isin(objects)
            if col in _exclude.keys():
                _index = _index & (~self.df[col].isin(_exclude[col]))

        filtered_df = self.df.loc[_index, :].sort_values(by="s")
        
        if str_fmt:
            _formatted_string = OntologyInfoRetriever.triple_fmt_str(
                filtered_df
                )
            
            if debug:
                ic(_formatted_string)
            return _formatted_string
        else: 
            return filtered_df
            

    @staticmethod
    def triple_fmt_str(
        df: pd.DataFrame
        ) -> str:
        required_columns = ["s", "p", "o"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                "triple_fmt_str requires dataframe columns: "
                + ", ".join(required_columns)
            )

        triples_df = (
            df.loc[:, required_columns]
            .dropna(subset=required_columns)
            .astype(str)
            .drop_duplicates()
            .sort_values(by=required_columns, kind="stable")
        )

        if triples_df.empty:
            return "No ontology triples available."

        prompt_lines = [
            "Ontology triples:",
            "Each line is formatted as `predicate -> object` under its subject.",
        ]

        for subject, subject_triples in triples_df.groupby("s", sort=False):
            prompt_lines.append(f"Subject: {subject}")
            for triple in subject_triples.itertuples(index=False):
                prompt_lines.append(f"- {triple.p} -> {triple.o}")

        return "\n".join(prompt_lines)
