import pandas as pd
from icecream import ic
from typing import Optional, List , Dict

class OntologyInfoRetriever:
    def __init__(self, file_path:str) -> None:
        self.df = pd.read_csv(
            file_path
        )
    
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