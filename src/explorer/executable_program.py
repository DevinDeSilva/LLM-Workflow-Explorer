from typing import Dict, Any, List, Union, Optional
import pandas as pd

class ExecutableProgram:
    """
    Represents an executable program with its metadata.
    """

    def __init__(
        self,
        program_id: str,
        name: str,
        solves: str,
        description: str,
        input_spec: Dict[str, Any],
        output_spec: Dict[str, Any],
        code: str,
        example_usage:str,
        example_output:pd.DataFrame,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.program_id: str = program_id
        self.name: str = name
        self.description: str = description
        self.input_spec: Dict[str, Any] = input_spec
        self.output_spec: Dict[str, Any] = output_spec
        self.code: str = code
        self.solves: str = solves
        self.example_usage: str = example_usage 
        self.example_output: pd.DataFrame = example_output
        self.tags: List[str] = tags if tags is not None else []
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}
        
    def to_dict(self) -> Dict[str, Union[str, int, Dict, List]]:
        return {
            "program_id": self.program_id,
            "name": self.name,
            "solves": self.solves,
            "description": self.description,
            "input_spec": self.input_spec,
            "output_spec": self.output_spec,
            "code": self.code,
            "example_usage": self.example_usage,
            "example_output": self.example_output.to_dict(orient="list"),
            "tags": self.tags,
            "metadata": self.metadata,
        }
