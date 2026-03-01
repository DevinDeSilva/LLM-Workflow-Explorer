from typing import Dict, Any, List, Optional
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