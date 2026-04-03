from typing import List
import dspy


class SchemaInfoSignature(dspy.Signature):
    """
    You need to check if the question given need information about the ontology schema to answer
    the question if so select the relevant information from the schema. You don't need to answer the question
    just need to identify the relevant schema information that is required to answer the question and return 
    it.
    """

    question: str = dspy.InputField(
        desc="Natural language question."
    )
    need_schema_info: bool = dspy.InputField(
        desc="Boolean indicating if schema information is needed."
    )
    relevant_schema_info: List[str] = dspy.OutputField(
        desc="Relevant schema information needed to answer the question."
    )
    
class IRRequirmeenteSignature(dspy.Signature):
    """
    Identify whether information requirements are needed to answer the question. You are given a Knowledge graph backed by an 
    ontology. If needed provide the candidate entities or classes that the information maybe contained in.
    """
    
    question: str = dspy.InputField(
        desc="Natural language question."
    )
    
    schema_info: List[str] = dspy.InputField(
        desc="Ontology schema information."
    )
    
    need_retrival: bool = dspy.InputField(
        desc="Boolean indicating if information retrieval is needed."
    )
    
    candidate_entities: List[str] = dspy.OutputField(
        desc="Candidate entities that may contain the information needed to answer the question. use the question to guide you."
    )
    
    candidate_classes: List[str] = dspy.OutputField(
        desc="Candidate classes that may contain the information needed to answer the question. use the question to guide you."
    )