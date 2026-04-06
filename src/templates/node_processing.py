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
    need_schema_info: bool = dspy.OutputField(
        desc="Boolean indicating if schema information is needed."
    )
    relevant_schema_info: List[str] = dspy.OutputField(
        desc="Relevant schema information needed to answer the question."
    )


class SchemaAnswerabilitySignature(dspy.Signature):
    """
    Decide whether the question can be fully answered from ontology schema
    information alone, without querying workflow instance data.
    """

    question: str = dspy.InputField(
        desc="Natural language question."
    )
    application_context: str = dspy.InputField(
        desc="Description of the application and its functional scope."
    )
    schema_context: str = dspy.InputField(
        desc="Compact ontology and schema summary."
    )
    predecessor_context: str = dspy.InputField(
        desc="Answers or evidence already gathered from predecessor nodes."
    )
    answerable_from_schema: bool = dspy.OutputField(
        desc="Boolean indicating whether the question can be fully answered from schema information alone."
    )
    relevant_schema_info: List[str] = dspy.OutputField(
        desc="The smallest set of schema facts from the provided schema context that are needed to support the answer."
    )


class SchemaAnswerSignature(dspy.Signature):
    """
    Answer the question using only the provided schema details.

    Keep the answer grounded in the schema and do not invent instance-level
    facts or workflow executions.
    """

    question: str = dspy.InputField(
        desc="Natural language question."
    )
    application_context: str = dspy.InputField(
        desc="Description of the application and its functional scope."
    )
    predecessor_context: str = dspy.InputField(
        desc="Answers or evidence already gathered from predecessor nodes."
    )
    relevant_schema_info: str = dspy.InputField(
        desc="Relevant schema facts selected from the ontology summary."
    )
    answer: str = dspy.OutputField(
        desc="A direct answer grounded only in the provided schema details."
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
    
    need_retrival: bool = dspy.OutputField(
        desc="Boolean indicating if information retrieval is needed."
    )
    
    candidate_entities: List[str] = dspy.OutputField(
        desc="Candidate entities that may contain the information needed to answer the question. use the question to guide you."
    )
    
    candidate_classes: List[str] = dspy.OutputField(
        desc="Candidate classes that may contain the information needed to answer the question. use the question to guide you."
    )
