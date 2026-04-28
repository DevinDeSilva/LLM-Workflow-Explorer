from typing import List

import dspy




class BuildTopologyGraphSignature(dspy.Signature):
    """
    Build a topology graph over the sub-questions and the main question.

    Represent the main question as Q, label each sub-question by number,
    and use '->' to show reasoning dependencies. Use ; to denote the
    seperation
    """

    original_question: str = dspy.InputField(
        desc="The main question represented as Q in the final graph."
    )
    application_context: str = dspy.InputField(
        desc="Description of the application and its functional scope."
    )
    sub_questions: List[str] = dspy.InputField(
        desc="Ordered sub-questions to connect in the reasoning topology."
    )
    topology_graph: str = dspy.OutputField(
        desc="A graph using numbered sub-questions and '->' edges ending at Q where appropriate."
    )




class UserQueryCoverageRewriteSignature(dspy.Signature):
    """
    Rewrite the user query into a self-contained question and list what the
    answer must cover to be complete. 
    
    Instructions 
    1. If the user query requests about any data (eg:- output data , input data). The question should EXPLICITLY request
       The values as literals stored in the Data Objects.
    """

    user_query: str = dspy.InputField(
        desc="Original natural language question from the user."
    )
    application_context: str = dspy.InputField(
        desc="Description of the application and its functional scope."
    )
    rewritten_user_query: str = dspy.OutputField(
        desc="A concise, self-contained rewrite of the user query that includes the required answer coverage."
    )
    answer_requirements: List[str] = dspy.OutputField(
        desc="Short list of things the final answer must cover to properly answer the user query."
    )


class SummarySignature(dspy.Signature):
    """
    Answer the original question using the provided relevant information.
    Be explanatory in your answer. If the given infomration is not enough to answer please
    explicitly say as such.
    """

    qa_dialog: str = dspy.InputField(
        desc="Relevant information or QA dialogue gathered during reasoning."
    )
    schema_context: str = dspy.InputField(
        desc="Compact ontology and schema summary relevant to the question."
    )
    application_context: str = dspy.InputField(
        desc="Description of the application and its functional scope."
    )
    original_question: str = dspy.InputField(
        desc="The question that should be answered."
    )
    answer: str = dspy.OutputField(
        desc="A verbose answer to the original question."
    )
    important_entities:str = dspy.OutputField(
        desc = "What are the KG entities used to come to this conclusion"
    )


class SyntheticQuestionGroundingSignature(dspy.Signature):
    """
    Ground the question to ontology classes, relations, and entity phrases that
    can be used to retrieve relevant synthetic SPARQL questions.

    Prefer CURIEs that appear in the provided available lists. Keep the output
    small and high precision.

    Instructions:
    1. Usually entity_phrases will be covered in ""
    3. Try to increase recall as much as possible, for candidates
    """

    question: str = dspy.InputField(
        desc="The question that should be answered."
    )
    application_context: str = dspy.InputField(
        desc="Description of the application and its functional scope."
    )
    schema_context: str = dspy.InputField(
        desc="Compact ontology and schema summary."
    )
    # predecessor_context: str = dspy.InputField(
    #     desc="Answers or evidence already gathered from predecessor nodes."
    # )
    available_classes: List[str] = dspy.InputField(
        desc="Available ontology classes that can be used as synthetic-question filters."
    )
    available_relations: List[str] = dspy.InputField(
        desc="Available ontology relations that can be used as synthetic-question filters."
    )
    candidate_classes: List[str] = dspy.OutputField(
        desc="Likely ontology classes relevant to the question, using only the provided available classes when possible."
    )
    candidate_relations: List[str] = dspy.OutputField(
        desc="Likely ontology relations relevant to the question, using only the provided available relations when possible."
    )
    entity_phrases: List[str] = dspy.OutputField(
        desc="Short free-text entity mentions or lookup phrases from the question that should be linked into the KG."
    )

class InitialDataSyntheticQuestion(dspy.Signature):
    """
    Choose which function to use to retrive data from a knowledge graph.
    You are given the function_id -> "question the function solves"

    Instructions:
    1. Only choose from prop / relation type function if the question gives hint to select
       a relation and entity_phrases eg:- {'prop':'dc:identifier', entity_phrase:['1_1']}.
    2. If no relation OR entity_phrases (these must be small MEANINGFUL text)
       choose "What are all the objects of a given class?"
    3. Default to What are all the objects of a given class? if unsure.

    Example 1:
    question:
    ONLY PROVIDE THE function_id
    """

    question: str = dspy.InputField(
        desc="The question that should be answered."
    )
    application_context: str = dspy.InputField(
        desc="Description of the application and its functional scope."
    )
    schema_context: str = dspy.InputField(
        desc="Compact ontology and schema summary."
    )
    entity_phrases: List[str] = dspy.InputField(
        desc="Entity mentions or lookup phrases extracted from the question."
    )
    functions: List[str] = dspy.InputField(
        desc="Functions in the format function_id : question the function solves"
    )
    function_id: str = dspy.OutputField(
        desc="function_id of selected function. ONLY PROVIDE THE function_id"
    )

    decision_reasoning: str = dspy.OutputField(
        desc="Brief reason for the chosen retrieval mode."
    )


class SelectSyntheticQuestionSignature(dspy.Signature):
    """
    Select the best synthetic question function to execute next.

    Choose exactly one function_id from the provided functions. The function_id
    should be copied exactly from the candidate list.

    ONLY PROVIDE THE function_id
    """

    question: str = dspy.InputField(
        desc="The current question or retrieval need."
    )
    application_context: str = dspy.InputField(
        desc="Description of the application and its functional scope."
    )
    schema_context: str = dspy.InputField(
        desc="Compact ontology and schema summary."
    )
    step_context: str = dspy.InputField(
        desc="Evidence and retrieval context gathered so far."
    )
    judge_context: str = dspy.InputField(
        desc="Current judge answer and feedback describing what is missing."
    )
    functions: List[str] = dspy.InputField(
        desc="Candidate functions in the format function_id : question the function solves."
    )
    entity_phrases: List[str] = dspy.InputField(
        desc="Entity mentions or lookup phrases relevant to the next retrieval."
    )
    function_id: str = dspy.OutputField(
        desc="Exact function_id of the selected function. Do not include the function description."
    )
    decision_reasoning: str = dspy.OutputField(
        desc="Brief reason for selecting this function."
    )


class SyntheticQuestionPathGroundingSignature(dspy.Signature):
    """
    Ground the next path-level synthetic question to available ontology classes.

    Use only class ids from available_classes when possible.
    """

    question: str = dspy.InputField(
        desc="The current question or retrieval need."
    )
    application_context: str = dspy.InputField(
        desc="Description of the application and its functional scope."
    )
    schema_context: str = dspy.InputField(
        desc="Compact ontology and schema summary."
    )
    step_context: str = dspy.InputField(
        desc="Evidence and retrieval context gathered so far."
    )
    judge_context: str = dspy.InputField(
        desc="Current judge answer and feedback describing what is missing."
    )
    available_classes: List[str] = dspy.InputField(
        desc="Available start-node ontology classes for path-level synthetic questions."
    )
    candidate_classes: List[str] = dspy.OutputField(
        desc="Likely start-node classes for path-level retrieval, copied from available_classes when possible."
    )
    entitys: List[str] = dspy.OutputField(
        desc="This are the associated entities and must be URIs (can use prefix)"
    )



class SyntheticQuestionParameterSignature(dspy.Signature):
    """
    Fill the SPARQL template placeholders for a selected synthetic question.

    Return a JSON object or JSON array of objects. Use prefixes if available else
    URI. 
    
    Instructions
    1. Use Either URI (with or without prefix) or strings withing qoutations. 
    2. if spec contains 
            prop_vlaue: then select important words from the question and return DO NOT SEND RETURN THE WHOLE QUESTION.
            obj, obj_uri: Must be entities in predecessor_context (Answers or evidence already available), Do NOT send Ontology entity , Select multiple if they apply
            class_url: Must be an Ontology/Schema entity.

    Formatting Instructions for parameter_values_json.
    1. for each object create a json list
    """

    question: str = dspy.InputField(
        desc="The question being answered."
    )
    application_context: str = dspy.InputField(
        desc="Description of the application and its functional scope."
    )
    schema_context: str = dspy.InputField(
        desc="Compact ontology and schema summary, this containes ontology prefixes     "
    )
    predecessor_context: str = dspy.InputField(
        desc="Answers or evidence already available."
        )
    step_spec: str = dspy.InputField(
        desc="The synthetic-question step being executed, including its program id, sub-question, and expected classes."
    )
    candidate_parameter_values: str = dspy.InputField(
        desc="Candidate parameter values derived from direct bindings, linked KG objects, and previous step outputs."
    )
    parameter_values_json: str = dspy.OutputField(
        desc="A JSON object or array of objects mapping SPARQL placeholder names to concrete values."
    )


class SyntheticQuestionResultSignature(dspy.Signature):
    """
    Summarize a synthetic-question step result in the context of the larger
    question and identify entities worth carrying forward.
    """

    question: str = dspy.InputField(
        desc="The original question being answered."
    )
    application_context: str = dspy.InputField(
        desc="Description of the application and its functional scope."
    )
    predecessor_context: str = dspy.InputField(
        desc="Answers or evidence already available before this step."
    )
    previous_step_results: str = dspy.InputField(
        desc="Natural-language summaries of earlier executed steps."
    )
    step_spec: str = dspy.InputField(
        desc="The synthetic-question step metadata."
    )
    sparql_results: str = dspy.InputField(
        desc="The concrete SPARQL query results for the current step."
    )
    answer: str = dspy.OutputField(
        desc="A grounded answer to the current step's sub-question."
    )
    important_entities: List[str] = dspy.OutputField(
        desc="Entities or values from the current step that should be considered by later steps."
    )


class ImportantEntitySelectionSignature(dspy.Signature):
    """
    Select the entities from the latest retrieval step that should be carried
    forward into the next retrieval step.
    """

    original_question: str = dspy.InputField(
        desc="The overall question being answered."
    )
    current_question: str = dspy.InputField(
        desc="The current question used for the latest retrieval step."
    )
    application_context: str = dspy.InputField(
        desc="Description of the application and its functional scope."
    )
    step_context: str = dspy.InputField(
        desc="Context accumulated before the latest retrieval step."
    )
    judge_context: str = dspy.InputField(
        desc="Current judge answer and feedback describing what is missing."
    )
    latest_step_results: str = dspy.InputField(
        desc="Latest retrieval step results containing candidate entity URIs and attributes."
    )
    candidate_entities: List[str] = dspy.InputField(
        desc="Candidate entity URIs from the latest retrieval step."
    )
    important_entities: List[str] = dspy.OutputField(
        desc="Entity URIs from candidate_entities that should be passed to the next step."
    )
    selection_reasoning: str = dspy.OutputField(
        desc="Brief reason these entities are relevant for the next retrieval step."
    )


class SyntheticQuestionNextStepSignature(dspy.Signature):
    """
    Decide the next single question to execute in the traversal.

    Use the original question as the overall objective. Ground the next
    question in the latest retrieved evidence and the judge feedback. Return
    one concise question that should move the reasoning forward.
    """

    original_question: str = dspy.InputField(
        desc="The overall question that the traversal is trying to answer."
    )
    current_question: str = dspy.InputField(
        desc="The question used in the current traversal round."
    )
    application_context: str = dspy.InputField(
        desc="Description of the application and its functional scope."
    )
    schema_context: str = dspy.InputField(
        desc="Compact ontology and schema summary."
    )
    step_context: str = dspy.InputField(
        desc="Accumulated traversal context from previous rounds."
    )
    latest_step_results: str = dspy.InputField(
        desc="Natural-language summary of the latest executed step."
    )
    partial_answer: str = dspy.InputField(
        desc="The current best grounded answer after the latest step."
    )
    judge_feedback: str = dspy.InputField(
        desc="What is still missing according to the judge."
    )
    next_question: str = dspy.OutputField(
        desc="The next concise question that should be executed to move closer to answering the original question."
    )



class LeafNodeVerificationSignature(dspy.Signature):
    """
    Decide whether the question can be answered directly.

    If it cannot be answered, return NO!. Otherwise return a simple answer.
    """

    question: str = dspy.InputField(
        desc="The leaf-node question to answer."
    )
    answer: str = dspy.OutputField(
        desc="Either NO! or a simple answer to the question."
    )


class LeafNodeRAGVerificationSignature(dspy.Signature):
    """
    Answer the question using the supplied context if possible.

    If the context is insufficient, return NO!.
    """

    context: str = dspy.InputField(
        desc="Context information available for answering the question."
    )
    question: str = dspy.InputField(
        desc="The question to answer from the context."
    )
    answer: str = dspy.OutputField(
        desc="Either NO! or an answer grounded in the context."
    )


class LeafNodeDecompositionSignature(dspy.Signature):
    """
    Identify unresolved aspects of the original question given the current context
    and decompose them into two simple sub-questions.
    """

    original_question: str = dspy.InputField(
        desc="The original question being reasoned about."
    )
    context: str = dspy.InputField(
        desc="Context already available for the original question."
    )
    sub_questions: List[str] = dspy.OutputField(
        desc="Two simple sub-questions focused on unresolved aspects not already covered by the context."
    )


class LeafNodeAnswerSignature(dspy.Signature):
    """
    Answer the question using the provided relevant information.
    """

    context: str = dspy.InputField(
        desc="Relevant information for answering the question."
    )
    question: str = dspy.InputField(
        desc="The question to answer."
    )
    answer: str = dspy.OutputField(
        desc="A direct answer to the question."
    )


class InternalNodeRewriteSignature(dspy.Signature):
    """
    Rewrite the main question by incorporating the answers from its sub-questions
    while avoiding repetition.
    """

    qa_dialog: str = dspy.InputField(
        desc="Answers collected from sub-questions."
    )
    question: str = dspy.InputField(
        desc="The current main question to rewrite."
    )
    original_question: str = dspy.InputField(
        desc="The original question that the rewritten question should still support."
    )
    rewritten_question: str = dspy.OutputField(
        desc="A concise rewritten question that incorporates the sub-question answers."
    )


class InternalNodeDecompositionSignature(dspy.Signature):
    """
    Identify unresolved aspects of the original question after considering the
    available information and decompose them into two new sub-questions.
    """

    original_question: str = dspy.InputField(
        desc="The original question being reasoned about."
    )
    context: str = dspy.InputField(
        desc="Relevant context already available."
    )
    qa_dialog: str = dspy.InputField(
        desc="Answers from previous sub-questions."
    )
    sub_questions: List[str] = dspy.OutputField(
        desc="Two new sub-questions covering unresolved aspects not addressed by the current information."
    )
