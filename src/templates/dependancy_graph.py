from typing import List

import dspy


class SubQuestionSignature(dspy.Signature):
    """
    Decompose the original user question into 2 to 4 logically connected smaller sub questions 
    that must be answered to answer the user question. Ensure that each question leads to a next question
    while ensuring that no semantically similar questions. You may denotes questions that are answered by
    context (previous question answer context or application context) or retrieve information
    from the execution KG.

    Return only list of concise strings. Each string should describe
    one information need, not a search strategy or an answer.
    """

    user_query: str = dspy.InputField(
        desc="Natural language question from the user."
    )
    schema_context: str = dspy.InputField(
        desc="Compact ontology and schema summary for valid classes and relations."
    )
    application_context: str = dspy.InputField(
        desc="Description of the application and its functional scope."
    )
    sub_questions: List[str] = dspy.OutputField(
        desc="Two to four logically connected sub-questions. These questions need to be answered to answer the main question"
    )


class BuildTopologyGraphSignature(dspy.Signature):
    """
    Build a topology graph over the sub-questions and the main question.

    Represent the main question as Q, label each sub-question by number,
    and use '->' to show reasoning dependencies.
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


class SubQuestionVerificationSignature(dspy.Signature):
    """
    Review sub-questions and identify the ones that should be filtered out.

    Filter any sub-question that is irrelevant, misses critical context,
    or semantically repeats another sub-question.
    """

    original_question: str = dspy.InputField(
        desc="The original question the sub-questions should support."
    )
    application_context: str = dspy.InputField(
        desc="Description of the application and its functional scope."
    )
    sub_questions: List[str] = dspy.InputField(
        desc="Candidate sub-questions to evaluate."
    )
    filtered_sub_question: List[int] = dspy.OutputField(
        desc="The numbers of sub-questions that should be filtered out."
    )


class SummarySignature(dspy.Signature):
    """
    Answer the original question using the provided relevant information.
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
        desc="A direct answer to the original question."
    )


class SyntheticQuestionGroundingSignature(dspy.Signature):
    """
    Ground the question to ontology classes, relations, and entity phrases that
    can be used to retrieve relevant synthetic SPARQL questions.

    Prefer CURIEs that appear in the provided available lists. Keep the output
    small and high precision.
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
    predecessor_context: str = dspy.InputField(
        desc="Answers or evidence already gathered from predecessor nodes."
    )
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


class SyntheticQuestionPlanningSignature(dspy.Signature):
    """
    Build a short ordered execution plan using the provided synthetic questions.

    Return a JSON array. Each step object must contain:
    - step_id
    - sub_question
    - program_id
    - input_bindings
    - expected_classes

    `program_id` must exactly match one of the provided candidates.
    `input_bindings` may use:
    - ontology CURIEs or full IRIs,
    - free-text lookup phrases,
    - STEP:<step_id> to reference entities from a previous step.
    """

    question: str = dspy.InputField(
        desc="The question being answered."
    )
    application_context: str = dspy.InputField(
        desc="Description of the application and its functional scope."
    )
    schema_context: str = dspy.InputField(
        desc="Compact ontology and schema summary."
    )
    predecessor_context: str = dspy.InputField(
        desc="Answers or evidence already available."
    )
    candidate_synthetic_questions: str = dspy.InputField(
        desc="Candidate synthetic questions with program ids, filters, and query inputs."
    )
    execution_plan_json: str = dspy.OutputField(
        desc="A JSON array describing the ordered execution plan."
    )


class SyntheticQuestionParameterSignature(dspy.Signature):
    """
    Fill the SPARQL template placeholders for a selected synthetic question.

    Return a JSON object or JSON array of objects. Use full IRIs for class or
    object URI placeholders. Use raw strings only for literal filter values.
    """

    question: str = dspy.InputField(
        desc="The question being answered."
    )
    application_context: str = dspy.InputField(
        desc="Description of the application and its functional scope."
    )
    predecessor_context: str = dspy.InputField(
        desc="Answers or evidence already available."
    )
    previous_step_results: str = dspy.InputField(
        desc="Results from previously executed synthetic-question steps."
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


class AnswerRevisionSignature(dspy.Signature):
    """
    Revise an answer using judge feedback while staying faithful to the provided
    evidence and without inventing facts.
    """

    question: str = dspy.InputField(
        desc="The question that should be answered."
    )
    application_context: str = dspy.InputField(
        desc="Description of the application and its functional scope."
    )
    predecessor_context: str = dspy.InputField(
        desc="Answers or evidence already gathered from predecessor nodes."
    )
    answer: str = dspy.InputField(
        desc="The current draft answer."
    )
    feedback: str = dspy.InputField(
        desc="Judge feedback describing what is missing or incorrect."
    )
    evidence_context: str = dspy.InputField(
        desc="Grounded execution evidence and prior answers."
    )
    revised_answer: str = dspy.OutputField(
        desc="A revised answer that addresses the feedback while remaining grounded in the evidence."
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
