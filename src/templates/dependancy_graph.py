from typing import List

import dspy


class SubQuestionSignature(dspy.Signature):
    """
    Decompose the original user question into 2 to 4 logically connected smaller sub questions 
    that must be answered to answer the user question.

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
    original_question: str = dspy.InputField(
        desc="The question that should be answered."
    )
    answer: str = dspy.OutputField(
        desc="A direct answer to the original question."
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
