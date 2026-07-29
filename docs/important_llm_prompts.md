# Evaluation Prompts

**Description:**  
The following prompts are used for LLM-as-a-judge evaluation. The first assigns normalized scores for *completeness*, *faithfulness*, and *relevance*. The second performs pairwise comparison for win-rate analysis based on faithfulness and understandability. The third determines the next synthetic question to execute during iterative traversal.

## AnswerQualityJudgeSignature

*DSPy prompt used for absolute LLM-as-a-judge scoring of generated explanations.*

```python
class AnswerQualityJudgeSignature(dspy.Signature):
    """Score answer completeness, accuracy, and relevance against
    the ground truth."""

    question: str = dspy.InputField()
    ground_truth_answer: str = dspy.InputField()
    model_answer: str = dspy.InputField()

    completeness: float = dspy.OutputField(
        desc="How much of the ground truth is covered, from 0 to 1."
    )
    faithfulness: float = dspy.OutputField(
        desc="How faithful the answer is to the ground truth, from 0 to 1."
    )
    relevance: float = dspy.OutputField(
        desc="How directly the answer addresses the question, from 0 to 1."
    )
```

## PairwiseAnswerWinrateSignature

*DSPy prompt used for pairwise LLM-as-a-judge win-rate evaluation.*

```python
class PairwiseAnswerWinrateSignature(dspy.Signature):
    """You are an evaluator and you are given outputs of two
    explanation generation systems for AI systems. Your job is
    to identify which method provides a better explanation to
    the asked question. A human-curated answer is also provided.

    Choose which answer is more understandable by a lay user.
    Be cautious whether the answer is faithful to the human-curated
    answer.
    """

    question: str = dspy.InputField()
    ground_truth_answer: str = dspy.InputField()
    method_a: str = dspy.InputField()
    answer_a: str = dspy.InputField()
    method_b: str = dspy.InputField()
    answer_b: str = dspy.InputField()
    winner: str = dspy.OutputField(
        desc="Return exactly one of: method_a, method_b, tie."
    )
```

## SyntheticQuestionNextStepSignature

*DSPy prompt used for question updating at each loop of the explanation QA system.*

```python
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
    partial_answer: str = dspy.InputField(
        desc="The current best grounded answer after the latest step."
    )
    judge_feedback: str = dspy.InputField(
        desc="What is still missing according to the judge."
    )
    next_question: str = dspy.OutputField(
        desc=(
            "The next concise question that should be executed to move closer "
            "to answering the original question."
        )
    )
```