import dspy
from icecream import List

class AnsweredSignature(dspy.Signature):
    """
    Does the answer provided answers the question asked. If not provide feedback on what is missing and what needs 
    to be revised in the answer.
    """

    question: str = dspy.InputField(
        desc="Natural language question."
    )
    answer: str = dspy.InputField(
        desc="Natural language question."
    )
    answered: bool = dspy.OutputField(
        desc="Boolean indicating if the question has been answered."
    )
    feedback: str = dspy.OutputField(
        desc="Feedback on what is missing or needs to be revised in the answer."
    )