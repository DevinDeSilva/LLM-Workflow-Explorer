import dspy

class AnsweredSignature(dspy.Signature):
    """
    Does the answer provided answers the question asked. If not provide feedback on what is missing and what needs 
    to be revised in the answer.
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
    answer: str = dspy.InputField(
        desc="Natural language question."
    )
    answered: bool = dspy.OutputField(
        desc="Boolean indicating if the question has been answered."
    )
    feedback: str = dspy.OutputField(
        desc="Feedback on what is missing or needs to be revised in the answer."
    )
