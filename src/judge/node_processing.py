import dspy
from typing import List


class AnsweredSignature(dspy.Signature):
    """
    Does the answer provided answers the question asked are Data literals present?. If not provide feedback on what is missing and what needs
    to be revised in the answer.

    Instructions:
    
    1. If the answer says a specific identity doesn't seem to exist make the answer False as data 
       might be need to retrived. NOTE majority of the time data exist in the graph the retrival is just incomplete Therefore answered=false
    2. If the answer heavily Depend on Data objects the answer should contain literal values not just URIs reflect this in your feedback and rerun
    3. Note if the program_id = 'explore_object_of_class' assume all the data are retrived.
    4. Ensure that the answer covers all aspects in the question. 
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
    evidence_context: str = dspy.InputField(
        desc="Grounded schema details and execution evidence collected so far."
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
