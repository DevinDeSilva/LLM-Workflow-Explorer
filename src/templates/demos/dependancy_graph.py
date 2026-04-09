from typing import List

import dspy

def build_information_required_fewshot_examples() -> List[dspy.Example]:
        return [
            dspy.Example(
                user_query="what are the unique executions that ran",
                application_context="Application context will be supplied at runtime.",
                information_required=[
                    "How is a 'unique experiment execution' defined in the context of the application",
                    "What attribute can be used to detect 'uniquenes' "
                    "what are the unique instances of executions insances",
                ],
            ).with_inputs(
                "user_query",
                "schema_context",
                "application_context",
            ),
            dspy.Example(
                user_query="what are the inputs used by LLMs to generate the function 'information extraction'",
                application_context="Application context will be supplied at runtime.",
                information_required=[
                    "what is the entity in the kg related to the function 'information extraction'",
                    "what is the input LLM Method used to generate the function",
                    "what are the inputs input entities used"
                ],
            ).with_inputs(
                "user_query",
                "schema_context",
                "application_context",
            ),
        ]
