from typing import List

import dspy

def build_information_required_fewshot_examples() -> List[dspy.Example]:
        return [
            dspy.Example(
                user_query="<ADD_USER_QUERY_EXAMPLE_1>",
                schema_context="Schema context will be supplied at runtime.",
                application_context="Application context will be supplied at runtime.",
                information_required=[
                    "<ADD_INFORMATION_REQUIRED_ITEM_1>",
                    "<ADD_INFORMATION_REQUIRED_ITEM_2>",
                ],
            ).with_inputs(
                "user_query",
                "schema_context",
                "application_context",
            ),
            dspy.Example(
                user_query="<ADD_USER_QUERY_EXAMPLE_2>",
                schema_context="Schema context will be supplied at runtime.",
                application_context="Application context will be supplied at runtime.",
                information_required=[
                    "<ADD_INFORMATION_REQUIRED_ITEM_1>",
                    "<ADD_INFORMATION_REQUIRED_ITEM_2>",
                    "<ADD_INFORMATION_REQUIRED_ITEM_3>",
                ],
            ).with_inputs(
                "user_query",
                "schema_context",
                "application_context",
            ),
        ]
