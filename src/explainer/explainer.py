from src.explainer.dependancy_graph import DependancyGraphCreation
from src.config.experiment import ExplainerConfig

class Explainer:
    def __init__(self, config:ExplainerConfig) -> None:
        self.dependancy_graph = DependancyGraphCreation(
            llm_type = config.llm_type,
            model = config.model,
            llm_config = config.llm_config
        )
    
    def request(self, user_query:str):
        user_query = user_query.strip()
        print("fuck", user_query)