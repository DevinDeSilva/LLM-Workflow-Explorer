from src.templates.base import BaseTemplate

class TraceBasedExpOutputRequirements:
    system_name:str

class TraceBasedExpOutputFormat(BaseTemplate):
    system_name:str
    template = """
Knowledge Based System: {system_name}

"""