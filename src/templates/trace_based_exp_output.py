from src.templates.base import BaseTemplate

class TraceBasedExpOutputRequirements:
    system_name:str

class TraceBasedExpOutputFormat(BaseTemplate):
    template = """
    ### Overall Answer to the Question:
    {overall_answer}
    
    ### Knowledge Based System: 
    {system_name}

    ### What were the system outputs associated with the user query and the system trace?: 
    (System Recommendation)
    {system_recommendation}

    ### What are the entities associate with the question: 
    (wasGeneratedBy)
    {generated_by}

    ### Traces Associated with the system recommendation:
    (System Trace)
    {system_trace}
    """
    
    system_name:str
    system_recommendation:str
    generated_by:str
    system_trace:str
    overall_answer:str