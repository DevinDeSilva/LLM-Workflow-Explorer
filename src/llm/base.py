from typing import Optional, Type,  Dict, Any, List
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage

class BaseLlm:
    def __init__(self, config, library:str):
        self.config = config
        self.library = library
        self.llm = self._create_client()
        
    def _create_client(self):
        raise NotImplementedError

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Asynchronously generates a standard text response.
        """
        messages:List[Any] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
            
        messages.append(HumanMessage(content=prompt))
        
        # Use ainvoke() for LangChain's asynchronous execution
        response = await self.llm.ainvoke(messages)
        
        # LangChain returns an AIMessage object; we extract the string content
        return str(response.content)

    async def structured_generate(
        self, 
        prompt: str, 
        structure: Type[BaseModel], 
        system_prompt: Optional[str] = ""
    ) -> Dict[str, Any]:
        """
        Asynchronously generates a response forced to match a Pydantic schema.
        """
        messages:List[Any] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
            
        messages.append(HumanMessage(content=prompt))
        
        # Bind the Pydantic schema directly to the model
        structured_llm = self.llm.with_structured_output(structure)
        
        # The result will be a fully validated instance of your Pydantic model
        result = await structured_llm.ainvoke(messages)
        
        # Since your signature expects a dict, we dump the Pydantic model to a dict
        if isinstance(result, dict):
            return result
            
        # 2. If LangChain returned the instantiated Pydantic model, dump it.
        # By checking isinstance, Pylance now knows it's safe to call model_dump().
        elif isinstance(result, BaseModel):
            return result.model_dump()
            
        # 3. Fallback for unexpected behavior
        else:
            raise TypeError(f"Unexpected return type from LangChain: {type(result)}")
