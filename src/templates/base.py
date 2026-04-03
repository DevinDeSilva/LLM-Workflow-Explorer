from pydantic import BaseModel as PydanticBaseModel

class BaseTemplate(PydanticBaseModel):
    template:str
    
    def format(self) -> str:
        return self.template.format(**self.model_dump())


