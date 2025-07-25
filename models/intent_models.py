from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class IntentAction(str, Enum):
    USE_EXISTING = "use_existing"
    CREATE_NEW = "create_new"

class IntentCategory(BaseModel):
    name: str
    description: str
    examples: List[str] = []
    
    def volume(self) -> int:
        return len(self.examples)

class IntentClassification(BaseModel):
    conversation_id: str
    intent_name: str
    intent_description: str
    action: IntentAction
    confidence: float
    reasoning: str

class IntentOntology(BaseModel):
    categories: List[IntentCategory] = []
    
    def get_category_names(self) -> List[str]:
        return [cat.name for cat in self.categories]
    
    def add_category(self, category: IntentCategory) -> None:
        if category.name not in self.get_category_names():
            self.categories.append(category)
    
    def get_category(self, name: str) -> Optional[IntentCategory]:
        for cat in self.categories:
            if cat.name == name:
                return cat
        return None