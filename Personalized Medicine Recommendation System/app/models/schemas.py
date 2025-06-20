from pydantic import BaseModel

class RecommendationRequest(BaseModel):
    query: str
    top_n: int = 5


from pydantic import BaseModel

class HybridRequest(BaseModel):
    symptom: str
