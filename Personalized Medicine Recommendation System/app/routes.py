from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import RecommendationRequest
from recommendation.content_based import recommend_medicines

import os

router = APIRouter()

# Load the dataset once
dataset_path = os.path.join("data", "D:\Personalized Medicine Recommendation System\project\data\drugsComTrain_raw.csv")
drug_df = recommend_medicines.load_drug_data(dataset_path)

@router.post("/recommend")
def recommend_medicine(request: RecommendationRequest):
    try:
        results = recommend_medicines.get_top_recommendations(request.query, drug_df, request.top_n)
        return {"recommendations": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from fastapi import APIRouter, Depends, HTTPException
from app.auth import get_current_user
from app.recommendation.hybrid import hybrid_recommend
from app.models.schemas import HybridRequest

router = APIRouter()

@router.post("/hybrid-recommend")
def hybrid_endpoint(req: HybridRequest, user: dict = Depends(get_current_user)):
    try:
        recommendations = hybrid_recommend(user_id=user['id'], symptom=req.symptom)
        return {"recommended_medicines": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
