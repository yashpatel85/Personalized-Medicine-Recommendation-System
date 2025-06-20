from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.recommendation.content_based import recommend_medicines

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))


app = FastAPI()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "app", "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "app", "templates"))


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "recommended_meds": None})


@app.post("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, condition: str = Form(...)):
    recommendations = recommend_medicines(condition)
    return templates.TemplateResponse("home.html", {
        "request": request,
        "recommended_meds": recommendations
    })
