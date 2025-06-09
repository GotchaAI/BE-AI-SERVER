from typing import List

from fastapi import APIRouter, Body
from pydantic import BaseModel, Field

from src.chat.lulu import LuLuAI
import os
router = APIRouter(prefix = '/lulu', tags = ['LuLu'])

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

lulu = LuLuAI(api_key=OPENAI_API_KEY)



@router.get(
    "/start",
    summary="루루 게임 시작 요청 API",
    responses={
        200:
            {
                "description": "성공",
                "content": {
                    "application/json": {
                        "example" :{
                            "game_id" : "1"
                        }
                    }
                }
            }
    }
)
def start_game():
    game_id = lulu.create_game()
    return { "game_id" : game_id }


@router.get(
    "/task/{game_id}",
    summary = "루루가 키워드와 상황을 그림 과제를 제시합니다.",
    responses={
        200:{
            "description":"성공",
            "content": {
                "application/json" : {
                    "example" : {
                        "keyword" : "고양이",
                        "situation": "고양이가 나무 위에서 자고있는 모습"
                    }
                }
            }
        }
    }
)
def generate_task(game_id: str):
    task = lulu.generate_drawing_task(game_id)
    return task




class EvaluationReq(BaseModel):
    description: str = Field(..., description="그린 그림에 대한 설명")


@router.post(
    "/task/{game_id}",
    summary="그린 그림에 대한 설명을 루루에게 제출하고 평가를 받습니다.",
    responses={
        200:{
            "description":"성공",
            "content":{
                "application/json":{
                    "example":{
                        "score": 20,
                        "feedback": "뜨거운 태양과 모래사장이라... 이게 무슨 뜻이야? 시적 묘사를 제대로 이해하고 있나? 흐름과 장막, 마지막 이야기를 속삭이는 곳, 잃어버린 순간들이 춤추는 곳... 이런 모든 것들이 바다를 묘사하는 것이지. 너의 그림은 바다의 본질을 전혀 담아내지 못했어. 예술적 표현력이나 창의성은 어디에 있는 거야? 너의 그림은 완성도나 기법 면에서도 많이 부족하다. 다시 그려와.",
                        "task": {
                            "hidden_keyword": "바다",
                            "poetic_description": "무심한 흐름이 청아한 장막을 존중하며, 세상의 마지막 이야기를 속삭이는 곳, 이를테면 그곳은 용기와 두려움이 공존하는 곳. 언젠가 잃어버린 모든 순간들이 수면 아래에서 춤추는 곳...",
                            "game_id": "5055"
                        },
                        "game_id": "5055"
                    }
                }
            }
        }
    }
)
def evaluate_task(game_id: str, req: EvaluationReq = Body()):
    evaluation = lulu.evaluate_drawing(game_id, req.description)
    return evaluation
