from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List
from src.core.myomyo import MyoMyoAI
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
myomyo = MyoMyoAI(api_key=OPENAI_API_KEY)

router = APIRouter(
    prefix="/myomyo",
    tags=["MyoMyo AI"]
)


class GPTResponse(BaseModel):
    message: str = Field(..., description="AI가 생성한 메시지")

# START_GAME
class GameStartReq(BaseModel):
    players: List[str] = Field(..., description="게임에 참여할 플레이어 이름 List")

@router.post("/{game_id}/start", summary="게임 시작 메시지 API")
async def start_game(game_id: str, request: GameStartReq = Body(...)):
    message = await myomyo.game_start_message(game_id=game_id, players=request.players)
    return GPTResponse(message=message)

class RoundStartReq(BaseModel):
    round_num: int = Field(..., description="현재 라운드 번호")
    total_rounds: int = Field(..., description="전체 라운드 수")

@router.post("/{game_id}/round/start", summary="라운드 시작 메시지 API")
async def start_round(game_id: str, request: RoundStartReq = Body(...)):
    message = await myomyo.round_start_message(
        game_id=game_id,
        round_num=request.round_num,
        total_rounds=request.total_rounds
    )
    return GPTResponse(message=message)


class GuessStartReq(BaseModel):
    round_num: int = Field(..., description="현재 라운드 번호")
    total_rounds: int = Field(..., description="전체 라운드 수")
    drawer: str = Field(..., description="그림을 그린 플레이어 이름")
    guesser: str = Field(..., description="그림을 맞출 플레이어 이름")

@router.post('/{game_id}/guess/start/', summary="추측 시작 시 묘묘의 도발 메시지")
async def start_guess(game_id: str, request: GuessStartReq = Body(...)):
    message = await myomyo.guess_start_message(game_id=game_id, round_num=request.round_num, total_rounds=request.total_rounds, drawer=request.drawer, guesser=request.guesser)
    return GPTResponse(message=message)

class MakeGuessReq(BaseModel):
    image_description: str = Field(..., description="그림에 대한 설명")

@router.post("/{game_id}/guess", summary="AI 정답 추론 API")
async def make_guess(game_id: str, request: MakeGuessReq = Body(...)):
    message = await myomyo.guess_message(
        game_id=game_id,
        image_description=request.image_description
    )
    return GPTResponse(message=message)

class GuessReactReq(BaseModel):
    is_correct: bool = Field(..., description="추측의 정답 여부")
    answer: str = Field(..., description="실제 정답")
    guesser: str = Field(default=None, description="추측한 플레이어")

@router.post("/{game_id}/guess/react", summary="예측 결과 반응 메시지 API")
async def react_to_guess(game_id: str, request: GuessReactReq = Body(...)):
    message = await myomyo.react_to_guess_message(
        game_id=game_id,
        is_correct=request.is_correct,
        guesser=request.guesser,
        answer=request.answer
    )
    return GPTResponse(message=message)

class GameEndReq(BaseModel):
    winner: str = Field(..., description="묘묘의 승리 여부")

@router.post("/{game_id}/end", summary="게임 종료 메시지 API")
async def end_game(game_id: str, request: GameEndReq = Body(...)):
    message = await myomyo.game_end_message(
        game_id=game_id,
        is_myomyo_win=request.winner == "AI"
    )
    myomyo.cleanup_game(game_id=game_id)
    return GPTResponse(message=message)