from typing import List

from fastapi import APIRouter, Body
from pydantic import BaseModel

from src.chat.myomyo import MyoMyoAI
import os

router = APIRouter(prefix="/chat")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

myomyo = MyoMyoAI(api_key=OPENAI_API_KEY)

# START_GAME
class GameStartReq(BaseModel):
    players: List[str]

class GameStartRes(BaseModel):
    game_id: str
    message: str

@router.post("/{game_id}/start", response_model=GameStartRes)
async def start_game(game_id: str, request: GameStartReq = Body(...)):
    message = await myomyo.game_start_message(game_id=game_id, players=request.players)
    return GameStartRes(game_id=game_id, message=message)

# START_ROUND
class RoundStartReq(BaseModel):
    drawing_player: str
    round_num: int
    total_rounds: int

class RoundStartRes(BaseModel):
    game_id: str
    message: str

@router.post("/{game_id}/round/start", response_model=RoundStartRes)
async def start_round(game_id: str, request: RoundStartReq = Body(...)):
    message = await myomyo.round_start_message(
        game_id=game_id,
        drawing_player=request.drawing_player,
        round_num=request.round_num,
        total_rounds=request.total_rounds
    )
    return RoundStartRes(game_id=game_id, message=message)

# MAKE_GUESS
class MakeGuessReq(BaseModel):
    image_description: str

class MakeGuessRes(BaseModel):
    game_id: str
    message: str

@router.post(
    "/{game_id}/guess",
    response_model=MakeGuessRes,
    description="묘묘의 정답 추론"
)
async def make_guess(game_id: str, request: MakeGuessReq = Body(...)):
    message = await myomyo.guess_message(
        game_id=game_id,
        image_description=request.image_description
    )
    return MakeGuessRes(game_id=game_id, message=message)


# GUESS_REACT
class GuessReactReq(BaseModel):
    game_id: str
    is_correct: bool
    answer: str
    guesser: str = None

class GuessReactRes(BaseModel):
    game_id: str
    message: str

@router.post(
    "/{game_id}/guess/react",
    response_model=GuessReactRes,
    description="예측 결과에 대한 묘묘의 반응"
)
async def guess_react(game_id: str, request: GuessReactReq = Body(...)):
    message = await myomyo.react_to_guess_message(
        game_id=game_id,
        is_correct=request.is_correct,
        guesser=request.guesser,
        answer=request.answer
    )

    return GuessReactRes(game_id=game_id, message=message)

# END_GAME
class EndGameReq(BaseModel):
    game_id: str
    is_myomyo_win: bool

class EndGameRes(BaseModel):
    game_id: str
    message: str

@router.post("/{game_id}/end", response_model=EndGameRes)
async def end_game(game_id: str, request: EndGameReq = Body(...)):
    message = await myomyo.game_end_message(
        game_id=game_id,
        is_myomyo_win=request.is_myomyo_win
    )
    myomyo.cleanup_game(game_id=game_id)
    return EndGameRes(game_id=game_id, message=message)