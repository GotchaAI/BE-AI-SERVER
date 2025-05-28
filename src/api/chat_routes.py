from typing import List

from fastapi import APIRouter, Body
from pydantic import BaseModel, Field

from src.chat.myomyo import MyoMyoAI
import os

router = APIRouter(prefix="/chat", tags=['Chat'])
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

myomyo = MyoMyoAI(api_key=OPENAI_API_KEY)

# START_GAME
class GameStartReq(BaseModel):
    players: List[str] = Field(..., description="게임에 참여할 플레이어 이름 List")



@router.post(
    "/{game_id}/start",
    summary="게임 시작 메시지 API",
    description="게임 시작에 따른 묘묘의 도발 메시지를 반환합니다.",
    responses={
        200:{
            "description": "성공",
                "content":{
                    "application/json" :{
                        "example" : {
                            "game_id": "1",
                            "message": "안녕하세여, 창모, 릴러말즈 친구들! 이번엔 묘묘가 총을 잡았다니까 맘 놓지 마! 내가 정확하게 그림을 맞추고 너희들을 제압해볼 건데, 준비 됐어? 꼭 즐겁게 놀자구~ ;)"
                        }
                    }
                }
        }
    }
)
async def start_game(game_id: str, request: GameStartReq = Body(..., example= { "players": [ "창모", "릴러말즈" ]})):
    message = await myomyo.game_start_message(game_id=game_id, players=request.players)
    return message

# START_ROUND
class RoundStartReq(BaseModel):
    roundNum: int = Field(..., description="현재 라운드(1~3)")
    totalRounds: int = Field(..., description="총 라운드 수(3)")


@router.post(
    path="/{game_id}/round/start",
    summary="라운드 시작 메시지 API",
    description="라운드 시작에 따른 묘묘의 도발 메시지를 반환합니다.",
    responses={
        200: {
            "description": "성공",
            "content": {
                "application/json": {
                    "example": {
                        "game_id": "1",
                        "message": "자, 이번에는 내가 예리한 눈썰미로 정답 맞출 차례니까, 신나게 그려봐! 😉🎨✨"
                    }
                }
            }
        }
    }
)
async def start_round(game_id: str, request: RoundStartReq = Body(..., example={
    "roundNum" : 1,
    "totalRounds" : 3
})):
    message = await myomyo.round_start_message(
        game_id=game_id,
        round_num=request.roundNum,
        total_rounds=request.totalRounds
    )
    return message



class RoundEndReq(BaseModel):
    roundNum: int
    totalRounds: int
    winner: str

@router.post(
    path="/{game_id}/round/end",
    summary = "라운드 종료 메시지 API",
    description="라운드 종료 및 결과에 따른 묘묘의 반응 메시지를 반환합니다."
)
async def round_end(game_id: str, request: RoundEndReq = Body):
    message = await myomyo.round_end_message(
        game_id = game_id,
        round_num = request.roundNum,
        total_rounds = request.totalRounds,
        is_myomyo_win= (request.winner == "AI")
    )
    return message






class GuessStartReq(BaseModel):
    roundNum: int
    totalRounds: int
    drawer: str
    guesser: str


@router.post(
    path = '/{game_id}/guess/start/',
    summary = "추측 시작 시 묘묘의 도발 메시지"
)
async def guess_start(game_id: str, request: GuessStartReq = Body(...,)):
    message = await myomyo.guess_start_message(game_id=game_id, round_num=request.roundNum, total_rounds=request.totalRounds, drawer=request.drawer, guesser = request.guesser)
    return message

# MAKE_GUESS
class MakeGuessReq(BaseModel):
    imageDescription: str = Field(..., description="그림에 대한 설명")

class MakeGuessRes(BaseModel):
    game_id: str
    message: str

@router.post(
    "/{game_id}/guess",
    summary="AI 정답 추론 API",
    description="그림에 대한 설명을 받아 해당 그림이 나타내는 정답을 추론하여 메시지로 반환합니다.",
    response_model=MakeGuessRes,
    responses={
        200: {
            "description": "성공",
            "content" : {
                "application/json" :{
                    "example": {
                        "game_id": "1",
                        "message": "노란 꽃에 바람을 불고 있는 한 남자? 우웅, 감이 와! '해바라기' 맞지? 내 추측이 맞다면 너에게 천재적 감각을 인정해줄게! 😉🌻✨"
                    }
                }
            }
        }
    }
)
async def make_guess(game_id: str, request: MakeGuessReq = Body(..., example={
    "image_description": "노란 꽃에 바람을 불고 있는 한 남자"
})):
    message = await myomyo.guess_message(
        game_id=game_id,
        image_description=request.imageDescription
    )
    return MakeGuessRes(game_id=game_id, message=message)


# GUESS_REACT
class GuessReactReq(BaseModel):
    is_correct: bool = Field(..., alias="isCorrect", description="추측의 정답 여부")
    answer: str = Field(..., description="실제 정답")
    guesser: str = Field(default=None, description="추측한 플레이어")

class GuessReactRes(BaseModel):
    game_id: str
    message: str

@router.post(
    "/{game_id}/guess/react",
    summary="예측 결과 반응 메시지 API",
    response_model=GuessReactRes,
    description="예측 결과에 대한 묘묘의 반응",
    responses={
        200: {
            "description" : "성공",
            "content" : {
                "application/json" : {
                    "example" : {
                        "game_id": "1",
                        "message": "민들레였어? 허허, 릴러말즈, 이번엔 잘 맞췄네. 하지만 다음엔 이길 거니까 기대해 봐! 😈"
                    }
                }
            }
        }
    }
)
async def guess_react(game_id: str, request: GuessReactReq = Body(..., example={
    "is_correct" : True,
    "answer" : "민들레",
    "guesser" : "릴러말즈"
})):
    message = await myomyo.react_to_guess_message(
        game_id=game_id,
        is_correct=request.is_correct,
        guesser=request.guesser,
        answer=request.answer
    )

    return GuessReactRes(game_id=game_id, message=message)

# END_GAME
class EndGameReq(BaseModel):
    winner: str = Field(..., description="묘묘의 승리 여부")


@router.post(
    path="/{game_id}/end",
    summary="게임 종료 메시지 API",
    description="게임 종료 로직 처리 및 결과에 대한 묘묘의 반응을 반환합니다.",
    responses={
        200: {
            "description" : "성공",
            "content" : {
                "application/json" : {
                    "example":{
                        "game_id": "1",
                        "message": "헉, 너네 둘이서 날 이기다니... 😒💔 근데 내가 질 줄 알았냐? 너무 신나지마, 다음엔 내가 이길거라구! 기다려봐~ 😏🔥"
                    }
                }
            }
        }
    })
async def end_game(game_id: str, request: EndGameReq = Body(...,)):
    message = await myomyo.game_end_message(
        game_id=game_id,
        is_myomyo_win=request.winner == "AI"
    )
    myomyo.cleanup_game(game_id=game_id)
    return message