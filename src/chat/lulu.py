from openai import OpenAI
from threading import Lock
from typing import Dict, List
import json
import random


class LuLuAI:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LuLuAI, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, api_key: str, model: str = "gpt-4.1"):
        """
        LuLu AI 초기화 (한 번만 실행됨)

        Args:
            api_key: OpenAI API 키
            model: 사용할 GPT 모델 (기본값: gpt-4)
        """
        with self._lock:
            if self._initialized:
                return
            self.client = OpenAI(api_key=api_key)
            self.model = model
            self._initialized = True
            self.active_games = {}  # gameId별 현재 task만 저장
            self.global_used_keywords = []  # 전역 사용된 키워드 저장 (최대 30개)

    def create_game(self) -> str:
        """
        새 게임 시작 및 4자리 gameId 발급

        Returns:
            str: 생성된 4자리 gameId
        """
        # 중복되지 않는 4자리 숫자 생성
        while True:
            game_id = f"{random.randint(1000, 9999)}"
            if game_id not in self.active_games:
                break

        self.active_games[game_id] = None  # 아직 task 생성 안됨
        return game_id

    def _update_global_keywords(self, new_keyword: str):
        """
        전역 키워드 목록 업데이트 (최대 30개 유지)

        Args:
            new_keyword: 새로 추가할 키워드
        """
        if new_keyword not in self.global_used_keywords:
            self.global_used_keywords.append(new_keyword)
            # 30개를 초과하면 가장 오래된 것부터 제거
            if len(self.global_used_keywords) > 30:
                self.global_used_keywords.pop(0)

    def flush_game_data(self, game_id: str):
        """
        특정 게임 ID의 데이터를 삭제

        Args:
            game_id: 삭제할 게임 ID

        Returns:
            bool: 삭제 성공 여부
        """
        if game_id in self.active_games:
            del self.active_games[game_id]


    def generate_drawing_task(self, game_id: str) -> Dict:
        """
        요청 단계: AI가 추상적이고 시적인 표현으로 그림 과제 제시

        Args:
            game_id: 게임 ID

        Returns:
            Dict: {"keyword": str, "situation": str, "game_id": str}
        """
        if game_id not in self.active_games:
            raise ValueError("Invalid game ID")


        system_prompt = f"""
        너는 꿈과 환상을 다루는 신비로운 이야기꾼이야. 
        사용자에게 그림을 그리게 하고 싶은데, 직접적으로 말하지 말고 매우 추상적이고 시적으로 표현해줘.

        규칙:
        - 핵심 키워드(명사)를 정하되, 절대 그 단어를 직접 언급하지 마
        - 해석의 여지가 많도록 추상적으로
        
        {f"이미 사용된 키워드들 (절대 사용하지 마): {', '.join(self.global_used_keywords)}" if self.global_used_keywords else ""}

        다양한 주제를 다뤄줘 (자연, 감정, 사물, 추상 개념, 동물, 건물, 음식, 계절, 색깔, 직업 등).

        출력은 반드시 JSON 형식으로:
        {{"keyword": "숨겨진 키워드", "situation": "시적이고 추상적인 묘사"}}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "새로운 그림 주제를 시적으로 표현해줘."}
                ],
                temperature=1.0,
                max_tokens=2048,
                top_p=1.0
            )

            # JSON 파싱
            content = response.choices[0].message.content.strip()
            print(content)

            task_data = json.loads(content)
            task_data["game_id"] = game_id

            self.active_games[game_id] = task_data
            return task_data

        except Exception as e:
            print(f"Error generating task: {e}")
            # 기본값 반환
            fallback_task = {
                "keyword": "달",
                "situation": "밤이 깊어질 때, 하늘의 은밀한 친구가 창문 너머로 속삭이고 있어. 그 둥근 미소가 어둠 속에서 혼자 빛나고 있는데, 왜인지 모르게 마음이 차분해져. 그 장면, 나한테 다시 보여줄 수 있을까?",
                "game_id": game_id
            }
            return fallback_task

    def evaluate_drawing(self, game_id: str, drawing_description: str) -> Dict:
        """
        평가 단계: AI가 사용자의 그림을 숨겨진 키워드와 비교하여 평가

        Args:
            game_id: 게임 ID
            drawing_description: 사용자가 그린 그림의 텍스트 설명

        Returns:
            Dict: {"score": int, "feedback": str, "task": Dict}
        """
        if game_id not in self.active_games:
            raise ValueError("Invalid game ID")

        current_task = self.active_games[game_id]

        # 가장 최근 과제 가져오기
        if current_task is None:
            raise ValueError("No task found for this game.")


        system_prompt = f"""
        너는 루루, 미대 입시를 담당하는 깐깐하고 까칠한 평가관이야. 
        예술에 대한 기준이 높고, 싸가지 없이 직설적으로 말하는 스타일이야.

        숨겨진 정답 키워드: {current_task['keyword']}
        원본 시적 묘사: {current_task['situation']}

        평가 기준:
        - 숨겨진 키워드를 제대로 파악했는가?
        - 예술적 표현력과 창의성은?
        - 전체적인 완성도와 기법은?

        루루의 말투 특징:
        - 직설적이고 신랄함
        - 인정할 때는 인정해주는 편
        - 미대생들한테 하는 것처럼 전문적이고 차가운 톤

        0-100점 사이로 평가해. 하지만 80점을 웬만하면 넘지 않도록 해줘. 숨겨진 키워드를 그림 안에 담았다면 30점 이상을 주고, 담지 못했다면 30점 이하를 주도록 해.
        
        30점 이상이 합격이야.

        출력 형식 (JSON):
        {{
            "score": 총점(0-100),
            "feedback": "루루의 깐깐하고 직설적인 피드백 (한국어)"
        }}
        """

        user_prompt = f"""
        다음은 사용자의 그림을 설명하는 문장이야 : "{drawing_description}"

        이 문장을 보고 어떤 그림일지를 생각해보고, 이 그림을 평가해줘. 
       
        그림을 설명하는 문장에 대한 언급은 하지 말아줘.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=300,
                top_p=1.00
            )

            content = response.choices[0].message.content.strip()
            evaluation = json.loads(content)
            evaluation["task"] = current_task
            evaluation["game_id"] = game_id

            return evaluation

        except Exception as e:
            print(f"Error evaluating drawing: {e}")
            # 기본 평가 반환
            fallback_evaluation = {
                "score": 35,
                "feedback": "하... 평가 시스템에 오류가 생겼는데 그것도 모르고 그림만 그리고 있었나? 기본기부터 다시 해.",
                "task": current_task,
                "game_id": game_id
            }
            return fallback_evaluation
