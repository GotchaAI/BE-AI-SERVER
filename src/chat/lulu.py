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
            self.game_contexts = {}  # gameId별 컨텍스트 저장

    def create_game(self) -> str:
        """
        새 게임 시작 및 4자리 gameId 발급

        Returns:
            str: 생성된 4자리 gameId
        """
        # 중복되지 않는 4자리 숫자 생성
        while True:
            game_id = f"{random.randint(1000, 9999)}"
            if game_id not in self.game_contexts:
                break

        self.game_contexts[game_id] = {
            "tasks": [],  # 생성된 과제들
            "evaluations": [],  # 평가 결과들
            "created_at": None
        }
        return game_id

    def generate_drawing_task(self, game_id: str) -> Dict:
        """
        요청 단계: AI가 추상적이고 시적인 표현으로 그림 과제 제시

        Args:
            game_id: 게임 ID

        Returns:
            Dict: {"keyword": str, "situation": str, "game_id": str}
        """
        if game_id not in self.game_contexts:
            raise ValueError("Invalid game ID")

        context = self.game_contexts[game_id]

        # 이전 과제들을 참고하여 중복 방지
        previous_tasks = context["tasks"]
        previous_keywords = [task["keyword"] for task in previous_tasks]

        system_prompt = f"""
        너는 꿈과 환상을 다루는 신비로운 이야기꾼이야. 
        사용자에게 그림을 그리게 하고 싶은데, 직접적으로 말하지 말고 매우 추상적이고 시적으로 표현해줘.

        규칙:
        - 핵심 키워드(명사)를 정하되, 절대 그 단어를 직접 언급하지 마
        - 감정적이고 모호한 표현 사용
        - 마치 꿈에서 본 장면을 애매하게 묘사하는 느낌
        - "어둠이 숨을 죽이고 있을 때..." 같은 스타일
        - 해석의 여지가 많도록 추상적으로

        {"이전에 사용한 키워드들: " + ", ".join(previous_keywords) + " (이 키워드들은 피해줘)" if previous_keywords else ""}

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
                temperature=0.9,
                max_tokens=250
            )

            # JSON 파싱
            content = response.choices[0].message.content.strip()
            print(content)

            task_data = json.loads(content)
            task_data["game_id"] = game_id

            # 컨텍스트에 저장
            context["tasks"].append(task_data)

            return task_data

        except Exception as e:
            print(f"Error generating task: {e}")
            # 기본값 반환
            fallback_task = {
                "keyword": "달",
                "situation": "밤이 깊어질 때, 하늘의 은밀한 친구가 창문 너머로 속삭이고 있어. 그 둥근 미소가 어둠 속에서 혼자 빛나고 있는데, 왜인지 모르게 마음이 차분해져. 그 장면, 나한테 다시 보여줄 수 있을까?",
                "game_id": game_id
            }
            context["tasks"].append(fallback_task)
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
        if game_id not in self.game_contexts:
            raise ValueError("Invalid game ID")

        context = self.game_contexts[game_id]

        # 가장 최근 과제 가져오기
        if not context["tasks"]:
            raise ValueError("No task found for this game")

        latest_task = context["tasks"][-1]

        # 이전 평가 결과들을 참고하여 일관성 있는 평가
        previous_evaluations = context["evaluations"]
        evaluation_context = ""
        if previous_evaluations:
            avg_score = sum(eval["score"] for eval in previous_evaluations) / len(previous_evaluations)
            evaluation_context = f"\n이전 평가들의 평균 점수: {avg_score:.1f}점 (일관성 있는 평가 기준 유지)"

        system_prompt = f"""
        너는 루루, 미대 입시를 담당하는 깐깐하고 까칠한 평가관이야. 
        예술에 대한 기준이 높고, 싸가지 없이 직설적으로 말하는 스타일이야.

        숨겨진 정답 키워드: {latest_task['keyword']}
        원본 시적 묘사: {latest_task['situation']}

        평가 기준:
        - 숨겨진 키워드를 제대로 파악했는가?
        - 시적 묘사의 본질을 이해했는가?
        - 예술적 표현력과 창의성은?
        - 전체적인 완성도와 기법은?

        {evaluation_context}

        루루의 말투 특징:
        - 직설적이고 신랄함
        - 가끔 인정할 때도 있지만 쉽게 칭찬 안 함
        - 미대생들한테 하는 것처럼 전문적이고 차가운 톤

        0-100점 사이로 평가하되, 웬만해서는 80점 이상 주지 마.

        출력 형식 (JSON):
        {{
            "score": 총점(0-100),
            "feedback": "루루의 깐깐하고 직설적인 피드백 (한국어)"
        }}
        """

        user_prompt = f"""
        사용자의 그림 설명: "{drawing_description}"

        위 그림을 평가해줘.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                max_tokens=300
            )

            content = response.choices[0].message.content.strip()
            evaluation = json.loads(content)
            evaluation["task"] = latest_task
            evaluation["game_id"] = game_id

            # 컨텍스트에 저장
            context["evaluations"].append(evaluation)

            return evaluation

        except Exception as e:
            print(f"Error evaluating drawing: {e}")
            # 기본 평가 반환
            fallback_evaluation = {
                "score": 35,
                "feedback": "하... 평가 시스템에 오류가 생겼는데 그것도 모르고 그림만 그리고 있었나? 기본기부터 다시 해.",
                "task": latest_task,
                "game_id": game_id
            }
            context["evaluations"].append(fallback_evaluation)
            return fallback_evaluation
