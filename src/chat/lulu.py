from openai import OpenAI
from threading import Lock
from typing import Dict, List
import json
import uuid


class LuLuAI:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LuLuAI, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, api_key: str, model: str = "gpt-4"):
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
        새 게임 시작 및 gameId 발급

        Returns:
            str: 생성된 gameId
        """
        game_id = str(uuid.uuid4())
        self.game_contexts[game_id] = {
            "tasks": [],  # 생성된 과제들
            "evaluations": [],  # 평가 결과들
            "created_at": None
        }
        return game_id

    def generate_drawing_task(self, game_id: str) -> Dict:
        """
        요청 단계: AI가 키워드와 상황을 생성하여 그림 과제 제시

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
        너는 창의적인 AI 게임 마스터야. 사용자에게 그림을 그리도록 요청할 키워드와 상황을 제시해줘.

        요구사항:
        - 명사(키워드) + 구체적인 상황으로 구성
        - 그리기에 적절한 중간 수준의 난이도

        {"이전에 사용한 키워드들: " + ", ".join(previous_keywords) + " (이 키워드들은 피해줘)" if previous_keywords else ""}

        출력은 반드시 JSON 형식으로 해줘:
        {{"keyword": "명사", "situation": "구체적인 상황 설명"}}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "새로운 그림 주제를 생성해줘."}
                ],
                temperature=0.8,
                max_tokens=200
            )

            # JSON 파싱
            content = response.choices[0].message.content.strip()
            task_data = json.loads(content)
            task_data["game_id"] = game_id

            # 컨텍스트에 저장
            context["tasks"].append(task_data)

            return task_data

        except Exception as e:
            print(f"Error generating task: {e}")
            # 기본값 반환
            fallback_task = {
                "keyword": "고양이",
                "situation": "고양이가 나무 위에서 자고 있는 모습",
                "game_id": game_id
            }
            context["tasks"].append(fallback_task)
            return fallback_task

    def evaluate_drawing(self, game_id: str, drawing_description: str) -> Dict:
        """
        평가 단계: AI가 사용자의 그림을 의도한 그림과 비교하여 평가

        Args:
            game_id: 게임 ID
            drawing_description: 사용자가 그린 그림의 텍스트 설명

        Returns:
            Dict: {"score": int, "feedback": str, "keyword_match": int, "situation_match": int, "creativity": int, "task": Dict}
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
        너는 공정한 그림 게임 심사위원이야. 

        원본 과제:
        - 키워드: {latest_task['keyword']}
        - 상황: {latest_task['situation']}

        평가 기준:
        1. 키워드 일치도 (0-40점): 핵심 키워드가 그림에 포함되어 있는가?
        2. 상황 표현도 (0-40점): 주어진 상황이 잘 표현되어 있는가?
        3. 창의성 (0-20점): 독창적이고 흥미로운 표현인가?

        {evaluation_context}

        출력 형식 (JSON):
        {{
            "score": 총점(0-100),
            "keyword_match": 키워드_점수(0-40),
            "situation_match": 상황_점수(0-40),
            "creativity": 창의성_점수(0-20),
            "feedback": "구체적인 피드백 메시지 (한국어)"
        }}
        """

        user_prompt = f"""
        사용자의 그림 설명: "{drawing_description}"

        위 그림을 원본 과제와 비교하여 평가해줘.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
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
                "score": 50,
                "keyword_match": 20,
                "situation_match": 20,
                "creativity": 10,
                "feedback": "평가 중 오류가 발생했습니다.",
                "task": latest_task,
                "game_id": game_id
            }
            context["evaluations"].append(fallback_evaluation)
            return fallback_evaluation

    def generate_and_evaluate(self, game_id: str, drawing_description: str) -> Dict:
        """
        과제 생성과 평가를 한 번에 처리하는 편의 메서드

        Args:
            game_id: 게임 ID
            drawing_description: 사용자가 그린 그림의 텍스트 설명

        Returns:
            Dict: {"task": Dict, "evaluation": Dict}
        """
        # 1. 과제 생성
        task = self.generate_drawing_task(game_id)

        # 2. 평가 수행
        evaluation = self.evaluate_drawing(game_id, drawing_description)

        return {
            "task": task,
            "evaluation": evaluation
        }