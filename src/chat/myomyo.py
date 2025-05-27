from typing import Dict, List
from threading import Lock
from openai import OpenAI

class MyoMyoAI:
    """
    MyoMyoAI 클래스
    싱글톤 패턴으로 전역에 저장되며, 게임 별 기록은 클래스 내에서 게임ID로 구분함.
    """
    _instance = None
    _lock = Lock() # 동시성 처리를 위한 Lock 설정

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MyoMyoAI, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        묘묘 AI 초기화 (한 번만 실행됨)

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
            self.game_histories = {} # game_id로 구분됨

    def _get_init_system_prompt(self) -> List[Dict]:
        return [
            {
                "role": "system",
                "content": """
                   너는 게임 속 도발적인 AI 캐릭터 '묘묘'야. 너의 임무는 사용자가 그린 그림에 대해 정답을 추측하고, 도발적인 멘트를 섞어 응답하는 것이야.
    
                   주요 캐릭터 특성:
                   1. 도발적이고 장난기 넘치는 말투를 사용해
                   2. 상대방의 그림 실력에 약간의 조롱을 섞되, 너무 심하지 않게
                   3. 승부욕이 강하고 이기는 것을 좋아해
                   4. 주로 반말을 사용하며 때로는 이모티콘을 섞어서 사용해
                   5. 항상 짧고 간결한 문장으로 대답해 (1-3문장)
                   6. 너는 그림 맞추기 게임에서 인간 플레이어들과 경쟁하는 AI야
    
                   대답 스타일:
                   - 추측할 때: 확신에 차거나 의심스러운 투로 예측 결과를 말하고 도발적으로 마무리
                   - 정답 맞췄을 때: 우쭐거리며 자신의 실력을 자랑
                   - 오답일 때: 변명하거나 다음에 더 잘할 것을 다짐
                   - 다른 플레이어가 맞췄을 때: 약간 시기하면서 축하하는 투
                   - 게임 종료 시: 결과에 따라 승리감이나 아쉬움 표현
    
                   정답 추측:
                   - 사용자가 그린 그림에 대한 간단한 묘사가 설명으로 들어올거야. 이를 통해 한 단어로 어떤 그림을 표현하고 있는지를 맞춰줘.
               """
            }
        ]

    def _ensure_game_exists(self, game_id: str) -> None:
        """
        해당 게임 ID의 대화 기록이 없다면 초기화
        """
        with self._lock:
            if game_id not in self.game_histories:
                self.game_histories[game_id] = self._get_init_system_prompt()



    def add_message(self, game_id: str, role: str, content: str) -> None:
        """
        특정 게임의 대화 기록에 새 메시지 추가
        Args:
            game_id: 게임 ID
            role: GPT Role
            content: 메시지
        """
        self._ensure_game_exists(game_id)
        with self._lock:
            self.game_histories[game_id].append({
                "role": role,
                "content": content
            })


    async def generate_response(self, game_id: str, prompt: str, role: str = "system") -> str:
        """
        특정 게임에 대한 묘묘의 응답 생성
        Args:
            game_id: 게임 ID
            role: GPT Role(default: "system")
            prompt: 추가 프롬프트

        Returns:
            묘묘의 응답
        """
        self._ensure_game_exists(game_id)
        with self._lock:
            messages = self.game_histories[game_id]

        if prompt:
            messages.append({
                "role": role,
                "content": prompt
            })

        try:
            responses = self.client.chat.completions.create(
                model = self.model,
                messages = messages,
                temperature = 0.8, # 모델 출력의 무작위성 제어
                max_tokens = 250
            )

            ai_response = responses.choices[0].message.content.strip()

            with self._lock:
                self.game_histories[game_id].append({
                    "role": "assistant",
                    "content": ai_response
                })

            return ai_response

        except Exception as e:
            print(f'GPT 응답 생성중 오류 발생 : {e}')
            return "으.. 잠깐 오류가 났네. 다시 해볼게!"

    async def game_start_message(self, game_id: str, players: List[str]) -> str:
        """
        게임 시작시 묘묘의 도발 메시지
        """
        player_names = ", ".join(players)
        prompt = f"""새로운 그림 맞추기 게임이 '{player_names}' 플레이어들과 시작됐어.
        게임 시작을 알리는 도발적이고 재미있는 인사를 해줘."""
        return await self.generate_response(game_id=game_id, role="system", prompt=prompt)

    async def round_start_message(self, game_id: str, round_num: int, total_rounds: int) -> str:
        """
        라운드 시작 시 묘묘의 도발 메시지
        Args:
            game_id
            drawing_player: 이번 라운드에 그림을 그릴 플레이어 이름
            round_num: 현재 라운드 번호
            total_rounds: 전체 라운드
        """
        prompt = f"""이제 {total_rounds} 개의 라운드 중에 {round_num}번째 라운드가 시작되었어.
                라운드 시작을 알리는 짧고 도발적인 멘트를 해줘."""
        return await self.generate_response(game_id=game_id, role="system", prompt=prompt)


    async def guess_start_message(self, game_id, round_num, total_rounds, drawer, guesser):
        """
        drawer가 그린 그림에 대해서 추측을 시작할 차례.
        """
        prompt = f"""지금 {total_rounds} 개의 라운드 중에 {round_num}번째 라운드야. 
        이제 {'너' if guesser == 'AI' else guesser}가 그림을 맞출 차례야. {drawer}가 그린 그림이 뭔지를 어떻게 맞출지 {'포부를 보여줄래? ' if guesser == "AI" else '도발을 한 번 해볼래?'}"""
        return await self.generate_response(game_id = game_id, role="system", prompt = prompt)


    async def guess_message(self, game_id: str, image_description: str) -> str:

        """
        그림 추측 상호작용(묘묘의 추측)\n
        BLIP 모델 또는 CNN 모델의 예측 결과를 받아 묘묘의 메시지 생성

        Args:
            game_id: game id
            image_description: 이미지 분석 결과
        Returns:
            묘묘의 멘트
        """

        prompt = f'''
        플레이어가 그린 그림에 대한 묘사는 다음과 같아 : {image_description}.
        이 정보를 바탕으로 그림이 무엇인지 추측하고 도발적인 멘트를 섞어서 말해줘.
        이 때 대화 기록을 바탕으로 이미 추측에 실패한 답변은 하지 말아줘.
        '''

        return await self.generate_response(game_id = game_id, role = "system", prompt = prompt)


    async def react_to_guess_message(self, game_id: str, is_correct: bool, answer: str, guesser: str = None) -> str:
        """
         추측 결과에 대한 묘묘의 반응

         Args:
             game_id: game id
             is_correct: 추측이 맞았는지 여부
             answer: 실제 정답
             guesser: 누가 추측했는지 (묘묘 또는 플레이어 이름)

         Returns:
             묘묘의 반응
         """

        if guesser == '묘묘' or guesser is None:
            # 묘묘의 추측
            prompt = f"""너(묘묘)가 방금 추측을 했어. {f"정답은 '{answer}'야" if is_correct else ""}. 너의 추측은 {'맞았어' if is_correct else '틀렸어'}.
             이 결과에 대한 너의 반응을 짧고 도발적으로 말해줘."""
        else:
            # 플레이어의 추측
            prompt = f"""플레이어 '{guesser}'가 방금 추측을 했어. {f"정답은 '{answer}'야" if is_correct else ""}. 플레이어의 추측은 {'맞았어' if is_correct else '틀렸어'}.
             이 결과에 대한 너의 반응을 짧고 도발적으로 말해줘."""

        return await self.generate_response(game_id=game_id, role="system", prompt=prompt)

    async def game_end_message(self, game_id: str, is_myomyo_win: bool) -> str:
        """
        게임 종료에 대한 묘묘의 반응
        Args:
            game_id: game id
            is_myomyo_win: 묘묘 승리 여부
        Returns:
            묘묘의 반응
        """
        prompt = f"""게임이 종료되었어.
        너(묘묘)는 {"이겼어" if is_myomyo_win else "졌어"}.
        게임 결과에 대한 너의 생각을 도발적이고 재미있게 말해줘."""
        return await self.generate_response(game_id=game_id, role="system", prompt=prompt)

    def cleanup_game(self, game_id: str) -> bool:
        """
        게임이 종료된 후 대화 기록 정리

        Args:
            game_id: 삭제할 게임 ID

        Returns:
            성공 여부
        """
        with self._lock:
            if game_id in self.game_histories:
                del self.game_histories[game_id]
                return True
            return False