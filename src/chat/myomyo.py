from openai import OpenAI

class MyoMyoAI:
    """
    MyoMyoAI 클래스
    싱글톤 패턴으로 전역에 저장되며, 게임 별 기록은 클래스 내에서 게임ID로 구분함.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
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
        if self._initialized:
            return
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._initialized = True
        self.game_histories = {} # game_id로 구분됨

