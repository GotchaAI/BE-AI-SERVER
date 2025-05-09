from openai import OpenAI
import os
import logging


client = OpenAI(
    api_key = os.getenv('OPENAI_API_KEY')
)


def get_message(result):
    predicted_class = result['predicted_class']
    confidence = result['confidence']
    prompt = f"{predicted_class} 확률 {confidence}"
    logging.info(f'프롬프트 전송 : {prompt}')
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            { "role" : "system", "content" :
            f"""당신은 어떤 그림인지를 맞추는 게임을 하고 있습니다. 
            당신이 예측한 결과를 예측결과, 확률로 줄테니 최대한 
            { "재치있고 재미있게" if confidence >= 0.5 else "사용자를 놀리는 듯이"} 
            바꿔주세요."""},
            { "role" : "user", "content" : prompt}
        ]
    )
    logging.info('GPT 호출 완료')
    return completion.choices[0].message.content

