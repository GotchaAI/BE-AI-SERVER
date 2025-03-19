FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


COPY src/ src/
COPY run.py config.py .

CMD ["python", "run.py"]
