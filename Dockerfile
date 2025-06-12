FROM python:3.10-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt



FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
COPY --from=builder /usr/local /usr/local

COPY src/ src/
COPY run.py config.py .

CMD ["python", "run.py"]