FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MEMBOX_HOST=0.0.0.0
ENV MEMBOX_PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY membox.py README.md ./
COPY membox_service.py README_LEADERBOARD.md ./

RUN mkdir -p /data/membox

EXPOSE 8080

CMD ["python", "membox_service.py"]
