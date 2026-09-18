FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY ecg_ai/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ecg_ai/ /app/ecg_ai/

WORKDIR /app/ecg_ai

EXPOSE 7860

CMD ["python", "dashboard/app.py"]
