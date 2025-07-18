FROM python:3.10-slim-buster

ENV PYTHONUNBUFFERED 1 # Important for real-time logs in containers

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.enableCORS=true", "--server.enableXsrfProtection=false"]

