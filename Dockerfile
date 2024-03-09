FROM python:3.11.8-bookworm

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY visage_vision visage_vision
# COPY setup.py setup.py
# RUN pip install .

CMD uvicorn  visage_vision.api.main:app --host 0.0.0.0 --port 8000
