FROM python:3.11.8-bookworm

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY apimodel.py apimodel.py
COPY setup.py setup.py
RUN pip install .
CMD uvicorn  api:app --host 0.0.0.0 --port $PORT
