FROM python:3.10.6-buster

WORKDIR /app
COPY . /app
COPY requirementsHL.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080

COPY Makefile Makefile
RUN make reset_local_files

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
