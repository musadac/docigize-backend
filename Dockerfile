FROM python:3.8-slim-buster

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git


COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

CMD gunicorn --bind 0.0.0.0:5000 app:app
