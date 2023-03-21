FROM python:3.9.7-slim-bullseye

RUN apt-get -qq update && apt-get -qq -y install gcc

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY . /app

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ENTRYPOINT ["python","run_ticket_generator.py"]
