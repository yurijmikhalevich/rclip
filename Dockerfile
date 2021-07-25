FROM python:3.8-slim

WORKDIR /opt/rclip

RUN apt-get update && apt-get install -y \
  git \
  && rm -rf /var/lib/apt/lists/*

RUN pip install pipenv
COPY Pipfile* ./
RUN pipenv sync

RUN mkdir /data && mkdir /images
ENV DATADIR /data

COPY . .
CMD pipenv shell "cd /images && /opt/rclip/bin/rclip.sh \"${QUERY}\" && exit"
