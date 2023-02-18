FROM python:3.8-slim

WORKDIR /opt/rclip

RUN apt-get update && apt-get install -y \
  git \
  && rm -rf /var/lib/apt/lists/*

RUN pip install poetry==1.3.2
RUN poetry --version
COPY pyproject.toml poetry.lock ./
# install deps
RUN poetry install --without dev

RUN mkdir /data && mkdir /images
ENV DATADIR /data

COPY . .
# install rclip
RUN poetry install --without dev

CMD poetry run bash -c "cd /images && rclip \"${QUERY}\" && exit"
