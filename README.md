# Streaming transcription

This is a portfolio project powered by Gradio, allowing real-time transcription of streaming audio.

## Features
- Real-time speech transcription
- Gradio interface for easy interaction


# Usage

The system can be put online using Docker, Docker-compose or locally

## Docker Compose and Makefile

If Docker/Docker compose and GNU make has been sucessfully installed on the machine[^1]. The program can simply be run with

```
make up
```

### Docker Compose

```
docker compose down
docker compose build
docker compose up
```

## Run local

Python `3.10`. It is highly recommended to create a virtual environment for the project `python -m venv .venv` and activate it

```
cd app
pip install -r requirements.txt
python main.py
```
[^1]: GNU make can be installed on windows using [chocolatey][https://chocolatey.org/install]
