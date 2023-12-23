FROM python:3.10-slim

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update

# packages for working on tesseract
RUN apt-get -y install ffmpeg libsm6 libxext6

# install tesseract models (eng-est-rus)

COPY ./requirements.txt /tmp/requirements.txt

RUN pip3 install --no-cache-dir --upgrade  -r /tmp/requirements.txt

WORKDIR /app
ADD . /app

CMD ["python3" ,"main.py"]