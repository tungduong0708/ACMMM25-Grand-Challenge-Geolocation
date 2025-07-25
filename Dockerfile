FROM python:3.12
WORKDIR /code

RUN apt-get update && apt-get install -y ffmpeg xvfb

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./src /code/src
RUN python /code/src/setup.py
RUN playwright install chrome
RUN playwright install-deps

COPY ./app.py /code/app.py
COPY ./entrypoint.sh /code/entrypoint.sh

ENTRYPOINT [ "/code/entrypoint.sh" ]