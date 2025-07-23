FROM python:3.12
WORKDIR /code

RUN apt update && apt install -y ffmpeg

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./src /code/src
RUN python /code/src/setup.py
RUN playwright install chromium
RUN playwright install-deps

COPY ./app.py /code/app.py

ENTRYPOINT ["fastapi", "run", "app.py"]
CMD ["--port", "80"]