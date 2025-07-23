FROM python:3.12
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./src /code/src
RUN python /code/src/setup.py
RUN playwright install chromium

COPY ./app.py /code/app.py

CMD ["fastapi", "run", "app.py", "--port", "80"]