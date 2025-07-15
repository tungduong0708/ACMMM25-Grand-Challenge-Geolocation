FROM python:3.12
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./g3 /code/g3
RUN python /code/g3/setup.py

COPY ./app.py /code/app.py

CMD ["fastapi", "run", "app.py", "--port", "80"]