FROM python:3.10

RUN pip install --upgrade pip

WORKDIR /code

RUN pip install fastapi pillow tensorflow

COPY . /code/

CMD ["fastapi", "dev", "main.py", "--port", "80"]
