## Python 3.11
FROM python:3.11 

## Set working directory
WORKDIR /app

## Install dependencies
COPY requirements.txt .

## Takes about an hour to run
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

## Copy scripts
COPY gui.py .
COPY main.css .

CMD ["python", "gui.py"]