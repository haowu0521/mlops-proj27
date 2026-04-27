FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY refresher.py ./refresher.py

CMD ["python", "refresher.py"]
