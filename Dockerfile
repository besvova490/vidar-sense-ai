FROM python:3.9.6

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

EXPOSE 3000

CMD ["python", "bootstrap.py"]
