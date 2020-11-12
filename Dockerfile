FROM tensorflow/tensorflow:2.3.1

WORKDIR /app

COPY requirements.txt server.py ./
COPY saved_models/triplet/efficientnet_v1.1/ /app/saved_models/triplet/efficientnet_v1.1

RUN apt-get update && apt-get install -y libsndfile1 libav-tools ffmpeg

ENV PORT=8080
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
CMD  ["python", "server.py"]

EXPOSE $PORT