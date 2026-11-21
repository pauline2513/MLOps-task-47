FROM python:3.14-slim

WORKDIR /app

RUN pip install --no-cache-dir scikit-learn==1.9.1

COPY model.pkl /app/model.pkl
COPY predict.py /app/predict.py

ENTRYPOINT ["python", "/app/predict.py"]
