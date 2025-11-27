FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel

WORKDIR /usr/local/app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .



CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]