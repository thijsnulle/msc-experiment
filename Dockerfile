FROM python:3.11

WORKDIR /msc-experiment

RUN apt-get update && apt-get install -y git && apt-get clean

RUN git clone https://github.com/luiscruz/pyEnergiBridge.git && \
    cd pyEnergiBridge && \
    pip install .

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY classes.py .
COPY data_processor.py .
COPY experiment.py .
COPY llm.py .
COPY data ./data
COPY models ./models

CMD ["python", "main.py"]

