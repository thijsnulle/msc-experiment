FROM python:3.11

WORKDIR /msc-experiment

RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean

COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY classes.py .
COPY data_processor.py .
COPY experiment.py .
COPY llm.py .
COPY data ./data
COPY models ./models
COPY pyenergibridge_config.json .

RUN git clone -b add-subprocess-support-from-within-docker-container https://github.com/thijsnulle/pyEnergiBridge.git && \
    cd pyEnergiBridge && \
    pip install .

CMD ["python3", "main.py"]

