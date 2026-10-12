# Iris Neural Network — MLOps Project

An end-to-end machine learning project that predicts Iris flower species using a TensorFlow neural network, with experiment tracking, data and pipeline versioning, API deployment, containerization, CI tests, and monitoring.

## Features

- Neural network training with TensorFlow/Keras
- Data preprocessing with scikit-learn
- Experiment tracking with MLflow
- Data and pipeline versioning with DVC
- FastAPI prediction service
- Streamlit user interface
- Docker containerization
- GitHub Actions testing and image publishing
- Prometheus metrics monitoring

## Project Structure

- `neuralnetwork.py` — model training and evaluation
- `app.py` — FastAPI prediction service
- `frontend.py` — Streamlit interface
- `data/` — Iris dataset and DVC metadata
- `models/` — generated model artifacts
- `tests/` — automated tests
- `dvc.yaml` — DVC training pipeline
- `dvc.lock` — recorded pipeline dependencies and outputs
- `prometheus.yml` — Prometheus scrape configuration
- `Dockerfile` — API container configuration

## Requirements

- Python 3.10
- Git
- DVC
- Docker (optional, for container execution)

## Local Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train the Model

Run the DVC pipeline:

```powershell
dvc repro
```

DVC skips stages whose dependencies and outputs are unchanged.

To run training directly:

```powershell
python neuralnetwork.py
```

Training evaluates the model and saves the neural network, scaler, and label encoder under `models/`. MLflow records the experiment parameters and metrics.

## Run the API

```powershell
python -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

Useful endpoints:

- `/` — API root
- `/health` — health check
- `/predict` — flower prediction
- `/metrics` — Prometheus metrics

## Run the Streamlit Frontend

Configure `IRIS_API_URL` to point to the API URL, then run:

```powershell
streamlit run frontend.py
```

For local API usage, set `IRIS_API_URL` to `http://127.0.0.1:8000`.

## MLflow

Start the MLflow interface with:

```powershell
mlflow ui
```

Open `http://127.0.0.1:5000` to inspect experiment runs.

## DVC

Check pipeline status:

```powershell
dvc status
```

Reproduce the pipeline:

```powershell
dvc repro
```

A configured DVC remote is needed to share dataset contents and other DVC-managed files between machines.

## Docker

Build the API image:

```powershell
docker build -t iris-neural-network-api .
```

Run it:

```powershell
docker run --rm -p 8000:8000 iris-neural-network-api
```

## Prometheus

Start the API locally on port 8000 and configure Prometheus using `prometheus.yml`. If running Prometheus in Docker, the target `host.docker.internal:8000` allows it to scrape an API running on the Windows host.

Open `http://localhost:9090` to query metrics such as `iris_http_requests_total`.

## Tests

Run automated tests with:

```powershell
pytest
```

## Evaluation

In one successful training run, the model achieved **95.65% test accuracy** on a test split of 23 samples. Results can vary if the data, model, or training configuration changes.

## Notes

- Generated model files are managed by DVC rather than committed directly to Git.
- Configure DVC remote storage before expecting another machine to retrieve DVC-managed data and model artifacts.
- Never commit credentials, access tokens, or secret configuration values.
