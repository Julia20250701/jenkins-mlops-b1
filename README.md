# Jenkins MLOps B1

A small MLOps demo project with:

- local development in a Python virtual environment
- model training
- model evaluation
- saved model artifact
- saved metrics artifact
- Jenkins pipeline
- archived artifacts in Jenkins

## Project structure

- `data/iris.csv` → dataset
- `src/train.py` → training script
- `src/evaluate.py` → evaluation script
- `models/` → trained model output
- `results/` → metrics and evaluation outputs
- `Jenkinsfile` → Jenkins pipeline definition

## Important note

Local setup uses `.venv`.

Jenkins runs the pipeline inside Docker, so the local `.venv` is not used by Jenkins.
