.PHONY: setup data test train tune gate api docker-build docker-run pipeline clean

setup:
	pip install -r requirements.txt

data:
	python data/generate_data.py --out data/churn.csv --rows 8000

test:
	pytest tests/ -v

train:
	cd src && python train.py --data ../data/churn.csv --register

tune:
	cd src && python tune.py --data ../data/churn.csv --trials 20 --register

gate:
	cd src && python evaluate.py --metrics artifacts/metrics.json --min-f1 0.70 --min-auc 0.75

pipeline:
	python orchestration/pipeline.py --register

api:
	cd api && uvicorn main:app --reload --port 8000

docker-build:
	docker build -t mlops-lab-api:local .

docker-run:
	docker run -p 8000:8000 mlops-lab-api:local

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf src/mlruns src/mlflow.db src/artifacts src/confusion_matrix.png
