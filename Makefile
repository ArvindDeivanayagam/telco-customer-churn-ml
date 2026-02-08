.PHONY: setup mlflow api ui run stop

setup:
	python -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip && pip install -r requirements.txt

mlflow:
	. .venv/bin/activate && mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000

api:
	. .venv/bin/activate && uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

ui:
	. .venv/bin/activate && python app/ui_gradio.py

run:
	@echo "Starting MLflow (5000), FastAPI (8000), Gradio (7860)..."
	@make -j 3 mlflow api ui

stop:
	@echo "Stopping services..."
	@pkill -f "mlflow ui" || true
	@pkill -f "uvicorn app.main:app" || true
	@pkill -f "app/ui_gradio.py" || true
