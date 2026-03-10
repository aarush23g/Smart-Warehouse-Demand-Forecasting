# ====================================
# Smart Warehouse Demand Forecasting
# ====================================

PYTHON=python

install:
	pip install -r requirements.txt

features:
	$(PYTHON) -m pipelines.build_features

train:
	$(PYTHON) -m pipelines.train_model

inventory-plan:
	$(PYTHON) -m pipelines.inventory_policy

baseline-eval:
	$(PYTHON) -m src.evaluation.baseline_vs_model

simulate:
	$(PYTHON) -m src.evaluation.inventory_simulation

cost:
	$(PYTHON) -m src.evaluation.cost_model

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

dashboard:
	streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

test:
	pytest -q

format:
	black .

lint:
	ruff check .

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache