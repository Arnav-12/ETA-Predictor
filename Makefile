.PHONY: install generate-data train evaluate serve notebook clean

install:
	pip install -r requirements.txt

generate-data:
	python data/generate_data.py

train:
	python src/train.py

evaluate:
	python src/error_analysis.py
	python src/drift_monitor.py

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

notebook:
	jupyter notebook notebooks/EDA_and_Modelling.ipynb

clean:
	rm -f data/raw/delivery_data.csv
	rm -f data/processed/*.pkl data/processed/*.json
	rm -f models/best_model.pkl
	rm -f reports/*.png reports/*.html
	rm -rf mlruns/
