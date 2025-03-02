init:
	conda env create -f environment.yml

test:
	pytest tests/ -v || echo "No tests found"

profile:
	python -m memory_profiler core/algebra/hopf_operator.py

clean:
	find . -name "*.pyc" -exec rm -f {} \;
	rm -rf __pycache__
