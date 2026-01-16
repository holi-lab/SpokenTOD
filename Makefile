# Voice Dataset Augmentation Pipeline
# Conda Environment: disfluenTOD

CONDA_ENV = disfluenTOD
CONDA_RUN = conda run -n $(CONDA_ENV) --no-capture-output
PYTHON = $(CONDA_RUN) python
MODEL = gpt-4.1-mini

# Directories
SRC_DIR = src
OUT_DIR = data
DATA_DIR = datasets

# Datasets
DATASETS = emowoz,sgd,abcd,spokenwoz,tm2

# Evaluation
SAMPLE_SIZE = 5
WORKERS = 10
EMOTION_SPLIT = train

.PHONY: help augment augment-sample test lint clean

help:
	@echo "Voice Dataset Augmentation Pipeline"
	@echo ""
	@echo "Usage:"
	@echo "  make augment          - Run full augmentation pipeline"
	@echo "  make augment-sample   - Run on small sample (10 per dataset)"
	@echo "  make augment-emowoz   - Augment EmoWOZ only"
	@echo "  make augment-sgd      - Augment SGD only"
	@echo "  make augment-abcd     - Augment ABCD only"
	@echo "  make augment-spokenwoz- Augment SpokenWOZ only"
	@echo "  make augment-tm2      - Augment TM-2 only"
	@echo "  make test             - Run unit tests"
	@echo "  make lint             - Run linter"
	@echo "  make clean            - Remove generated files"
	@echo "  make eval-emotion     - Evaluate emotion classification on EmoWOZ (set EMOTION_SPLIT=train|valid|test)"

# Full pipeline
augment:
	$(PYTHON) $(SRC_DIR)/augment.py \
		--datasets $(DATASETS) \
		--output-dir $(OUT_DIR) \
		--batch-size 100

# Sample run for testing
augment-sample:
	$(PYTHON) $(SRC_DIR)/augment.py \
		--datasets $(DATASETS) \
		--output-dir $(OUT_DIR)/sample \
		--sample-size 10

# Individual datasets
augment-emowoz:
	$(PYTHON) $(SRC_DIR)/augment.py --datasets emowoz --output-dir $(OUT_DIR)

augment-sgd:
	$(PYTHON) $(SRC_DIR)/augment.py --datasets sgd --output-dir $(OUT_DIR)

augment-abcd:
	$(PYTHON) $(SRC_DIR)/augment.py --datasets abcd --output-dir $(OUT_DIR)

augment-spokenwoz:
	$(PYTHON) $(SRC_DIR)/augment.py --datasets spokenwoz --output-dir $(OUT_DIR)

augment-tm2:
	$(PYTHON) $(SRC_DIR)/augment.py --datasets tm2 --output-dir $(OUT_DIR)

# Evaluation
eval-emotion:
	$(PYTHON) $(SRC_DIR)/evaluate_emotions.py --sample-size ${SAMPLE_SIZE} --workers ${WORKERS} --model ${MODEL} --split ${EMOTION_SPLIT}

# Viewer
viewer:
	$(CONDA_RUN) streamlit run viewer.py

# Testing
test:
	$(CONDA_RUN) pytest tests/augmentation/ -v

lint:
	$(CONDA_RUN) ruff check $(SRC_DIR)/augmentation/

# Cleanup
clean:
	rm -rf $(OUT_DIR)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
