import os

DATASET_DIR = os.getenv('BENCHMARK_DATASET_DIR', os.path.join(os.path.dirname(__file__), 'datasets'))
BATCH_SIZE = int(os.getenv('BENCHMARK_BATCH_SIZE', 256))
