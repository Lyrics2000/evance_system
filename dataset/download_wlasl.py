import os
import sys
import logging
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path("ksl_project_data").resolve()
LOG_DIR = PROJECT_DIR / "logs"

PROJECT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / f"load_wlasl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)

import fiftyone as fo
import fiftyone.utils.huggingface as fouh

logger.info("Loading WLASL from Hugging Face/FiftyOne cache...")

dataset = fouh.load_from_hub("Voxel51/WLASL")
dataset.persistent = True

logger.info(f"Dataset loaded: {dataset.name}")
logger.info(f"Total samples in metadata: {len(dataset)}")

# Check how many video files actually exist locally
existing = 0
missing = 0
missing_examples = []

for sample in dataset:
    if sample.filepath and Path(sample.filepath).exists():
        existing += 1
    else:
        missing += 1
        if len(missing_examples) < 20:
            missing_examples.append(sample.filepath)

logger.info(f"Existing local video files: {existing}")
logger.info(f"Missing local video files: {missing}")

if missing_examples:
    logger.warning("Missing video examples:")
    for path in missing_examples:
        logger.warning(path)

logger.info("Launching FiftyOne app...")
session = fo.launch_app(dataset)
session.wait()