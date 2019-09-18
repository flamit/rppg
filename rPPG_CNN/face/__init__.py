import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('mssskin')

PACKAGE_PATH = Path(__file__).parent.resolve()
MODEL_PATH =  Path(os.getenv('MODEL_PATH', default=str(PACKAGE_PATH / 'model')))

logger.info(f'MODEL PATH: {MODEL_PATH}')

DETECTION_MANUAL = 'manual'
DETECTION_FOREHEAD = 'forehead'
DETECTION_SKIN = 'skin'
