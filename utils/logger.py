import logging
import yaml
from pathlib import Path

config_path = Path(__file__).parent.parent / "config.yaml"
config = yaml.safe_load(open(config_path))

logging.basicConfig(
    level=config['logging']['mutation_level'],
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config['logging']['log_path']),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
