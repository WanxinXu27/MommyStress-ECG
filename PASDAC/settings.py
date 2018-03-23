import yaml
import os
import logging

logger = logging.getLogger(__name__)

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

setting_path = os.path.join(__location__, 'config.yaml')

with open(setting_path) as f:
    logger.info("Loading setting file %s", setting_path)

    SETTINGS = yaml.load(f)

