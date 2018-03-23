import os
import logging
from PASDAC.Evaluation.crossvalidation import run_validation
from PASDAC.Tools import load_data
from PASDAC import SETTINGS


logFormatter = logging.Formatter("%(message)s")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler(os.path.join(SETTINGS['PATH_OUTPUT'], "evaluation.log"))
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

logger.info('run')
dataAll = load_data()
logger.info('runall')
run_validation(dataAll)
