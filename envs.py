import os
import logging


logger = logging.getLogger(__name__)


PROJECT_FOLDER = os.path.dirname(__file__)
HOME_DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
HOME_OUTPUT_FOLDER = os.path.join(HOME_DATA_FOLDER, 'outputs/KD')
PREDICTION_FOLDER = os.path.join(HOME_DATA_FOLDER, 'outputs/predictions')
