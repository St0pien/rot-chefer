
import wandb
import warnings
warnings.filterwarnings("ignore")
import coloredlogs
coloredlogs.install()

import logging
logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(pathname)s:%(lineno)d in ' \
           'function %(funcName)s] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG
)
# logger.setLevel(logging.DEBUG)
logging.getLogger('timm').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_wandb(name, project='PETL-ViT'):
    wandb.init(project=project, name=name)
    return wandb