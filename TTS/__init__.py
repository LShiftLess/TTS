import os
import dotenv
import wandb

dotenv.load_dotenv()

with open(os.path.join(os.path.dirname(__file__), "VERSION"), "r", encoding="utf-8") as f:
    version = f.read().strip()

wandb.login(key=os.getenv('WANDB_KEY'))

__version__ = version
