import os
from pathlib import Path
import dotenv


dotenv.load_dotenv('.env')
PROJECT_ROOT = Path(os.environ['PROJECT_ROOT'])

os.chdir(PROJECT_ROOT)