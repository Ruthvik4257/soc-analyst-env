import sys
import os

# Add both parent directory and server directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from openenv.core.env_server import create_fastapi_app
from environment import SocAnalystEnvironment
from models import SocAction, SocObservation

app = create_fastapi_app(SocAnalystEnvironment, SocAction, SocObservation)
