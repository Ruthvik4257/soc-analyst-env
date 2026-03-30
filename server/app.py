import sys
import os

# Add both parent directory and server directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from openenv.core.env_server import create_fastapi_app
from fastapi.responses import FileResponse
from environment import SocAnalystEnvironment
from models import SocAction, SocObservation

app = create_fastapi_app(SocAnalystEnvironment, SocAction, SocObservation)

@app.get("/")
def read_root():
    frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "index.html")
    return FileResponse(frontend_path)

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)

if __name__ == '__main__':
    main()


