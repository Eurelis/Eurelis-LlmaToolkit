from eurelis_llmatoolkit.api.routers.chatbot import chatbot_endpoints
# from eurelis_llmatoolkit.api.routers.version import version_endpoints
from eurelis_llmatoolkit.api.service.agent_manager import AgentManager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import datetime

def create_app():
    app = FastAPI()

    # TODO a TESTER CORS
    
    # Middleware CORS
    allowed_origins = AgentManager().get_allowed_origines()
    # allowed_origins = [
    #     "http://localhost",
    #     "http://localhost:8080",
    #     "https://example.com",
    # ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # TODO
    # Middleware Sentry
    # app.add_middleware(SentryAsgiMiddleware)

    @app.get("/")
    def hello():
        return "Hello, World!"

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/demo/{id_agent}")
    def demo(id_agent: str):
        return {"id_agent": id_agent, "timestamp": datetime.datetime.now().timestamp()}

    # app.include_router(version_endpoints.router, prefix="/api/version")
    app.include_router(chatbot_endpoints.router, prefix="/api/chat")

    return app
