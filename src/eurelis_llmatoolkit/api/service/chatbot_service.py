from fastapi import HTTPException

from eurelis_llmatoolkit.api.misc.console_manager import ConsoleManager
logger = ConsoleManager().get_output()

def hello():
    return {"message": "Hello"}
