from tools.web import register as register_web
from tools.datetime_tool import register as register_datetime
from tools.python_runner import register as register_python_runner
from tools.store_tool import register as register_store
from tools.status_tool import register as register_status
from tools.strong_llm import register as register_strong_llm
from tools.image_gen import register as register_image_gen
from tool_definitions import TOOL_DEFINITIONS, TOOL_MAP

def register_tools():
    print("Registering tools...")
    register_web(TOOL_DEFINITIONS, TOOL_MAP)
    register_datetime(TOOL_DEFINITIONS, TOOL_MAP)
    register_python_runner(TOOL_DEFINITIONS, TOOL_MAP)
    register_store(TOOL_DEFINITIONS, TOOL_MAP)
    register_status(TOOL_DEFINITIONS, TOOL_MAP)
    register_strong_llm(TOOL_DEFINITIONS, TOOL_MAP)
    register_image_gen(TOOL_DEFINITIONS, TOOL_MAP)
