from langchain.tools import Tool
from kedro.framework.session import KedroSession
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
import structlog

logger = structlog.get_logger(__name__)


def run_kedro_pipeline(pipeline_name: str = "default") -> str:
    logger.info("Tool called: run_kedro_pipeline", pipeline_name=pipeline_name)
    try:
        logger.info("Starting Kedro session", pipeline_name=pipeline_name)
        with KedroSession.create() as session:
            logger.info("Running pipeline", pipeline_name=pipeline_name)
            session.run(pipeline_name=pipeline_name)
        result = f"Pipeline '{pipeline_name}' executed successfully."
        logger.info(
            "Pipeline execution completed", pipeline_name=pipeline_name, result=result
        )
        return result
    except Exception as e:
        error_msg = f"Error running pipeline: {str(e)}"
        logger.error(
            "Pipeline execution failed", pipeline_name=pipeline_name, error=str(e)
        )
        return error_msg


kedro_tool = Tool(
    name="run_kedro_pipeline",
    func=run_kedro_pipeline,
    description="Runs a specified Kedro pipeline by name (default: 'default').",
)

logger.info("Setting up LLM and agent")
llm = ChatOllama(model="qwen3:8b", temperature=0)

logger.info("Creating ReAct agent with tools", tools=["run_kedro_pipeline"])
try:
    graph = create_react_agent(
        llm,
        [kedro_tool],
        prompt=(
            "You are a helpful assistant that can run Kedro pipelines. "
            "When a user asks you to run a pipeline, you should use the run_kedro_pipeline tool "
            "with the appropriate pipeline name.",
        )
    )
    logger.info("ReAct agent created successfully")
except Exception as e:
    logger.error("Failed to create ReAct agent", error=str(e))
    raise

logger.info("Invoking agent with input")
try:
    result = graph.invoke({"input": "Run the default Kedro pipeline."})
    logger.info(
        "Agent execution completed",
        result_keys=list(result.keys())
        if isinstance(result, dict)
        else "non-dict-result",
    )
    logger.info("Agent result", result=result)
    if isinstance(result, dict) and "output" in result:
        logger.info("Agent output", output=result["output"])
    else:
        logger.info("Full agent result", full_result=result)
except Exception as e:
    logger.error("Agent execution failed", error=str(e))
    raise
