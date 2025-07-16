"""Kedro Management Agent using LangGraph ReAct framework."""

import os
import subprocess

import structlog
from kedro.framework.session import KedroSession
from langchain.tools import StructuredTool, Tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class CreateKedroProjectArgs(BaseModel):
    """Arguments for creating a Kedro project."""

    project_name: str = Field(
        description=(
            "EXACT name of the new Kedro project as specified by the user "
            "- extract from quotes or after 'called'/'named'"
        )
    )
    tools: str = Field(
        default="none",
        description="Tools to include: 'all', 'none', or comma-separated list",
    )
    example: str = Field(
        default="n", description="Include example pipeline: 'y' or 'n'"
    )
    telemetry: str = Field(default="n", description="Enable telemetry: 'y' or 'n'")


def run_kedro_pipeline(pipeline_name: str = "default") -> str:
    """Run a Kedro pipeline by name."""
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


def create_kedro_project(
    project_name: str, tools: str = "none", example: str = "n", telemetry: str = "n"
) -> str:
    """
    Create a new Kedro project using kedro new command.

    Args:
        project_name: Name of the new Kedro project
        tools: Tools to include (none, all, or comma-separated list like
            'lint,test,log,docs,data,pyspark')
        example: Whether to include example pipeline ('y' or 'n')
        telemetry: Whether to enable telemetry ('y' or 'n')
    """
    logger.info(
        "Tool called: create_kedro_project",
        project_name=project_name,
        tools=tools,
        example=example,
        telemetry=telemetry,
    )

    try:
        # Build the kedro new command
        cmd = [
            "kedro",
            "new",
            "--name",
            project_name,
            "--tools",
            tools,
            "--example",
            example,
            "--telemetry",
            telemetry,
        ]

        logger.info("Executing kedro new command", command=" ".join(cmd))

        # Run the command
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=os.getcwd(), check=False
        )

        if result.returncode == 0:
            success_msg = f"Kedro project '{project_name}' created successfully!"
            if result.stdout:
                success_msg += f"\n\nOutput:\n{result.stdout}"
            logger.info(
                "Project creation completed",
                project_name=project_name,
                result=success_msg,
            )
            return success_msg
        else:
            error_msg = (
                f"Error creating Kedro project: {result.stderr or result.stdout}"
            )
            logger.error(
                "Project creation failed", project_name=project_name, error=error_msg
            )
            return error_msg

    except Exception as e:
        error_msg = f"Error creating Kedro project: {str(e)}"
        logger.error("Project creation failed", project_name=project_name, error=str(e))
        return error_msg


kedro_tool = Tool(
    name="run_kedro_pipeline",
    func=run_kedro_pipeline,
    description="Runs a specified Kedro pipeline by name (default: 'default').",
)

create_project_tool = StructuredTool.from_function(
    func=create_kedro_project,
    name="create_kedro_project",
    description=(
        "Creates a new Kedro project. Always extract the EXACT project name "
        "from the user's input. Look for project names in quotes, after "
        "'called', 'named', or similar phrases. Parameters: "
        "project_name (required): Extract the exact name from user input - "
        "look carefully for quoted names or names after 'called'/'named'. "
        "tools (optional, default 'none'): Use 'all' if user mentions "
        "'all tools', otherwise 'none'. "
        "example (optional, default 'n'): Use 'y' if user mentions "
        "'example' or 'sample', otherwise 'n'. "
        "telemetry (optional, default 'n'): Use 'y' only if user explicitly "
        "asks for telemetry, 'n' if they say disable/no telemetry."
    ),
    args_schema=CreateKedroProjectArgs,
)

logger.info("Setting up LLM and agent")
llm = ChatOllama(model="qwen3:8b", temperature=0)

logger.info(
    "Creating ReAct agent with tools",
    tools=["run_kedro_pipeline", "create_kedro_project"],
)
try:
    graph = create_react_agent(
        llm,
        [kedro_tool, create_project_tool],
        prompt=(
            "You are a helpful assistant that specializes in Kedro project "
            "management. You can help users create new Kedro projects and run "
            "existing pipelines. IMPORTANT: Always read the user's input "
            "carefully and extract the EXACT project name they specify. "
            "Pay close attention to names in quotes or after words like "
            "'called' or 'named'."
        ),
    )
    logger.info("ReAct agent created successfully")
except Exception as e:
    logger.error("Failed to create ReAct agent", error=str(e))
    raise

logger.info("Invoking agent with input")
try:
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Create a new Kedro project called 'analytics-pipeline' "
                        "with all tools, include an example pipeline, and "
                        "disable telemetry."
                    ),
                }
            ]
        }
    )
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
