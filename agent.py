from langchain.tools import Tool
from kedro.framework.session import KedroSession
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama


def run_kedro_pipeline(pipeline_name: str = "default") -> str:
    try:
        with KedroSession.create() as session:
            session.run(pipeline_name=pipeline_name)
        return f"Pipeline '{pipeline_name}' executed successfully."
    except Exception as e:
        return f"Error running pipeline: {str(e)}"


kedro_tool = Tool(
    name="run_kedro_pipeline",
    func=run_kedro_pipeline,
    description="Runs a specified Kedro pipeline by name (default: 'default').",
)


llm = ChatOllama(model="qwen3:8b", temperature=0)

graph = create_react_agent(llm, [kedro_tool])

result = graph.invoke({"input": "Run the default Kedro pipeline."})
print(result["output"])
