from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from typing import Dict, List, Any, Tuple, Union
from pydantic import BaseModel, Field
import json
from dotenv import load_dotenv
import datetime
load_dotenv()

# ✅ Pydantic model for task output
class ActivitySelection(BaseModel):
    selected_activities: List[str]
    total_cost: float

# ✅ Guardrail function to validate JSON output from parser_agent
def validate_activity_output(result) -> Tuple[bool, Any]:
    """Validate activity output: must be a JSON dict {str: float}"""
    try:
        if not isinstance(result, str):
            result = result.output
        print(f"Search result: {result}")
        data = json.loads(result)
        if not isinstance(data, dict):
            return (False, "Output is not a dictionary")
        for k, v in data.items():
            if not isinstance(k, str) or not isinstance(v, (int, float)):
                return (False, f"Invalid key-value: {k}: {v}")
        return (True, data)
    except Exception as e:
        return (False, f"JSON parsing error: {str(e)}")

# ✅ Tool for calculating total cost
class CalculatorTool(BaseTool):
    name: str = "Calculator"
    description: str = (
        "Useful to calculate the total cost of a list of activities. "
        "Takes a JSON string as input with activity names as keys and costs as values."
    )

    def _run(self, attivita: Any) -> str:
        """Calculates the total cost from a JSON dict of activity costs"""
        try:
            if isinstance(attivita, dict):
                data = attivita
            else:
                data = json.loads(attivita)
            total = sum(data.values())
            return f"Estimated total: {total:.2f} euro"
        except Exception as e:
            return f"Calculation error: {str(e)}"

# ✅ Tool for web search using DuckDuckGo
class SearchTool(BaseTool):
    name: str = "Search"
    description: str = (
        "Useful for search-based queries. Use this to find current information about markets, companies, and trends."
    )
    search: DuckDuckGoSearchRun = Field(default_factory=DuckDuckGoSearchRun)

    def _run(self, query: Any) -> str:
        """Execute the search query and return results"""
        try:
            if isinstance(query, dict):
                query = query.get("query") or query.get("description") or str(query)
            return self.search.run(query)
        except Exception as e:
            return f"Error performing search: {str(e)}"

# 🔍 Agent 1: Travel Researcher
research_agent = Agent(
    role='Travel Researcher',
    goal='Find interesting activities to do in a city and estimate their costs',
    backstory='An expert in online research and tourism, uses DuckDuckGo to search for tourist activities and prices.',
    tools=[SearchTool()],
    verbose=True,
)

# 🧾 Agent 2: Text Parser
parser_agent = Agent(
    role='Text Interpreter',
    goal='Transform unstructured text into a structured JSON dictionary of activity: cost',
    backstory="Expert in extracting structured data from unstructured content.",
    verbose=True,
)

# 💸 Agent 3: Budget Selector
selector_agent = Agent(
    role='Itinerary Optimizer',
    goal='Select a combination of activities that stay within a given budget',
    backstory="Specialist in travel budget optimization. Uses calculation tools to find sustainable combinations.",
    tools=[CalculatorTool()],
    verbose=True,
)

# 📌 Task 1: Search for activities
task_research = Task(
    description="Use the search engine to find top tourist activities in {city}.",
    expected_output="A description of places to visit and estimated costs.",
    agent=research_agent,
)

# 📌 Task 2: Parse the search results into structured format
task_parsing = Task(
    description="Receive a textual description of tourist activities and costs. Extract a valid JSON dictionary from it with structure activity: cost. "
                "Provide a valid JSON dictionary where the keys are the activity names and the values are the estimated costs in euros. "
                "Example: 'Museo del Prado': 15.0, 'City Tour': 30.0, 'Tapas Tasting': 25.0.",
    expected_output="A valid JSON dictionary with activities and costs.",
    agent=parser_agent,
    #guardrail=validate_activity_output
)

# 📌 Task 3: Select activities within budget
task_selection = Task(
    description="Receive a JSON dictionary with activities and their costs. "
                "Select the optimal combination of activities whose total cost does not exceed {budget} euros. "
                "Return a valid JSON with the selected activities and total cost.",
    expected_output="A valid JSON with selected activities and the total cost.",
    agent=selector_agent,
    #output_json=ActivitySelection
)

# 🧠 Crew configuration
crew = Crew(
    agents=[research_agent, parser_agent, selector_agent],
    tasks=[task_research, task_parsing, task_selection],
    process=Process.sequential,
    verbose=True,
    output_log_file=f'log_crew_{datetime.datetime.day}.txt'
)

# 🚀 Launch the crew with dynamic inputs
result = crew.kickoff(inputs={"city": "Madrid", "budget": 4000})
print(result)