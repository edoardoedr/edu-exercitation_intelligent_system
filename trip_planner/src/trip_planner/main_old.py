from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun# Initialize the tool
from typing import Dict, List, Any, Tuple, Union
from pydantic import BaseModel
import json
from pydantic import Field
from dotenv import load_dotenv

load_dotenv() 

class ActivitySelection(BaseModel):
    selected_activities: List[str]
    total_cost: float


def validate_activity_output(result) -> Tuple[bool, Any]:
    """Validazione output attività: deve essere un dict JSON {str: float}"""
    try:
        if not isinstance(result, str):
            result = result.output
        print(f"Risultato della ricerca: {result}")
        dati = json.loads(result)
        if not isinstance(dati, dict):
            return (False, "Output non è un dizionario")
        for k, v in dati.items():
            if not isinstance(k, str) or not isinstance(v, (int, float)):
                return (False, f"Chiave non stringa o valore non numerico: {k}: {v}")
        return (True, dati)
    except Exception as e:
        return (False, f"Errore di parsing JSON: {str(e)}")
    
class CalculatorTool(BaseTool):
    name: str = "Calculator"
    description: str = (
        "Useful to calculate the total cost of a list of activities. "
        "Takes a JSON string as input with activity names as keys and costs as values."
    )

    def _run(self, attivita: Any) -> str:
        """Calcola il totale da un dizionario JSON con costi"""
        try:
            if isinstance(attivita, dict):
                dati = attivita
            else:
                dati = json.loads(attivita)
            totale = sum(dati.values())
            return f"Totale stimato: {totale:.2f} euro"
        except Exception as e:
            return f"Errore nel calcolo: {str(e)}"

class SearchTool(BaseTool):
    name: str = "Search"
    description: str = (
        "Useful for search-based queries. Use this to find current information about markets, companies, and trends."
    )
    search: DuckDuckGoSearchRun = Field(default_factory=DuckDuckGoSearchRun)

    def _run(self, query: Any) -> str:
        """Execute the search query and return results"""
        try:
            # Gestione input che potrebbe essere un dizionario (errato)
            if isinstance(query, dict):
                query = query.get("query") or query.get("description") or str(query)
            return self.search.run(query)
        except Exception as e:
            return f"Error performing search: {str(e)}"

# 🔍 Agente 1: Ricercatore
research_agent = Agent(
    role='Ricercatore Viaggi',
    goal='Trovare attività interessanti da fare in una città e stimare i costi',
    backstory='Esperto di ricerche online e turismo, utilizza DuckDuckGo per cercare attività turistiche e prezzi.',
    tools=[SearchTool()],
    verbose=True,
)

# 🧾 Agente 3: Interprete Risultati
parser_agent = Agent(
    role='Interprete Testuale',
    goal='Trasformare contenuti testuali in un dizionario JSON strutturato attività: costo',
    backstory="Esperto nell'estrazione di dati strutturati da contenuti non strutturati.",
    verbose=True,
)


# 💸 Agente 2: Selezionatore Economico
selector_agent = Agent(
    role='Ottimizzatore Itinerario',
    goal='Selezionare una combinazione di attività che rientrino in un budget dato',
    backstory="Esperto nell'ottimizzazione e gestione di budget turistici. Usa strumenti di calcolo per trovare combinazioni sostenibili.",
    tools=[CalculatorTool()],
    verbose=True,
)

# 📌 Task 1: Ricerca
task_research = Task(
    description="Utilizza il motore di ricerca per identificare le principali attività turistiche a {city}.",
    expected_output="una descrizione di posti da visitare e costi stimati.",
    agent=research_agent,
)

# 📌 Task 2: Parsing
task_parsing = Task(
    description="Ricevi una descrizione testuale con attività turistiche e costi. Estrai da essa un dizionario JSON valido con struttura attività: costo."
                "Fornisci un dizionario JSON valido dove le chiavi sono i nomi delle attività e i valori sono i costi stimati in euro. "
                "Esempio: 'Museo del Prado': 15.0, 'Tour della città': 30.0, 'Degustazione di tapas': 25.0.",
    expected_output="Un dizionario JSON valido con attività e costi.",
    agent=parser_agent,
    #guardrail=validate_activity_output
)


# 📌 Task 3: Selezione
task_selection = Task(
    description="Ricevi un dizionario JSON con attività e relativi costi. "
                "Seleziona la combinazione ottimale di attività il cui costo totale non superi {budget} euro. "
                "Restituisci un JSON valido con la lista delle attività selezionate e il costo totale.",
    expected_output="Un JSON valido con le attività selezionate e il costo totale.",
    agent=selector_agent,
    #output_json=ActivitySelection
)

# 🧠 Crew
crew = Crew(
    agents=[research_agent, parser_agent, selector_agent],
    tasks=[task_research, task_parsing, task_selection],
    process=Process.sequential,
    verbose=True,
    output_log_file = 'log_crew.txt'
)

# 🚀 Avvio della Crew con input dinamici
result = crew.kickoff(inputs={"city": "Madrid", "budget": 400})
print(result)