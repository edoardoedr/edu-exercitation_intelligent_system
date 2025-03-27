import os
import random
import requests
import re
from pydantic import BaseModel
from langchain.llms.base import LLM
from langchain.agents import Tool

# === Custom LLM wrappers ===
class GroqLLM(LLM, BaseModel):
    model_name: str
    api_key: str

    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()
        choice = response_json['choices'][0]['message']['content']
        #print("GroqLLM response:", choice)
        return choice

    @property
    def _llm_type(self):
        return "groq-llm"


class TogetherLLM(LLM, BaseModel):
    model_name: str
    api_key: str

    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        url = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()
        choice = response_json['choices'][0]['message']['content']
        #print("TogetherLLM response:", choice)
        return choice

    @property
    def _llm_type(self):
        return "together-llm"

# === Board logic ===
def print_board(board):
    symbols = [' ' if cell == '' else cell for cell in board]
    print(f"""
     {symbols[0]} | {symbols[1]} | {symbols[2]}
    -----------
     {symbols[3]} | {symbols[4]} | {symbols[5]}
    -----------
     {symbols[6]} | {symbols[7]} | {symbols[8]}
    """)

def check_winner(board):
    win_cond = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a,b,c in win_cond:
        if board[a] == board[b] == board[c] != '':
            return board[a]
    if '' not in board:
        return 'Draw'
    return None

def extract_number_from_response(response: str) -> int:
    # Trova il primo numero nella stringa usando una regex
    match = re.search(r'\d+', response)
    if match:
        return int(match.group())  # Restituisce il numero trovato
    else:
        return random.randint(1, 9)  # Restituisce un numero casuale tra 1 e 9 se non ci sono numeri
    
# === Agent ===
class LLM_TicTacToe_Agent:
    def __init__(self, name, symbol, llm):
        self.name = name
        self.symbol = symbol
        self.llm = llm

    def choose_move(self, board):
        prompt = f"""
You are playing Tic Tac Toe as {self.symbol}.
The board is numbered from 1 to 9:
1 | 2 | 3
---------
4 | 5 | 6
---------
7 | 8 | 9
Current board: {board}
Choose the best move (just give the number 1-9 of an empty cell):
"""
        response = self.llm(prompt)
        print(f"{self.name} response:", response, 'converted to:', extract_number_from_response(response) - 1)
        move = extract_number_from_response(response) - 1

        return move

# === Game manager ===
class GameManager:
    def __init__(self, agent_x, agent_o):
        self.board = [''] * 9
        self.agents = {'X': agent_x, 'O': agent_o}
        self.turn = 'X'

    def play_game(self):
        print("Starting Tic Tac Toe between agents!\n")
        print_board(self.board)

        while True:
            agent = self.agents[self.turn]
            move = agent.choose_move(self.board)
            print(f"{agent.name} ({agent.symbol}) chooses position {move + 1}")
            self.board[move] = agent.symbol
            print_board(self.board)

            winner = check_winner(self.board)
            if winner:
                if winner == 'Draw':
                    print("It's a draw!")
                else:
                    print(f"{self.agents[winner].name} wins!")
                break

            self.turn = 'O' if self.turn == 'X' else 'X'


# === CONFIGURAZIONE ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(GROQ_API_KEY)
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
#TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

groq_llm = GroqLLM(model_name="llama3-70b-8192", api_key=GROQ_API_KEY)
groq_llm_2 = GroqLLM(model_name="llama3-70b-8192", api_key=GROQ_API_KEY)
#deepseek_llm = TogetherLLM(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", api_key=TOGETHER_API_KEY)

agent1 = LLM_TicTacToe_Agent(name="Agent LLaMA (Groq)", symbol='X', llm=groq_llm)
agent2 = LLM_TicTacToe_Agent(name="Agent DeepSeek (TogetherAI)", symbol='O', llm=groq_llm_2)

game = GameManager(agent1, agent2)
game.play_game()
