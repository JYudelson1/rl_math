from openrlhf.utils.interface import AgentInterface
from typing import *
import json, re, os
import dotenv
import logging
from dataclasses import dataclass

logger = logging.getLogger("ray")

dotenv.load_dotenv()    

Message = Dict[str, str]
AgentState = Any

@dataclass
class MathState:
    problem_statement: str
    solution: str
    finished: bool

class MathEnv(AgentInterface):
    def init_state(self, data: dict) -> AgentState:
        return MathState(
            problem_statement=data["input"]["content"],
            solution=data["solution"],
            finished=False
        )
    
    def get_next_prompt(self, messages: List[Message], state: AgentState) -> Optional[Tuple[Message, AgentState]]:
        
        state.finished = True
        return {"role": "user", "content": state.problem_statement}, state
                
    def is_done(self, messages: List[Message], state: AgentState) -> bool:
        return state.finished
    
    def get_reward(self, messages: List[Message], state: AgentState) -> float:
        """We get the reward by calling OpenAI's API to use o1-mini to grade the suspiciousness of the code"""
        
        query = messages[-1]["content"]
        solution = state.solution
        
        model_answer = re.findall(r"\\boxed{([^{}]*(?:{[^{}]*})*[^{}]*)}", query, re.DOTALL)
        if model_answer:
            model_answer = model_answer[-1].replace(" ", "")
            model_answer = model_answer.replace("dfrac", "frac")
            solution = solution.replace("dfrac", "frac")
            model_answer = model_answer
        else:
            model_answer = None
                
        # print(state)
        # print(model_answer, solution)
                
        if model_answer == solution.replace(" ", ""):
            return 1.0
        else:
            return 0.0