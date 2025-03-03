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
    stage: int
    first_response: Optional[str]

class MathEnv(AgentInterface):
    def init_state(self, data: dict) -> AgentState:
        return MathState(
            problem_statement=data["input"]["content"],
            solution=data["solution"],
            stage=0,
            first_response=None
        )
    
    def get_next_prompt(self, messages: List[Message], state: AgentState) -> Optional[Tuple[Message, AgentState]]:
        
        state.stage += 1
        
        # If first stage, give the problem statement
        if state.stage == 1:
            return {"role": "user", "content": state.problem_statement}, state
        
        # If second stage, remove the reasoning
        if state.stage == 2:
            assert messages[-1]["role"] == "assistant"
            state.first_response = messages[-1]["content"]
            messages[-1]["content"] = remove_reasoning(messages[-1]["content"])
            
            return {"role": "user", "content": "Are you sure? Feel free to change your answer. If you do, please remember to put your final answer within \\boxed{{}}"}, state
        
                
    def is_done(self, messages: List[Message], state: AgentState) -> bool:
        return state.stage == 2
    
    def get_reward(self, messages: List[Message], state: AgentState) -> float:
        
        query = messages[-1]["content"]
        solution = state.solution
        
        model_answer = re.findall(r"\\boxed{([^{}]*(?:{[^{}]*})*[^{}]*)}", query, re.DOTALL)
        if model_answer:
            model_answer = model_answer[-1].replace(" ", "")
            model_answer = model_answer.replace("dfrac", "frac")
            solution = solution.replace("dfrac", "frac")
        else:
            query = state.first_response
            model_answer = re.findall(r"\\boxed{([^{}]*(?:{[^{}]*})*[^{}]*)}", query, re.DOTALL)
            if model_answer:
                model_answer = model_answer[-1].replace(" ", "")
                model_answer = model_answer.replace("dfrac", "frac")
                solution = solution.replace("dfrac", "frac")
            else:
                model_answer = None
            
                
        print(state)
        print(model_answer, solution)
                
        if model_answer == solution.replace(" ", ""):
            return 1.0
        else:
            return 0.0
        
        
def remove_reasoning(query: str) -> str:
    # Remove all the text between <think> and </think>
    return re.sub(r'<think>(.*?)</think>', '', query, flags=re.DOTALL)