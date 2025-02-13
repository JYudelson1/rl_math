import argparse
import re
import json
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)

class RewardModelProxy:
    def __init__(self, args):
        self.max_length = args.max_len
        self.batch_size = args.batch_size

    def get_reward(self, queries, solutions):
        print(len(queries), len(solutions))
        print(queries[0], "\n\nSolution:", solutions[0])
        print("--------------------------------")
        
        scores = []
        # Correct answer will be the very last piece of text inside /boxed{}
        # We can check if the solution is correct by checking if the solution is the same as the correct answer 
        model_answers = []
        for query, solution in zip(queries, solutions):
            # Search for the correct answer in the query
            model_answer = re.findall(r"\\boxed{([^{}]*(?:{[^{}]*})*[^{}]*)}", query, re.DOTALL)
            if model_answer:
                model_answer = model_answer[-1].replace(" ", "")
                model_answer = model_answer.replace("dfrac", "frac")
                solution = solution.replace("dfrac", "frac")
                model_answers.append(model_answer)
            else:
                model_answers.append(None)
                
            if model_answer == solution.replace(" ", ""):
                scores.append(1.0)
            else:
                scores.append(0.0)


        for model_answer, real_answer in zip(model_answers, solutions):
            print("Model Answer:", model_answer, "\n\nReal Answer:", real_answer)
            print("--------------------------------")
        return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--max_len", type=int, default="2048")

    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    # Performance
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=None)

    args = parser.parse_args()

    # server
    reward_model = RewardModelProxy(args)
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        queries = data.get("query")
        solutions = data.get("solution")
        rewards = reward_model.get_reward(queries, solutions)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")                
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")