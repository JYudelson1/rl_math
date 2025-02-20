from typing import List, Callable, Dict, Any, Sequence, Tuple
from vllm import LLM, SamplingParams, RequestOutput
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
import re

import torch

from os.path import isfile
import pickle
import json
from collections.abc import Callable, Iterable

def pickle_load(filename: str) -> Any:
    with open(filename, "rb") as f:
        return pickle.load(f)

def pickle_dump(filename: str, object: Any) -> None:
    with open(filename, "wb") as f:
        return pickle.dump(object, f)

def json_load(filename: str) -> Any:
    with open(filename, "r") as f:
        return json.load(f)

def json_dump(filename: str, object: Any) -> None:
    with open(filename, "w") as f:
        return json.dump(object, f)

def jsonl_load(filename: str) -> list[Any]:
    with open(filename, "r") as f:
        return [json.loads(line) for line in f if line.strip() != ""]

def jsonl_dump(filename: str, objects: Iterable[Any]) -> None:
    with open(filename, "w") as f:
        for object in objects:
            f.write(json.dumps(object))
            f.write("\n")

def run_or_load(filename: str, function: Callable, *args, verbose_load: bool = True, **kwargs) -> Any:
    assert any(filename.endswith(format) for format in [".pkl", ".json", ".txt", ".pt"])
    if isfile(filename):
        if filename.endswith(".pkl"):
            saved_result = pickle_load(filename)
        elif filename.endswith(".json"):
            saved_result = json_load(filename)
        elif filename.endswith(".txt"):
            with open(filename, "r") as f:
                saved_result = f.read()
        elif filename.endswith(".pt"):
            saved_result = torch.load(filename)
        else:
            assert False, "unreachable"
        if verbose_load:
            print(f"Loaded from file '{filename}'.")
        return saved_result
    result = function(*args, **kwargs)
    if filename.endswith(".pkl"):
        pickle_dump(filename, result)
    elif filename.endswith(".json"):
        json_dump(filename, result)
    elif filename.endswith(".txt"):
        assert isinstance(result, str), \
               "Can only save a string to a txt file. " \
               "Pass a filename ending in .pkl or .json to save richer datatypes."
        with open(filename, "w") as f:
            f.write(result)
    elif filename.endswith(".pt"):
        torch.save(result, filename)
    else:
        assert False, "unreachable"
    return result

class MathEnv:
    def step(self,
             states: List[Dict[str, Any]],
             llm: LLM,
             sampling_params: SamplingParams) -> Tuple[List[Dict[str, Any]], List[RequestOutput]]:
        
        outputs = llm.chat([state["messages"] for state in states], sampling_params=sampling_params) # type: ignore
        for i, state in enumerate(states):
            state["messages"].append({'role': 'assistant', 'content': outputs[i].outputs[0].text})
            state["messages"].append({'role': 'user', 'content': 'Are you sure? If you don\'t want to change your answer, just return the same answer, in \\boxed{{}}. If you want to change your answer, return the new answer, in \\boxed{{}}.'})
            state["prompt_tokens"] = len(outputs[i].prompt_token_ids)

        outputs = llm.chat([state["messages"] for state in states], sampling_params=sampling_params) # type: ignore
        for i, state in enumerate(states):
            state["messages"].append({'role': 'assistant', 'content': outputs[i].outputs[0].text})
            state["completed"] = True
        return states, outputs

    def generate(self, prompts: List[List[Dict[str, Any]]], llm: LLM, sampling_params: SamplingParams) -> List[Sequence[int]]:
        print("First prompt: ", prompts[0])
        all_completed = False
        template = """Please solve the following mathematical problem, placing your final answer within \\boxed{{}}:\n{math_problem}"""
        states = [{"messages": [{"role": "user", "content": template.format(math_problem=prompt)}], "completed": False, "prompt_tokens": -1} for prompt in prompts]
        outputs = [None] * len(prompts)
        while not all_completed:
            h = hash(str([state["messages"][-1]["content"] for state in states]))
            filename = f"cache/step_{h}.pkl"
            
            states, outputs = run_or_load(filename, self.step, states, llm, sampling_params)
            all_completed = all(state["completed"] for state in states)
        all_ids = [list(output.prompt_token_ids) + list(output.outputs[0].token_ids) for output in outputs]
        completion_ids = [all_ids[i][states[i]["prompt_tokens"]:] for i in range(len(outputs))]
        #print("Device: ", completion_ids[0][0].device)
        return completion_ids
    
math_dataset = load_dataset("EleutherAI/hendrycks_math", "intermediate_algebra" ,split="test")
# rename the "problem" column to "prompt"
math_dataset = math_dataset.rename_column("problem", "prompt")
print(f"Length of dataset: {len(math_dataset)}")
math_solutions = {}
bad_indices = []
for i in range(len(math_dataset)):
    full_solution = math_dataset[i]['solution']
    # To get the final solution, get the very last piece of text inside /boxed{}
    solution_groups = re.findall(r"\\boxed{([^{}]*(?:{[^{}]*})*[^{}]*)}", full_solution)
    if solution_groups:
        math_solutions[math_dataset[i]['prompt']] = solution_groups[-1]
    else:
        bad_indices.append(i)
good_indices = [i for i in range(len(math_dataset)) if i not in bad_indices]
math_dataset = math_dataset.select(good_indices)
print(f"Length of dataset: {len(math_dataset)}")

eval_dataset = math_dataset.select(range(88))
train_dataset = math_dataset.select(range(88, len(math_dataset)))

def get_reward(completions: List[str], **kwargs) -> List[float]:
        """We get the reward by calling OpenAI's API to use o1-mini to grade the suspiciousness of the code"""
        #print(kwargs["prompts"][:5])
        #print(completions[:5])
        solutions = [math_solutions.get(prompt) for prompt in kwargs["prompts"]]
        #print("Solutions: ", solutions[:5])
        solutions = [solution.replace("dfrac", "frac") if solution else None for solution in solutions]
        
        model_answers = [re.findall(r"\\boxed{([^{}]*(?:{[^{}]*})*[^{}]*)}", completion, re.DOTALL) for completion in completions]
        model_answers = [model_answer[-1] if model_answer else None for model_answer in model_answers]
        model_answers = [model_answer.replace(" ", "") if model_answer else None for model_answer in model_answers]
        model_answers = [model_answer.replace("dfrac", "frac") if model_answer else None for model_answer in model_answers]
        
        rewards = [1.0 if model_answer == solution else 0.0 for model_answer, solution in zip(model_answers, solutions)]
        return rewards

training_args = GRPOConfig(
    output_dir="output/trl_test", 
    logging_steps=1, 
    run_name="trl_test", 
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    max_prompt_length=1024,
    max_completion_length=4000,
    num_generations=3,
    bf16=True,
    bf16_full_eval=True,
    eval_strategy="steps",
    eval_steps=4,
    load_best_model_at_end=True,
    save_strategy="steps",
    save_steps=4,
    use_vllm=True,
    vllm_gpu_memory_utilization=0.45, # If you run out of memory, try reducing this
    vllm_max_model_len=11000,
    #deepspeed="deepspeed-config.json",
    #accelerator_config="trl_config.yaml",
    #learning_rate=1e-6,
)

trainer = GRPOTrainer(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    reward_funcs=get_reward,
    env=MathEnv(),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)
trainer.train()

