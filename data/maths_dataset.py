import json, random, re
from datasets import load_dataset

math_dataset = load_dataset("interview-eval/MATH", split="test")

template = """Please solve the following mathematical problem, placing your final answer within \\boxed{{}}:
{math_problem}"""

def make_dataset():
    dataset = []
    for i in range(len(math_dataset)):
        prompt = template.format(math_problem=f"{math_dataset[i]['initial_question']}")
        full_solution = math_dataset[i]['solution']
        # To get the final solution, get the very last piece of text inside /boxed{}
        solution_groups = re.findall(r"\\boxed{(.*?)}", full_solution)
        if solution_groups:
            solution = solution_groups[-1]
        else:
            print(f"No solution found for {i}")
            continue
        
        dataset.append({
            "input": {"role": "user", "content": prompt},
            "solution": solution
        })
    
    with open("/data1/joey/deepseek-tests/data/maths_dataset.json", "w+") as f:
        json.dump(dataset, f)

if __name__ == "__main__":
    make_dataset()
