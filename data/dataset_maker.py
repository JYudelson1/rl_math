import json, random

template = """
You are a helpful assistant. Please answer the following question, and put your final answer within \\boxed{{}}.
{input}
"""

def make_dataset():
    dataset = []
    for i in range(5000):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        op = random.choice(["+", "-", "*"])
        prompt = template.format(input=f"What is {a} {op} {b}?")
        datum = {"input": [{"role": "user", "content": prompt}]}
        datum["test_cases"] = {"input": "", "output": f"{a + b if op == '+' else a - b if op == '-' else a * b}"}
        dataset.append(datum)
    
    with open("/data1/joey/rl_math/data/train.json", "w+") as f:
        json.dump(dataset, f)

if __name__ == "__main__":
    make_dataset()
