import json
import os
import re
import numpy as np
from litellm import completion
from sympy.parsing.latex import parse_latex  # :contentReference[oaicite:0]{index=0}
from sympy import simplify
from tqdm import tqdm
from collections import defaultdict
from latex2sympy2 import latex2sympy

all_models = {
    "0.5": "ollama/qwen3:0.6b",
    "1": "ollama/llama3.2:1b",
    "3": "ollama/llama3.2:3b",
}

def get_responses(model, prompt, k=1):
    responses = completion(
        model=all_models[model], 
        messages=[{"content": prompt,"role": "user"}], 
        api_base="http://localhost:11434",
        n=k,
        max_tokens=2048,
    )
    return [
        responses.choices[i].message['content'] for i in range(len(responses.choices))
    ]

def fish_answer(response):
    answers = []
    # Regex pattern to find content within <answer>...</answer> OR \boxed{...}
    # The pattern uses two groups, one for each format.
    # - Group 1: (.*?) captures content inside <answer>...</answer>
    # - Group 2: (.*?) captures content inside \boxed{...}
    # re.DOTALL makes '.' match newlines as well, in case the answer spans multiple lines.
    # The non-greedy '.*?' is used to match the shortest possible string.
    pattern = r"<answer>(.*?)</answer>|\\boxed{(.*?)}"
    
    for match in re.finditer(pattern, response, re.DOTALL):
        # Check which group was matched
        if match.group(1) is not None:  # Matched <answer>content</answer>
            answers.append(match.group(1).strip()) # .strip() to remove leading/trailing whitespace
            break
        elif match.group(2) is not None:  # Matched \boxed{content}
            answers.append(match.group(2).strip()) # .strip()
            break
            
    if len(answers) == 0: return None
    return answers[-1]

def check_answer(answers, truth, backend='antlr'):
    results = []
    try:
        ans_sympy = latex2sympy(truth)
    except:
        return None
    for ans in answers:
        if ans is None:
            results.append(False)
        else:
            try:
                expr1 = latex2sympy(ans)
                eq = simplify(expr1 - ans_sympy) == 0
                results.append(bool(eq))
            except:
                print("parsing fail, ", ans)
                results.append(False)

    return np.array(results)

def pass_k(model, prompt, truth, k):
    responses = get_responses(model, prompt, k)
    answers = list(map(fish_answer, responses))
    success = check_answer(answers, truth)
    if success is None: return None
    return success.any()


def load_data(file_path="omnimath_100.json"):
    dataset_with_hints = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f): # Added line_num for better error reporting
            try:
                dataset_with_hints.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line {line_num+1}")

    data = defaultdict(list)
    for l in dataset_with_hints:
        for k, v in l.items():
            data[k].append(v)
    return data

def get_prompt(question, hint):
    return f"""
    Here is a math question that you should solve. 
    {question}

    Follow these step-by-step instructions to arrive at the answer. 
    {hint}

    Include your answer in <answer></answer> tag. 
    """

def evaluate(model, data, k=1):
    success = []
    for prompt, answer in tqdm(zip(data['prompt'], data['answer']), desc=f"Evaluating {model}"):
        s = pass_k(model, prompt, answer, k)
        if s is not None:
            success.append(s)

    return np.array(success)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, nargs='+', default=all_models.values(), help="One or more models to evaluate")
    parser.add_argument("--data", type=str, default="omnimath_100_with_hints_v2.jsonl")
    args = parser.parse_args()

    data = load_data(args.data)
    data['answer'] = data['final_answer_gt']
    data['prompt'] = list(map(get_prompt, data['question'], data['hint']))
    for model in args.model:
        success = evaluate(model, data)
        print(f"\n\n---------Evaluating model {model}--------\n\n\n")
        print("success rate for pass@1: ", np.mean(success))