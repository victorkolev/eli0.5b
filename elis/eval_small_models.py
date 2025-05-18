import json
import os
import re
import numpy as np
from litellm import completion
from sympy.parsing.latex import parse_latex  # :contentReference[oaicite:0]{index=0}
from sympy import simplify
from tqdm import tqdm

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
    )
    return [
        responses.choices[i].message['content'] for i in range(len(responses.choices))
    ]

def fish_answer(response):
    text_content = [response]
    answers = []
    # Regex pattern to find content within <answer>...</answer> OR \boxed{...}
    # The pattern uses two groups, one for each format.
    # - Group 1: (.*?) captures content inside <answer>...</answer>
    # - Group 2: (.*?) captures content inside \boxed{...}
    # re.DOTALL makes '.' match newlines as well, in case the answer spans multiple lines.
    # The non-greedy '.*?' is used to match the shortest possible string.
    pattern = r"<answer>(.*?)</answer>|\\boxed{(.*?)}"
    
    for match in re.finditer(pattern, text_content, re.DOTALL):
        # Check which group was matched
        if match.group(1) is not None:  # Matched <answer>content</answer>
            answers.append(match.group(1).strip()) # .strip() to remove leading/trailing whitespace
        elif match.group(2) is not None:  # Matched \boxed{content}
            answers.append(match.group(2).strip()) # .strip()
            
    return answers[-1]

def check_answer(answers, truth, backend='antlr'):
    results = []
    expr2 = parse_latex(truth, backend=backend)
    for ans in answers:
        expr1 = parse_latex(ans, backend=backend)

        eq = simplify(expr1 - expr2) == 0
        results.append(bool(eq))

    return np.array(results)

def pass_k(model, prompt, truth, k):
    responses = get_responses(model, prompt, k)
    answers = map(fish_answer, responses)
    success = check_answer(answers, truth).any()
    return success


def load_data(file_path="omnimath_100.json"):
    with open(file_path) as f:
        data = json.load(f)
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
        success.append(pass_k(model, prompt, answer, k))

    return np.array(success)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, nargs='+', default=all_models.values(), help="One or more models to evaluate")
    parser.add_argument("--data", type=str, default="omnimath_100_with_hints_v2.jsonl")
    args = parser.parse_args()

    data = load_data(args.data)
    data['answer'] = data['final_answer_gt']
    data['prompt'] = map(get_prompt, zip(data['question'], data['hint']))
    for model in args.model:
        success = evaluate(model)
        print(f"\n\n---------Evaluating model {model}--------\n\n\n")
        print("success rate for pass@1: ", np.mean(success))