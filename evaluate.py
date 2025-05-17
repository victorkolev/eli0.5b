import json
from litellm import completion
from tqdm import tqdm

# Load the 20-question dataset
with open("omnimath_20.json", "r") as f:
    data = [json.loads(line) for line in f]

results = []
correct = 0

for i, example in enumerate(tqdm(data, desc="Evaluating")):
    question = example["problem"]
    ground_truth = example["answer"].strip()

    model_name = "ollama/mistral"

    try:
        response = completion(
            model=model_name,
            # model="ollama/qwen:0.5b",
            api_base="http://localhost:11434",
            messages=[{"role": "user", "content": question}]
        )
        model_response = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        model_response = f"[ERROR] {str(e)}"

    is_correct = ground_truth in model_response
    if is_correct:
        correct += 1

    results.append({
        "question": question,
        "ground_truth": ground_truth,
        "model_response": model_response,
        "correct": is_correct
    })

# Print a summary
print(f"\n✅ {model_name} Accuracy: {correct} / {len(data)}")

# Optionally: save results
with open("qwen_eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Optional: view incorrect ones
print("\n🔍 Incorrectly answered questions:")
for r in results:
    if not r["correct"]:
        print(f"\nQ: {r['question']}\nGT: {r['ground_truth']}\nModel: {r['model_response']}\n")
