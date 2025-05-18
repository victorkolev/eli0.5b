import litellm
import os
import json
import time

# --- Configuration ---
TOGETHERAI_API_KEY = ""
LLM_MODEL = "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
INPUT_DATASET_PATH = "omnimath_100.jsonl" # Your original dataset with problem, detailed_solution, AND concise_answer
OUTPUT_DATASET_PATH = "omnimath_100_with_hints_v2.jsonl" # Use a new name for the output
MAX_RETRIES = 3
RETRY_DELAY = 10

os.environ["TOGETHERAI_API_KEY"] = TOGETHERAI_API_KEY

def generate_hint_for_problem(problem_text):
    # ... (this function remains the same) ...
    hint_prompt = f"""
Here is a math question. Your task is to provide step-by-step hints that would guide a student towards the solution.
IMPORTANT:
1. Do NOT reveal the final answer or the final numerical/symbolic result.
2. Focus on the problem-solving process, key concepts, or intermediate steps.
3. Break down the problem if it's complex.
4. Hints should be progressive, leading the student logically.
5. Keep hints concise and clear.

Question:
{problem_text}

Hints:
"""
    for attempt in range(MAX_RETRIES):
        try:
            print(f"    Attempting API call for hint (attempt {attempt + 1}/{MAX_RETRIES})...")
            response = litellm.completion(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": hint_prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                print(f"    Warning: Received unexpected response structure: {response}")
                if hasattr(response, 'model_dump_json'):
                    print(f"    Full response dump: {response.model_dump_json(indent=2)}")
                return None
        except litellm.exceptions.RateLimitError as rle:
            print(f"    Rate limit error (attempt {attempt + 1}/{MAX_RETRIES}): {rle}")
            if attempt < MAX_RETRIES - 1:
                current_delay = RETRY_DELAY * (2 ** attempt)
                print(f"    Retrying in {current_delay} seconds due to rate limit...")
                time.sleep(current_delay)
            else:
                print("    Max retries reached due to rate limit. Skipping hint generation for this problem.")
                return None
        except Exception as e:
            print(f"    Error generating hint (attempt {attempt + 1}/{MAX_RETRIES}): {type(e).__name__} - {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"    Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print("    Max retries reached. Skipping hint generation for this problem.")
                return None

def main():
    input_data = []
    print(f"Attempting to load dataset from: {INPUT_DATASET_PATH}")
    # ... (dataset loading logic remains the same) ...
    try:
        if INPUT_DATASET_PATH.endswith(".jsonl"):
            with open(INPUT_DATASET_PATH, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        input_data.append(json.loads(line))
                    except json.JSONDecodeError as je:
                        print(f"Error decoding JSON on line {line_num+1} in {INPUT_DATASET_PATH}: {je}")
        elif INPUT_DATASET_PATH.endswith(".json"):
            with open(INPUT_DATASET_PATH, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
        else:
            print(f"Error: Unsupported input file format: {INPUT_DATASET_PATH}.")
            return
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_DATASET_PATH}")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if not isinstance(input_data, list):
        print(f"Error: Expected a list of problems from {INPUT_DATASET_PATH}, but got {type(input_data)}")
        return
    print(f"Loaded {len(input_data)} problems from {INPUT_DATASET_PATH}")

    processed_data = []
    failed_to_get_hint_count = 0

    for i, item in enumerate(input_data):
        print(f"\nProcessing problem {i+1}/{len(input_data)}...")
        original_problem_text = item.get("problem")
        # --- MODIFIED SECTION TO GET BOTH SOLUTION TYPES ---
        original_detailed_solution = item.get("solution") # This is the long official solution
        original_concise_answer = item.get("answer")    # This is the short final answer (e.g., "60")
        # --- END OF MODIFIED SECTION ---

        if not original_problem_text:
            print(f"  Warning: Skipping item {i+1} due to missing 'problem' field.")
            continue
        if original_detailed_solution is None:
             print(f"  Warning: Item {i+1} has a missing or null 'detailed_solution' field.")
        if original_concise_answer is None:
             print(f"  Warning: Item {i+1} has a missing or null 'answer' (concise final answer) field. This will be critical for evaluation.")


        print(f"  Original Problem (first 100 chars): {original_problem_text[:100].replace(os.linesep, ' ')}...")

        generated_hint = generate_hint_for_problem(original_problem_text)

        if generated_hint:
            print(f"  Generated Hint (first 100 chars): {generated_hint[:100].replace(os.linesep, ' ')}...")
            # --- MODIFIED DATAPOINT STRUCTURE ---
            new_datapoint = {
                "question": original_problem_text,
                "hint": generated_hint,
                "detailed_solution": original_detailed_solution, # Store the long one for reference
                "final_answer_gt": original_concise_answer     # Store the short one for evaluation
            }
            # You can still include other original fields if needed:
            # new_datapoint["domain"] = item.get("domain")
            # new_datapoint["difficulty"] = item.get("difficulty")
            # new_datapoint["source"] = item.get("source")
            # --- END OF MODIFIED DATAPOINT STRUCTURE ---
            processed_data.append(new_datapoint)
        else:
            print(f"  Failed to generate hint for problem {i+1}. This problem will not be included in the output.")
            failed_to_get_hint_count += 1

    try:
        with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as f_out:
            for entry in processed_data:
                json.dump(entry, f_out, ensure_ascii=False)
                f_out.write('\n')
        print(f"\nSuccessfully generated hints for {len(processed_data)} problems.")
        if failed_to_get_hint_count > 0:
            print(f"Failed to generate hints for {failed_to_get_hint_count} problems.")
        print(f"New dataset saved to {OUTPUT_DATASET_PATH}")
    except IOError as e:
        print(f"Error writing output file {OUTPUT_DATASET_PATH}: {e}")

if __name__ == "__main__":
    main()