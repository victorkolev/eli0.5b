import litellm
import os
import json
import time # To add delays if needed

# --- Configuration ---
TOGETHERAI_API_KEY = "4e8fd7f0826fb9dbaefcbaa1e3788ca9cad3614c17f88989087c6e3a38bafb00" # Replace with your actual key if different
LLM_MODEL = "together_ai/meta-llama/Meta-Llama-3-70B-Instruct-Turbo"
INPUT_DATASET_PATH = "omnimath_100.jsonl"
OUTPUT_DATASET_PATH = "omnimath_100_with_hints.jsonl" # Output filename
MAX_RETRIES = 3
RETRY_DELAY = 10 # seconds (increased slightly for potentially busy APIs)

# Set TogetherAI API key (can also be set as an environment variable outside the script)
os.environ["TOGETHERAI_API_KEY"] = TOGETHERAI_API_KEY

def generate_hint_for_problem(problem_text):
    """
    Generates a hint for a given math problem using the specified LLM.
    """
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
                temperature=0.3, # Lower temperature for more focused/deterministic hints
                max_tokens=500,  # Adjust as needed, 500 should be generous for hints
                # You might add other parameters like top_p if desired
                # timeout=60 # Optional: set a timeout for the API call
            )
            # Accessing content from litellm.ModelResponse
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                print(f"    Warning: Received unexpected response structure: {response}")
                # Try to log more details from the response if it's not as expected
                if hasattr(response, 'model_dump_json'):
                    print(f"    Full response dump: {response.model_dump_json(indent=2)}")
                return None # Indicate failure to get content
        except litellm.exceptions.RateLimitError as rle:
            print(f"    Rate limit error (attempt {attempt + 1}/{MAX_RETRIES}): {rle}")
            if attempt < MAX_RETRIES - 1:
                # Implement exponential backoff or longer fixed delay for rate limits
                current_delay = RETRY_DELAY * (2 ** attempt) # Exponential backoff
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
                return None # Or raise the exception if you want to stop the whole script

def main():
    # Load the input dataset
    input_data = []
    print(f"Attempting to load dataset from: {INPUT_DATASET_PATH}")
    try:
        if INPUT_DATASET_PATH.endswith(".jsonl"):
            with open(INPUT_DATASET_PATH, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        input_data.append(json.loads(line))
                    except json.JSONDecodeError as je:
                        print(f"Error decoding JSON on line {line_num+1} in {INPUT_DATASET_PATH}: {je}")
                        # Decide whether to skip this line or stop
        elif INPUT_DATASET_PATH.endswith(".json"):
            with open(INPUT_DATASET_PATH, 'r', encoding='utf-8') as f:
                input_data = json.load(f) # Assumes a list of objects
        else:
            print(f"Error: Unsupported input file format: {INPUT_DATASET_PATH}. Please use .json or .jsonl")
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
        original_solution = item.get("solution") # Assuming this key exists as per your example

        if not original_problem_text:
            print(f"  Warning: Skipping item {i+1} due to missing 'problem' field.")
            continue
        # 'solution' might be optional for hint generation, but required for the output structure
        if original_solution is None: # Check for None explicitly if empty string is valid
             print(f"  Warning: Item {i+1} has a missing or null 'solution' field. It will be null in output.")


        print(f"  Original Problem (first 100 chars): {original_problem_text[:100].replace(os.linesep, ' ')}...")

        generated_hint = generate_hint_for_problem(original_problem_text)

        if generated_hint:
            print(f"  Generated Hint (first 100 chars): {generated_hint[:100].replace(os.linesep, ' ')}...")
            new_datapoint = {
                "question": original_problem_text,
                "hint": generated_hint,
                "solution": original_solution # Will be None if original_solution was None
            }
            # You might want to include other fields from the original item if needed
            # For example:
            # new_datapoint["domain"] = item.get("domain")
            # new_datapoint["difficulty"] = item.get("difficulty")
            # new_datapoint["answer"] = item.get("answer") # The short final answer
            # new_datapoint["source"] = item.get("source")
            processed_data.append(new_datapoint)
        else:
            print(f"  Failed to generate hint for problem {i+1}. This problem will not be included in the output.")
            failed_to_get_hint_count += 1
            # If you want to include it anyway with a null/empty hint:
            # new_datapoint = {
            #     "question": original_problem_text,
            #     "hint": None,
            #     "solution": original_solution
            # }
            # processed_data.append(new_datapoint)


    # Save the new dataset
    # Writing as JSONL (one JSON object per line) is good for large datasets
    try:
        with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as f_out:
            for entry in processed_data:
                json.dump(entry, f_out, ensure_ascii=False) # ensure_ascii=False for better unicode handling
                f_out.write('\n')
        print(f"\nSuccessfully generated hints for {len(processed_data)} problems.")
        if failed_to_get_hint_count > 0:
            print(f"Failed to generate hints for {failed_to_get_hint_count} problems.")
        print(f"New dataset saved to {OUTPUT_DATASET_PATH}")
    except IOError as e:
        print(f"Error writing output file {OUTPUT_DATASET_PATH}: {e}")


if __name__ == "__main__":
    # litellm.set_verbose=True # Uncomment for more detailed litellm debugging
    main()