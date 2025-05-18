import litellm
import os
import json
import time

# --- Configuration ---
TOGETHERAI_API_KEY = "4e8fd7f0826fb9dbaefcbaa1e3788ca9cad3614c17f88989087c6e3a38bafb00" 
LLM_MODEL = "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
INPUT_DATASET_PATH = "/Users/ifditahasanorney/Documents/GitHub/eli0.5b/omnimath_100.jsonl"
# OUTPUT_DATASET_PATH will be set dynamically
MAX_RETRIES = 3
RETRY_DELAY = 10 # Initial delay in seconds

os.environ["TOGETHERAI_API_KEY"] = TOGETHERAI_API_KEY

# --- Student Model Descriptions ---
# Define descriptions for the different student models you are targeting
STUDENT_MODEL_PROFILES = {
    "Qwen2.5-0.5B": {
        "description": "A very basic student AI with limited reasoning capabilities (comparable to Qwen2.5-0.5B). It needs extremely simple, atomic steps and very explicit guidance for every single action. It is good at direct calculation if told exactly what to calculate, but struggles with abstraction or inferring multi-step actions from a single hint. Use very simple language.",
        "output_suffix": "qwen0.5b"
    },
    "Llama-3.2-1B": {
        "description": "A student AI with foundational understanding (comparable to Llama-3.2-1B). It can follow sequences of instructions but benefits greatly from clarity and well-defined, manageable steps. A hint can guide through a short sequence of related operations if they are logically connected. Standard mathematical terms are okay if clear.",
        "output_suffix": "llama1b"
    },
    "Llama-3.2-3B": {
        "description": "A reasonably capable student AI (comparable to Llama-3.2-3B). It can follow more complex instructions and understand higher-level concepts. Hints can cover conceptual steps or a few related mathematical operations to encourage efficiency. It is expected to understand and apply appropriate mathematical terminology and formulas if guided.",
        "output_suffix": "llama3b"
    },
    "Generic": { # Your original generic approach
        "description": "a general student.", # Generic description
        "output_suffix": "generic"
    }
}

# --- CHOOSE THE TARGET STUDENT MODEL PROFILE FOR THIS RUN ---
TARGET_STUDENT_PROFILE_KEY = "Qwen2.5-0.5B" # Change this key to generate hints for a different model
# Example: TARGET_STUDENT_PROFILE_KEY = "Qwen2.5-0.5B"
# Example: TARGET_STUDENT_PROFILE_KEY = "Generic"

SELECTED_STUDENT_MODEL_DESCRIPTION = STUDENT_MODEL_PROFILES[TARGET_STUDENT_PROFILE_KEY]["description"]
OUTPUT_DATASET_SUFFIX = STUDENT_MODEL_PROFILES[TARGET_STUDENT_PROFILE_KEY]["output_suffix"]
OUTPUT_DATASET_PATH = f"omnimath_tailored_hints_{OUTPUT_DATASET_SUFFIX}.jsonl"


def generate_hint_for_problem(problem_text, student_model_description_for_prompt):
    """
    Generates hints for a given math problem, tailored for a student model
    with the specified properties.
    """
    # The placeholder {student_model_description_for_prompt} will be filled
    hint_prompt = f"""
You are an expert math tutor. Your primary task is to provide step-by-step hints that would guide a student AI towards the solution of the given math question.

**The student AI you are generating hints for has the following properties:
{student_model_description_for_prompt}**

Please carefully tailor your hints to be perfectly suitable for this specific type of student AI, considering its described capabilities and limitations.

IMPORTANT - Instructions for generating these tailored hints:
1.  Do NOT reveal the final answer or any final numerical/symbolic result of the overall problem in your hints.
2.  Focus on the problem-solving process, key concepts, or intermediate steps, adapting the complexity according to the student AI's properties.
3.  Break down the problem into steps that are appropriately sized for the student AI's capabilities. For less capable AIs, this means very small, atomic steps. For more capable ones, hints can be more consolidated.
4.  Hints must be progressive, leading the student AI logically and clearly.
5.  Keep hints concise and use language appropriate for the described student AI. Ensure clarity above all.

Question:
{problem_text}

Hints (tailored for the student AI with properties: {student_model_description_for_prompt}):
"""
    for attempt in range(MAX_RETRIES):
        try:
            print(f"    Attempting API call for hint (target: {TARGET_STUDENT_PROFILE_KEY}, attempt {attempt + 1}/{MAX_RETRIES})...")
            response = litellm.completion(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": hint_prompt}],
                temperature=0.3, # Lower temperature for more deterministic and focused hints
                max_tokens=500,  # Max tokens for the hint generation
            )
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                print(f"    Warning: Received unexpected response structure: {response}")
                if hasattr(response, 'model_dump_json'): # litellm v1.15.10+
                    print(f"    Full response dump: {response.model_dump_json(indent=2)}")
                elif isinstance(response, dict): # Older litellm or raw dict
                     print(f"    Full response dump: {json.dumps(response, indent=2)}")
                return None
        except litellm.exceptions.RateLimitError as rle:
            print(f"    Rate limit error (attempt {attempt + 1}/{MAX_RETRIES}): {rle}")
            if attempt < MAX_RETRIES - 1:
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
                return None
    return None # Should be unreachable if loop completes, but as a fallback

def main():
    input_data = []
    print(f"Attempting to load dataset from: {INPUT_DATASET_PATH}")
    print(f"Hints will be tailored for a student model with profile: '{TARGET_STUDENT_PROFILE_KEY}'")
    print(f"Output will be saved to: {OUTPUT_DATASET_PATH}")

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
            print(f"Error: Unsupported input file format: {INPUT_DATASET_PATH}. Must be .jsonl or .json")
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
    if not input_data:
        print(f"Error: No data loaded from {INPUT_DATASET_PATH}. Please check the file and its content.")
        return
        
    print(f"Loaded {len(input_data)} problems from {INPUT_DATASET_PATH}")

    processed_data = []
    failed_to_get_hint_count = 0

    for i, item in enumerate(input_data):
        print(f"\nProcessing problem {i+1}/{len(input_data)}...")
        original_problem_text = item.get("problem")
        original_detailed_solution = item.get("solution")
        original_concise_answer = item.get("answer")

        if not original_problem_text:
            print(f"  Warning: Skipping item {i+1} due to missing 'problem' field.")
            continue
        # Warnings for missing solution/answer are fine, but they should still be processed for hints
        if original_detailed_solution is None:
             print(f"  Warning: Item {i+1} has a missing or null 'detailed_solution' (long official solution) field.")
        if original_concise_answer is None:
             print(f"  Warning: Item {i+1} has a missing or null 'answer' (concise final answer) field.")


        print(f"  Original Problem (first 100 chars): {str(original_problem_text)[:100].replace(os.linesep, ' ')}...")

        # Pass the selected student model description to the hint generation function
        generated_hint = generate_hint_for_problem(original_problem_text, SELECTED_STUDENT_MODEL_DESCRIPTION)

        if generated_hint:
            print(f"  Generated Hint (first 100 chars): {generated_hint[:100].replace(os.linesep, ' ')}...")
            new_datapoint = {
                "question": original_problem_text,
                "hint_for_profile": TARGET_STUDENT_PROFILE_KEY, # Record which profile these hints are for
                "hint": generated_hint,
                "detailed_solution": original_detailed_solution,
                "final_answer_gt": original_concise_answer
            }
            processed_data.append(new_datapoint)
        else:
            print(f"  Failed to generate hint for problem {i+1} after {MAX_RETRIES} retries. This problem will not be included in the output.")
            failed_to_get_hint_count += 1
            # Optionally, still add the problem but with a null/empty hint
            # processed_data.append({
            #     "question": original_problem_text,
            #     "hint_for_profile": TARGET_STUDENT_PROFILE_KEY,
            #     "hint": None, # Or "FAILED_TO_GENERATE"
            #     "detailed_solution": original_detailed_solution,
            #     "final_answer_gt": original_concise_answer
            # })


    try:
        with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as f_out:
            for entry in processed_data:
                json.dump(entry, f_out, ensure_ascii=False)
                f_out.write('\n')
        print(f"\nSuccessfully processed {len(input_data)} problems.")
        print(f"Generated hints for {len(processed_data) - (failed_to_get_hint_count if failed_to_get_hint_count else 0)} problems tailored for '{TARGET_STUDENT_PROFILE_KEY}'.")
        if failed_to_get_hint_count > 0:
            print(f"Failed to generate hints for {failed_to_get_hint_count} problems.")
        print(f"New dataset saved to {OUTPUT_DATASET_PATH}")
    except IOError as e:
        print(f"Error writing output file {OUTPUT_DATASET_PATH}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during file writing: {e}")

if __name__ == "__main__":
    main()