from datasets import load_dataset
import os

# Paths
json_output_path = "/iliad/u/jubayer/omnimath_20.json"
hf_output_dir = "/iliad/u/jubayer/omnimath_20"

# Load the 'test' split of the OmniMath dataset
print("Loading OmniMath (test split)...")
dataset = load_dataset("KbsdJames/omni-math", split="test")

# Select 20 examples (shuffle first for diversity)
print("Shuffling and selecting 20 examples...")
subset = dataset.shuffle(seed=42).select(range(20))

# Save as JSON
print(f"Saving subset to {json_output_path}...")
subset.to_json(json_output_path, orient="records", lines=True)

# Save as Hugging Face dataset format
print(f"Saving subset to {hf_output_dir}...")
subset.save_to_disk(hf_output_dir)

print("✅ Done! 20-example subset created.")

