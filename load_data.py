from datasets import load_dataset

# Load the dataset
print("Loading Omni-MATH dataset...")
omni_math_dataset = load_dataset("KbsdJames/Omni-MATH")
print("Dataset loaded successfully!")

# The dataset is a DatasetDict. It only has a 'test' split based on your output.
print("\nAvailable splits:", omni_math_dataset.keys())

# Access the 'test' split
test_split = omni_math_dataset['test']

print(f"\nInformation about the 'test' split:")
print(test_split)
print(f"Features available: {test_split.features}")


# Let's look at the first few examples from the 'test' split
print(f"\nFirst 3 examples from the 'test' split:")
for i in range(3):
    print(f"\n--- Example {i+1} (Index: {i}) ---") # Using index as a temporary ID
    example = test_split[i]
    # print(f"Original Index/ID (if available from source): Not directly available as 'id' field") # Or potentially example['source'] if it's unique
    print(f"Domain: {example['domain']}")         # Mapped from 'topic'
    print(f"Difficulty: {example['difficulty']}") # Mapped from 'level'
    print(f"Problem: {example['problem']}")
    print(f"Detailed Solution: {example['solution']}")
    print(f"Final Answer: {example['answer']}")   # Mapped from 'final_answer'
    print(f"Source: {example['source']}")
    # 'type' field from my original example doesn't have a direct match here.

