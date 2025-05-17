from datasets import load_dataset

# Step 1: Load the dataset (no split specified yet)
print("Loading available splits for OmniMath...")
dataset_dict = load_dataset("KbsdJames/omni-math")

print("\nAvailable splits:")
print(dataset_dict)

# Step 2: Load the 'test' split (only available one)
dataset = dataset_dict["test"]

# Step 3: Print dataset length and features
print(f"\nNumber of examples: {len(dataset)}")
print("Features (keys in each example):")
print(dataset.column_names)

# Step 4: Show a few examples
print("\nSample examples:")
for i in range(3):
    print(f"\nExample {i+1}:")
    print(dataset[i])

