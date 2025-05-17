from litellm import completion
import os

# Set TogetherAI API key
os.environ["TOGETHERAI_API_KEY"] = "4e8fd7f0826fb9dbaefcbaa1e3788ca9cad3614c17f88989087c6e3a38bafb00"

# Compose your prompt with hints
question = "Compute the value of \\(\\sqrt{105^3 - 104^3}\\), given that it is a positive integer."
hint_prompt = f"""
Here is a math question. Give step-by-step hints to guide a student to the solution, without giving away the final answer.

Question:
{question}

Hints:
"""

# Call the TogetherAI model with the provider prefixed in the model name
response = completion(
    model="together_ai/meta-llama/Meta-Llama-3-70B-Instruct-Turbo", # Corrected: added "together_ai/" prefix
    # provider="togetherai",  # This is now optional and can often be removed if prefixed
    messages=[{"role": "user", "content": hint_prompt}]
)

# Print the generated hints
# The response object might be a ModelResponse object, so accessing via attributes is common
# but dictionary-style access should also work if it's emulating a dict.
# Let's try the attribute style first, which is more idiomatic for litellm >= v1.0
try:
    print(response.choices[0].message.content)
except (TypeError, AttributeError): # Fallback for older litellm or if it's still a dict
    print(response["choices"][0]["message"]["content"])