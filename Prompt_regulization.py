from openai import OpenAI

def correct_text(text):

    client = OpenAI(
        api_key="your key", 
        base_url="your model"
    )
    
    meta_prompt = """You are a meticulous assistant specializing in refining and standardizing biomedical image analysis prompts. Your user is preparing a list of text prompts for the BiomedParse vision foundation model, which requires precise, context-aware inputs.

Your task is to correct, refine, and normalize a list of "dirty" prompts I will provide.

Core Requirements
You must address two types of issues:

1. Linguistic Error Correction: You must find and fix all errors in each individual prompt, including but not limited to the following:
   - Spelling Mistakes
   - Letter Transposition
   - Missing Words/Characters
   - Repeated Words

2. Contextual Standardization: The BiomedParse model performs best when prompts are standardized. You must normalize every prompt to follow this strict format:

   [Target] in [Anatomical Site] [Modality]

To do this, you must first infer the global context (the default Modality and Site) from the entire list of prompts.

Step-by-Step Process
Before providing the final answer, you must perform the following internal reasoning steps:

Step 1: Analyze Global Context
Read the complete list of prompts I provide. Infer the most likely Modality and Anatomical Site that applies to the entire batch.

Step 2: State Your Context
I have analyzed the list. The inferred global context is:
Modality = [Your Inferred Modality]
Site = [Your Inferred Site]

Step 3: Iterate and Refine
Go through each prompt from the original list one by one.

- Analyze: The original prompt is: [original_prompt]
- Correct: The linguistic correction is: [corrected_prompt]
- Standardize: The prompt is missing context. Applying the global context, the final standardized prompt is: [standardized_prompt]
  (if the corrected prompt is liver and context is Abdomen CT, the final prompt is liver in abdomen CT)
- Exception: If a prompt already specifies its own context, like: tumor in brain MRI, respect that context and only correct its linguistic errors.

Final Output Format
Your final response to me MUST ONLY contain the list of fully corrected and standardized prompts.

- Do NOT include your internal Step-by-Step Thinking Process in the final output.
- Separate each standardized prompt with [SEP].

Example Output:
liver in abdomen CT[SEP]right kidney in abdomen CT[SEP]pancreas in abdomen CT[SEP]tumor in brain MRI

Task Starts Now
Here is the list of prompts for you to correct and normalize. Note the subprompt is also separated by [SEP]"""
    
    response = client.chat.completions.create(
        model="Qwen3-VL-8B-Instruct",
        messages=[
            {
                'role': 'user', 
                'content': f"{meta_prompt}\n\n{text}"
            }
        ],
        stream=True
    )
    
    corrected_text = ""
    
    for chunk in response:
        if not chunk.choices:
            continue

        if chunk.choices[0].delta.content:
            corrected_text += chunk.choices[0].delta.content
            
    return corrected_text