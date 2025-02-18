from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import argparse
import torch
from datasets import load_dataset
import coverage
from io import StringIO
import tempfile
import os
import pytest
import csv

def main(FLAGS):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model fetch
    if(FLAGS.model == 1):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    else:
        model_name = "meta-llama/Meta-Llama-3.1-8B"

    if(FLAGS.model != 0):
        print("model choosen: ", model_name)
    
        # Load tokenizer with padding setup
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Fix for the warning

        # Load model with GPU optimization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        ).to(device)
        
        human_eval(model, tokenizer, device)
    else:
        print("Getting intital results")
        data = load_dataset("openai_humaneval")
        coverage_results = []
        
        # Iterate over the test examples
        for idx, example in enumerate(data['test']):
            formatted_code = format_code(example)
            # Run coverage measurement on the formatted code
            line_coverage = get_coverage(formatted_code)
            # Use the provided task_id if available, else use the index
            task_id = example.get("task_id", idx)
            coverage_results.append({"task_id": task_id, "line_coverage": line_coverage})
        
        # Write the coverage results to a CSV file
        csv_file = "coverage_results_no_model_baseline.csv"
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["task_id", "line_coverage"])
            writer.writeheader()
            writer.writerows(coverage_results)
        print(f"Coverage results written to {csv_file}")

    
    '''# Tokenize input
    inputs = tokenizer(input_query, return_tensors="pt").to(device)

    # Generate with optimized parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,  # Prevent infinite generation
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id  # Explicitly set to avoid warning
        )

    # Decode and print
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nInputted Query:\n", input_query)
    print("\nGenerated Response:\n", output_text)'''
 

def human_eval(model, tokenizer, device):
    print("datasets: Human Eval") # 164 python programming problems
    data = load_dataset("openai_humaneval")
    
    example = data["test"][0] # only test in humaneval
    # print("\nExample Entry:")
    # print(f"Task ID: {example['task_id']}")
    # print(f"Prompt:\n{example['prompt']}")
    # print(f"Canonical Solution:\n{example['canonical_solution']}")
    print(f"Test:\n{example['test']}")
    # print(f"Entry Point: {example['entry_point']}")

    query = f'Below is a Canonical Solution to a python programming problem. I need to to create a function of assert statements to test the code for the given Canonical Solution. Canonical Solution code:\n {example['canonical_solution']}'
    # Tokenize the query
    inputs = tokenizer(query, return_tensors="pt")
    print("inputs: ", inputs)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Feed the tokens into the model. Saving the output.
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    # Print the output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nGenerated Test Code:\n", output_text)


#Measures the coverage of the tests
def get_coverage(code):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as temp_file:
        temp_file_name = temp_file.name
        temp_file.write(code)

    # Run coverage.py on the temporary file
    cov = coverage.Coverage(branch=True)
    cov.start()
    pytest.main([temp_file_name]) # Run pytest on the temp file 
    # pytest.main(['test.py'])
    cov.stop()

    # Capture the coverage report output in a StringIO object
    report_output = StringIO()
    total_coverage = cov.report(file=report_output) # line coverage

    # Clean up the temporary file and return 
    os.remove(temp_file_name)
    return total_coverage # line coverage


def format_code(example):
    lines = example['prompt'].split('\n')
    header = ''
    
    for line in lines:
        header += line
        header += '\n'
        if line.lstrip().startswith("def "):
            # get function name
            new_string = line.split('(')[0]
            function = new_string[3:].lstrip()   
            break
    
    base = example['canonical_solution'] 

    test_funct = 'def test_' + function  + '():' + '\n\t' + 'check(' + function + ')'
    test = example['test'] + '\n' + test_funct
    
    return header + base + test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Query')
    
    parser.add_argument('--dataset', type=str,  help='Which dataset do you want to use ') # not needed anymore
    parser.add_argument('--model', type=int, default=0, help='0 for None/Testing, 1 for DeepSeek-R1-Distill-Llama-8B, else for Meta-Llama-3.1-8B')
    
    FLAGS = parser.parse_args()
    main(FLAGS)
