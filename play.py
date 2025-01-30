from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch
from datasets import load_dataset

def main(FLAGS):
    # Get FLAGS
    #input_query = FLAGS.query
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model fetch
    #model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    
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
    print("datasets: Human Eval")
    data = load_dataset("openai_humaneval")
    print(data)
    
    example = data["test"][0]
    print("\nExample Entry:")
    print(f"Task ID: {example['task_id']}")
    print(f"Prompt:\n{example['prompt']}")
    print(f"Canonical Solution:\n{example['canonical_solution']}")
    print(f"Test:\n{example['test']}")
    print(f"Entry Point: {example['entry_point']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepSeek Model Query')
    
    parser.add_argument('--query', type=str,  help='Input query for the model')
    parser.add_argument('--dataset', type=str,  help='Which dataset do you want to use ')
    
    FLAGS = parser.parse_args()
    main(FLAGS)
