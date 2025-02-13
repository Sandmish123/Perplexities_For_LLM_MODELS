'''
This project is created by Sandeep Mishra, 
Kindly Note I have taken a small portion of set of data from the passage a s the modified txt. 
In modified text I have replaced "the" -------> "teh"
Also note the code is working but for GPT-NEO it is taking a bit time as the model is heavy to dowload.

Thank you for this task .... Have a great day!!

'''


import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import functional as F
import pandas as pd 
import threading

# The below function is for the GPT 2 
def load_model_and_tokenizer_for_gpt2(model_name="gpt2"):
    """Load a pre-trained model and tokenizer from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# The below funtion is for GPT-NEO

def load_model_and_tokenizer_for_gpt_neo(model_name="EleutherAI/gpt-neo-1.3B"):
    """Load a pre-trained GPT-Neo model and tokenizer from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def load_wikitext2(split="test"):
    """Load the Wikitext-2 dataset."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    dataset = [text.strip() for text in dataset["text"] if text.strip() and len(text.strip()) > 10]
    # print("\nDataset :: \n",dataset[:100])
    return dataset[:10]  # Using a subset for efficiency

def compute_perplexity(model, tokenizer, text_samples):
    """Compute perplexity for given text samples using a pre-trained model."""
    perplexities = []
    for text in text_samples:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        # print("Tokens:", inputs)
        # Check if input_ids are empty
        if inputs["input_ids"].numel() == 0:
            print(f"Warning: Empty input for text: {text}")
            perplexities.append(float("inf"))  # Assign a high perplexity to invalid inputs
            continue
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        perplexity = torch.exp(loss).item()
        perplexities.append(perplexity)
    return perplexities

def compute_perplexity_for_neo(model, tokenizer, text_samples):
    """Compute perplexity for given text samples using a pre-trained model."""
    perplexities = []
    for text in text_samples:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        # print("Tokens:", inputs)
        # Check if input_ids are empty
        if inputs["input_ids"].numel() == 0:
            print(f"Warning: Empty input for text: {text}")
            perplexities.append(float("inf"))  # Assign a high perplexity to invalid inputs
            continue
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        perplexity = torch.exp(loss).item()
        perplexities.append(perplexity)
    return perplexities

def plot_perplexity(perplexities, title="Perplexity Distribution"):
    """Plot a histogram of perplexity scores."""
    plt.figure(figsize=(7, 5))
    plt.hist(perplexities, bins=10, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel("Perplexity")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()


def modified_perplexity():
    # Read modified samples from the file
    with open("modified_text.txt", "r", encoding="utf-8") as file:
        modified_samples = [text.strip() for text in file.readlines() if text.strip()]  # Remove empty lines

    if not modified_samples:
        print("Error: No valid modified text samples found.")
        return

    # Compute perplexity for the modified samples
    modified_perplexities = compute_perplexity(model, tokenizer, modified_samples)
    print("\nModified_perplexities",modified_perplexities)
    print("\nOnly one sample in the output because modified_text.txt likely contains only one valid line after filtering out empty or invalid lines.\n")
    return modified_perplexities
import matplotlib.pyplot as plt

def plot_perplexity_comparison(original_perplexities, modified_perplexities, old, new):
    """Plot original and modified perplexity distributions in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2 rows, 2 columns

    # Original perplexity plot
    axes[0, 0].hist(original_perplexities, bins=10, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel("Perplexity")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("GPT-2 Perplexity on Wikitext-2")

    # Modified perplexity plot
    axes[0, 1].hist(modified_perplexities, bins=10, alpha=0.7, color='red', edgecolor='black')
    axes[0, 1].set_xlabel("Perplexity")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("GPT-2 Modified Text Perplexity on Wikitext-2")

    # Additional modified perplexity plot (old)
    axes[1, 0].hist(old, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].set_xlabel("Perplexity")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("GPT-NEO Perplexity on Wikitext-2")

    # Additional modified perplexity plot (new)
    axes[1, 1].hist(new, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_xlabel("Perplexity")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("GPT-NEO Modified Text Perplexity on Wikitext-2")

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


def modified_perplexity_for_neo():
    
    # Read modified samples from the file
    with open("modified_text.txt", "r", encoding="utf-8") as file:
        modified_samples = [text.strip() for text in file.readlines() if text.strip()]  # Remove empty lines
    if not modified_samples:
        print("Error: No valid modified text samples found.")
        return
    # Compute perplexity for the modified samples
    modified_perplexities = compute_perplexity(model, tokenizer, modified_samples)
    print("\nModified_perplexities for Neo",modified_perplexities)
    print("\nOnly one sample in the output because modified_text.txt likely contains only one valid line after filtering out empty or invalid lines.\n")
    return modified_perplexities

def preplexity_by_gpt_neo(neo_results):

    tokenizer, model = load_model_and_tokenizer_for_gpt_neo()
    test_samples_for_neo = load_wikitext2()

    # Compute perplexity
    perplexities_for_neo = compute_perplexity_for_neo(model, tokenizer, test_samples_for_neo)
    for i, p in enumerate(perplexities_for_neo):
        print(f"Sample {i+1}: Perplexity By Neo= {p:.2f}")

    # Added a file name modified_text.txt where i have replaced all  " the  " or "The"  into ----------------------> "teh".
    Neo_modified_perplexities=modified_perplexity_for_neo()

    # Append results to the shared list
    neo_results.append(perplexities_for_neo)
    neo_results.append(Neo_modified_perplexities)
    return neo_results 

def run_neo():
    global neo_results
    neo_results = preplexity_by_gpt_neo()  # Assuming it returns (old, new)


if __name__== "__main__":
    # Load model, tokenizer, and dataset
    tokenizer, model = load_model_and_tokenizer_for_gpt2()
    test_samples = load_wikitext2()

    # Compute perplexity
    perplexities = compute_perplexity(model, tokenizer, test_samples)
    for i, p in enumerate(perplexities):
        print(f"Sample {i+1}: Perplexity = {p:.2f}")

    # Added a file name modified_text.txt where i have replaced all  " the " or "The"  into ----------------------> "teh".
    obj=modified_perplexity()

    neo_results = []
    neo_thread=threading.Thread(target=preplexity_by_gpt_neo, args=(neo_results,))
    neo_thread.start()
    neo_thread.join()
    plot_perplexity_comparison(perplexities, obj, neo_results[0], neo_results[1])


