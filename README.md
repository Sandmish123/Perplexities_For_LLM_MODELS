# Perplexities_For_LLM_MODELS
# GPT-2 and GPT-Neo Perplexity Analysis

## 📌 Project Overview
This project, created by **Sandeep Mishra**, evaluates the perplexity of **GPT-2** and **GPT-Neo** models on the **Wikitext-2** dataset. It also measures the impact of minor text modifications, such as replacing all instances of "the" with "teh." The results are visualized using histograms for comparison.

## 🔍 What is Perplexity?
Perplexity is a measure of how well a probabilistic model predicts a sample. In the context of language models like GPT-2 and GPT-Neo, lower perplexity indicates that the model is more confident in its predictions, meaning it generates more fluent and coherent text. A high perplexity score suggests that the model finds the text difficult to predict.

Mathematically, perplexity (PPL) is defined as:
\[ PPL = 2^{H} \]
where \( H \) is the cross-entropy of the model's predictions.

## 🚀 Features
- **Supports GPT-2 and GPT-Neo models** for perplexity analysis.
- **Loads the Wikitext-2 dataset** for testing.
- **Computes perplexity scores** using pre-trained models.
- **Compares original vs. modified text perplexity** to assess impact.
- **Executes GPT-Neo evaluation** similarly to GPT-2.
- **Generates visualizations** comparing perplexity distributions.

---

## 📦 Dependencies
This project relies on the following Python libraries:
- `torch`
- `numpy`
- `matplotlib`
- `transformers`
- `datasets`
- `pandas`

Install dependencies using:
```bash
pip install torch numpy matplotlib transformers datasets pandas
```

---

## 🏗 Code Structure

### 1️⃣ Loading Pre-trained Models
- **`load_model_and_tokenizer_for_gpt2(model_name="gpt2")`**
  - Loads GPT-2 model and tokenizer from Hugging Face.
  - Returns the tokenizer and model.

- **`load_model_and_tokenizer_for_gpt_neo(model_name="EleutherAI/gpt-neo-1.3B")`**
  - Loads GPT-Neo model and tokenizer from Hugging Face.
  - Returns the tokenizer and model.

### 2️⃣ Loading Dataset
- **`load_wikitext2(split="test")`**
  - Loads the Wikitext-2 dataset.
  - Cleans and filters the dataset.
  - Returns the top 10 samples.

### 3️⃣ Computing Perplexity
- **`compute_perplexity(model, tokenizer, text_samples)`**
  - Computes perplexity scores for given text samples.
  - Returns a list of perplexity scores.

- **`compute_perplexity_for_neo(model_neo, tokenizer_for_neo, text_samples_for_neo)`**
  - Similar to `compute_perplexity`, but optimized for GPT-Neo.

### 4️⃣ Perplexity for Modified Text
- **`compute_perplexity_for_modified_text(model, tokenizer, text_samples)`**
  - Computes perplexity using the provided model and tokenizer.
  - Returns modified perplexity scores.

- Similar function for GPT-Neo.

### 5️⃣ Visualization
- **`plot_perplexity(perplexities, title="Perplexity Distribution")`**
  - Plots a histogram of perplexity scores.
  - Compares original vs. modified perplexity for both models in a **2x2 grid**.

- **`plot_perplexity_comparison(original_perplexities, modified_perplexities)`**
  - Similar to `plot_perplexity_comparison`, but optimized for GPT-Neo.

---

## 🔄 Workflow Summary
1. Loads GPT-2 model and dataset.
2. Computes perplexity for original text.
3. Computes perplexity for modified text.
4. Runs GPT-Neo perplexity analysis.
5. Repeats the process for comparison.
6. Plots perplexity comparison for both models.

---

## ⚠️ Notes
- **GPT-Neo** takes longer to download and execute due to its size.
- **Modified text analysis** helps understand how minor changes impact perplexity.

---

## 🙌 Acknowledgment
This project was created by **Sandeep Mishra** as part of a research task on language model perplexity evaluation. Thank you for exploring this project! Have a great day! 🎉
