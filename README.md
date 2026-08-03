# Professor Review Rating Prediction

A machine learning and natural language processing project that predicts a professor's **1–5 star rating directly from the text of a student review**.

The project collects real professor reviews from **PlanetTerp** and compares two approaches to text classification:

* Prompting pretrained large language models (LLMs) to predict ratings without task-specific training
* Fine-tuning a transformer model specifically for 5-class review classification

The goal is to explore how effectively the sentiment and language of a written professor review can be mapped to its numerical rating and compare the performance of general-purpose LLMs with a specialized fine-tuned model.

## Project Overview

Student reviews often contain enough sentiment and contextual information to indicate how a student rated a professor. This project investigates whether NLP models can infer a numerical **1–5 star rating** using only the written review.

Each sample contains:

| Feature            | Description                   |
| ------------------ | ----------------------------- |
| `professor`        | Name of the professor         |
| `text`             | Student's written review      |
| `rating`           | Actual rating from 1–5        |
| `predicted_rating` | Rating predicted by the model |

The resulting dataset contains more than **1,100 professor reviews**.

## Approach

The project compares two NLP strategies.

### 1. Prompted Large Language Models

Pretrained causal language models are prompted to infer the rating directly from a review.

The model receives a prompt similar to:

```text
Give a single number of stars, between 1 and 5, that you think this review is assigning.
Only output the number, nothing else.

Review: <student review>

Stars:
```

The generated response is parsed to extract a rating from 1–5.

Models explored include:

* Qwen 0.5B
* Qwen 1.5B
* Mistral 7B

This approach requires **no task-specific model training** and instead relies on the language understanding already learned by the pretrained model.

### 2. Fine-Tuned Transformer Classification

The second approach treats rating prediction as a **5-class text classification problem**.

A pretrained `roberta-large` model is fine-tuned on the collected professor reviews.

Ratings are converted into classification labels:

```text
1 star → class 0
2 stars → class 1
3 stars → class 2
4 stars → class 3
5 stars → class 4
```

The dataset is split using an **80/20 stratified train-validation split** so the rating distribution is preserved across both sets.

Reviews are tokenized using the RoBERTa tokenizer with:

```text
Maximum sequence length: 256 tokens
Training batch size: 8
Validation batch size: 16
Optimizer: AdamW
Learning rate: 2e-5
```

Because the dataset contains significantly more reviews for some ratings than others, **class-weighted cross-entropy loss** is used to reduce the impact of class imbalance.

## Results

### Prompted Mistral 7B

Mistral 7B achieved approximately:

**70% accuracy across 1,123 reviews**

| Rating  | Precision | Recall |   F1 |
| ------- | --------: | -----: | ---: |
| 1 Star  |      0.79 |   0.81 | 0.80 |
| 2 Stars |      0.46 |   0.49 | 0.47 |
| 3 Stars |      0.36 |   0.59 | 0.44 |
| 4 Stars |      0.43 |   0.37 | 0.40 |
| 5 Stars |      0.94 |   0.83 | 0.88 |

The model performs especially well on highly negative **1-star reviews** and highly positive **5-star reviews**, while intermediate ratings are more difficult to distinguish.

### Fine-Tuned RoBERTa

The fine-tuned RoBERTa classifier achieved approximately:

**69% validation accuracy**

| Rating  | Precision | Recall |   F1 |
| ------- | --------: | -----: | ---: |
| 1 Star  |      0.73 |   0.78 | 0.75 |
| 2 Stars |      0.53 |   0.36 | 0.43 |
| 3 Stars |      0.45 |   0.39 | 0.42 |
| 4 Stars |      0.39 |   0.58 | 0.46 |
| 5 Stars |      0.90 |   0.82 | 0.86 |

Despite being substantially smaller than Mistral 7B, the fine-tuned classifier achieves comparable performance by being trained specifically for the rating-prediction task.

## Model Comparison

The experiments show a clear relationship between model capability and rating-prediction performance.

| Model         | Parameters | Approach              | Accuracy |
| ------------- | ---------: | --------------------- | -------: |
| Qwen 0.5B     |       0.5B | Prompted LLM          |     ~18% |
| Qwen 1.5B     |       1.5B | Prompted LLM          |     ~48% |
| Mistral 7B    |         7B | Prompted LLM          |     ~70% |
| RoBERTa-large |      ~0.4B | Fine-tuned classifier |     ~69% |

One of the main observations is that **fine-tuning a smaller model can produce performance comparable to a much larger prompted language model**.

The experiments also show that rating prediction is easiest at the extremes. Both Mistral and RoBERTa classify 1-star and 5-star reviews considerably better than ratings in the 2–4 range, where sentiment is often more ambiguous.

## Evaluation

Model performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrices

Confusion matrices are used to analyze which ratings models most frequently confuse with one another.

Training loss and validation accuracy are also tracked across epochs for the fine-tuned transformer to visualize model learning and performance over time.

## Tech Stack

**Language**

* Python

**Machine Learning / NLP**

* PyTorch
* Hugging Face Transformers
* scikit-learn

**Models**

* Mistral 7B
* Qwen
* RoBERTa-large

**Data**

* PlanetTerp API
* pandas
* NumPy

**Visualization**

* Matplotlib
* Seaborn


## Running the Project

Install the primary dependencies:

```bash
pip install planetterp
pip install transformers
pip install torch
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
```

Then run the notebook:

```text
planetterp_project.ipynb
```

The notebook will:

1. Retrieve professor reviews from PlanetTerp
2. Build the review dataset
3. Run prompted LLM rating predictions
4. Evaluate predictions using classification metrics
5. Fine-tune a RoBERTa classifier
6. Evaluate the fine-tuned model
7. Generate confusion matrices and model comparison visualizations

A CUDA-capable GPU is recommended for running and fine-tuning the transformer models.

## Key Takeaways

This project demonstrates how transformer-based NLP models can infer structured numerical ratings from unstructured student review text.

The experiments highlight three main findings:

* Larger prompted LLMs perform substantially better than smaller prompted models on rating prediction.
* A task-specific fine-tuned transformer can achieve performance comparable to a much larger general-purpose LLM.
* Extreme ratings are easier to identify than intermediate ratings, suggesting that mixed or moderate sentiment remains a significant challenge for review-rating prediction.

## Acknowledgments

Review data is collected using the PlanetTerp Python package and originates from PlanetTerp, a student-built platform for University of Maryland course and professor information.
