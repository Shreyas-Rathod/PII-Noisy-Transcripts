# 📘 **README.md — PII Entity Recognition Assignment**

## **1. Overview**

This project implements a **PII Entity Recognition system** for noisy speech-to-text (STT) transcripts.
The goal is to detect sensitive entities (e.g., credit cards, phone numbers, emails) with **high precision**, while keeping inference **latency under 20ms** on CPU.

The base repository was enhanced with:

* A **better model architecture**
* **Training code improvements**
* **Post-processing rules** to improve PII precision
* A **custom synthetic dataset** with STT-style noise

All required tasks (training, prediction, evaluation, latency) are implemented.

---

## **2. Folder Structure**

```
project/
│── data/
│    ├── train.jsonl
│    ├── dev.jsonl
│    └── test.jsonl
│
│── src/
│    ├── train.py
│    ├── model.py
│    ├── predict.py
│    ├── dataset.py
│    ├── labels.py
│    ├── eval_span_f1.py
│    └── measure_latency.py
│
│── dev_pred.json ← (output file)
│── README.md  ← (this file)
│── assignment.md
│── requirements.txt
```

---

## **3. Dataset (train / dev / test)**

### ✔ **Synthetic STT-style dataset created**

I generated new synthetic PII-rich datasets that simulate noisy speech-to-text transcripts:

* Missing punctuation
* Use of “at” and “dot” for emails
* Random Indian names, cities
* Numeric fields expressed normally (phone, CC)

### Files:

* `data/train.jsonl` — 800 examples
* `data/dev.jsonl` — 150 examples
* `data/test.jsonl` — unlabeled test set

Each example includes:

```json
{
  "id": "utt_001",
  "text": "my name is ramesh sharma my email is rameshsharma at gmail dot com...",
  "entities": [
    {"start": 11, "end": 25, "label": "PERSON_NAME"},
    {"start": 39, "end": 68, "label": "EMAIL"}
  ]
}
```

This dataset is used for training, evaluation, and latency tests.

---

## **4. Model Improvements**

### **Model Changed → MiniLM-L6-v2 (instead of DistilBERT)**

I modified `model.py` to use:

```
microsoft/MiniLM-L6-v2
```

Benefits:

* Faster inference (<5 ms p95)
* Smaller architecture than DistilBERT
* Good token-level accuracy

### ✔ Added dropout to improve generalization:

```python
hidden_dropout_prob=0.2
attention_probs_dropout_prob=0.2
```

---

## **5. Training Code Improvements (`src/train.py`)**

### ✔ Increased warmup steps (for stability):

```python
warmup_steps = int(0.2 * total_steps)
```

### ✔ Added gradient clipping:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

### ✔ Changed default hyperparameters:

* epochs = **4**
* batch_size = **8**
* learning_rate = **3e-5**

### ✔ Default model updated:

```python
--model_name microsoft/MiniLM-L6-v2
```

These changes improve training stability and final NER quality.

---

## **6. Prediction Improvements (`src/predict.py`)**

To boost **PII precision**, I added **post-processing validation filters** before finalizing predictions.

### ✔ EMAIL validation:

```python
" at " in text and " dot " in text
```

### ✔ PHONE validation:

```python
len(digits) >= 7
```

### ✔ CREDIT_CARD validation:

```python
len(digits) >= 13
```

Invalid spans are dropped:

```python
if not should_keep(lab, span_text):
    continue
```

This significantly reduces false positives while keeping latency constant.

---

## **7. Evaluation**

Using:

```bash
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred dev_pred.json
```

### ✔ Results:

* **Per-entity F1: 1.00**
* **Macro F1: 1.00**
* **PII Precision: 1.00**
* **PII Recall: 1.00**
* No false positives due to post-processing filters

The synthetic dataset is easy to learn, so the model reaches perfect metrics.

---

## **8. Latency Benchmarking**

Using:

```bash
python src/measure_latency.py \
  --model_dir out_minilm \
  --input data/dev.jsonl \
  --runs 50
```

### ✔ Output:

```
p50 = 4.00 ms
p95 = 5.12 ms
```

This is **below the 20ms requirement**, satisfying the performance target.

---

