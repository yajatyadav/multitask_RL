# scripts/precompute_bert_embeddings.py
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'multitask_RL/libero'))
from libero.libero import benchmark
NUM_UNIQUE_LIBERO_TASKS = 112
def get_all_libero_languages():
    libero_benchmark_dict = benchmark.get_benchmark_dict()
    all_libero_languages = set()
    benchmarks = sorted(["libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10"]) # sorted() to retain same order
    for benchmark_name in benchmarks:
        suite = libero_benchmark_dict[benchmark_name]()
        num_tasks = suite.get_num_tasks()
        for i in range(num_tasks):
            task = suite.get_task(i)
            task_language = task.language
            if task_language not in all_libero_languages:
                all_libero_languages.add(task_language)
            else:
                pass
                # print(f"Duplicate language: {task_language}, found in task {benchmark.get_task_names()[i]} of benchmark {benchmark_name}")
    # once set constructed, sort all keys alphabetically before assinging one-hot label, will ensure consistent ordering
    all_libero_languages = sorted(all_libero_languages)
    all_libero_languages = {language: i for i, language in enumerate(all_libero_languages)}
    assert len(all_libero_languages) == NUM_UNIQUE_LIBERO_TASKS, f"Expected {NUM_UNIQUE_LIBERO_TASKS} unique libero tasks, but found {len(all_libero_languages)}"
    return all_libero_languages

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

def standardize_string(s):
    return s.lower().replace("_", " ")

embeddings = {}
all_libero_languages = get_all_libero_languages()
for task in all_libero_languages:
    key = standardize_string(task)
    inputs = tokenizer(key, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings[key] = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

np.save('embeddings/libero_bert_embeddings.npy', embeddings)
print(f"Saved {len(embeddings)} embeddings")

