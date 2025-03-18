# 🐝 BeesUp-LLM-Framework
_Codebase for LLM-based tools in the BeesUp project_


## ✨ Key Features
- ``LLMPipeline``: An easy-to-use pipeline for loading models, interacting with them, and streaming tokens in real time.
- ``EMBPipeline``: Same thing for embedding models.
- ``RAGPipeline``: Enables Retrieval-Augmented Generation for enhanced context-aware answers.
- ``FinetuningExperiment``: A streamlined setup for parameter-efficient fine-tuning of quantized LLMs on medium-sized GPUs (48 GB), producing reusable LoRA adapters.


### 📊 Structured Data Extraction from Wildbee Observation Reports

- Custom evaluation metrics that compare LLM-generated JSON outputs to ground-truth labels using fuzzy string matching, converting results into standard precision, recall, and F1 scores, and finally calculating an overall extraction score (S_extract).

- ``ExtractionPipeline``: Takes one or multiple report passages as input and returns structured data on the wildbee observations. If labeled data is provided, automatic evaluation is included.

- ``ExtractionExperiment``: Handles fine-tuning and evaluation of an LLM on labeled extraction datasets.


### 💉 Structure-Aware Knowledge Injection via LLM Fine-Tuning

- ``Taxomizer``: Automatically organizes unstructured knowledge chunks into a clear taxonomy (table of contents or tree structure) using hierarchical clustering and smart tree-cutting methods, while generating meaningful headers for each cluster.

- Evaluator classes ``MCQEvaluator``, ``QDQEvaluator``, and ``FFQEvaluator``: Used to assess an LLM’s knowledge and factual accuracy before and after knowledge injection.

- ``InjectionExperiment``: Injects knowledge into the LLM and tracks improvements over epochs with detailed evaluations.


## 🧩 Dependencies & Prequesites

1. **🐍 Python Environment:** Specified in provided `beesup_environment.yaml`, Note there's a second environment `beesup_nvembed_environment.yaml` neccessary to run [nvidia/NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2). Experiments will not work in this environment.

2. **🐳 Docker Image:** shafi.tu-ilmenau.de:30500/project-conda:latest


## 📦 Installation
1. **Clone this repository:**
```bash
git clone https://gitlab.tu-ilmenau.de/mase4201/beesup-llm-framework.git
```

2. **Set up the Conda environment:**
```bash
conda env create -f beesup_environment.yaml
conda activate beesup
```

3. **Install:**
``` bash
cd beesup_llm
pip install -e .
```

