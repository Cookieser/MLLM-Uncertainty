# LLM Hallucination 

A framework to summarize metrics and ground truth implementations related to LLM hallucination, with experiment management powered by Weights & Biases (wandb).

This framework modularizes and encapsulates various components (detailed below) to facilitate the addition of experimental metrics and streamline experimentation. This will be helpful to build our system, and facilitate the follow-up comprehensive and extensive experiment.

The framework builds upon an existing semantic entropy-based implementation, which provided inspiration and foundational concepts. However, the original code was focused on its specific methodology and lacked an overall modular design, leading to reduced readability and limited scalability.

Take Care: The following structure reflects the current design, but adjustments may be made as needed during implementation to achieve maximum modularity and decoupling.



## Module Overview

| **Module**        | **Dataset**              | **Prompt Engineer**                                          | **Model**                                                    | **Sample**                                                   | **Metrics**                                                  | **Aggregator**                                               |
| ----------------- | ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Function**      | Load and format datasets | Generate prompts based on dataset combinations, including zero-shot and few-shot methods. Key functionalities include managing `unused_indices` and correctly sampling items to create prompts. | Use `HuggingfaceModel`, inheriting from a `BaseModel` class to connect with LLMs, handle requests, and manage internal states. This includes fine details like truncating output when repetition occurs. | Support various sampling methods: few-shot sampling, synonym-based prompt transformations, induced sampling, visualization of sampling processes, and data logging. | Implement specific metrics for quantification.               | Combine metrics and sample results for analysis, e.g., semantic entropy. |
| **Input**         | `dataset_name`, `seed`   | Formatted `dataset_item` + `prompt_template` (optional) + prompt generation settings (e.g., enable few-shot/brief mode/tags, use context). | `model_name`, `max_new_tokens`, `stop_sequences`.            | `model`, `PromptGenerator`, `method`.                        | Data required for specific metrics.                          | `metric_name`, `sample_result`.                              |
| **Output**        | Formatted dataset        | Generated prompts                                            | Model-generated responses.                                   | `sample_result.pkl`.                                         | Corresponding quantitative metrics.                          | Aggregate results for specific metrics.                      |
| **Function**      | `load_data()`            | `generate_template_prompt_by_id(id)`, `generate_template_prompts_by_count(count)`, `generate_template_prompts_from_indices(indices)`, `get_unused_indices()`, `construct_fewshot_prompt_by_nums(shot_num)`. | `predict(self, input_data, temperature=1.0, device='cuda', return_full=False)`, `get_p_true(self, input_data, answer="A")`. | `few_shot_sample(model, few_shot_prompt, promptgenerator)`, `simple_sample(model, promptgenerator)`, `similar_sample(model, promptgenerator)`. | `compute_acc(xx, xx)`, `compute_metrics2(xx)`, `compute_metrics3(xx)`. | `aggregator_semantic_entropy(xx)`.                           |
| **Extensibility** | Add new datasets         | Add more Prompt Engineering methods.                         | Support additional LLMs, parameters, and local models.       | Introduce new sampling methods.                              | Implement additional metrics.                                | Add more aggregation methods.                                |



### Features

- **wandb integration** for experiment tracking and simple data visualization.



## File Structure

Current directory structure:

```
LLM_Hallu/
│
├── environment_export.yaml
├── environment.yaml
├── LICENSE
├── notebooks
│   ├── example_evaluation.ipynb
├── README.md
└── semantic_uncertainty
    ├── analyze_results.py
    ├── compute_uncertainty_measures.py
    ├── generate_answers_llm.py
    ├── generate_answers_mllm_plus.py
    ├── generate_answers_mllm.py
    ├── uncertainty
    │   ├── data
    │   │   ├── data_utils.py
    │   │   └── Questions.csv
    │   ├── __init__.py
    │   ├── models
    │   │   ├── base_model.py
    │   │   ├── huggingface_models.py
    │   │   ├── __init__.py
    │   │   ├── mllm_model.py
    │   ├── uncertainty_measures
    │   │   ├── p_ik.py
    │   │   ├── p_true.py
    │   │   └── semantic_entropy.py
    │   └── utils
    │       ├── eval_utils.py
    │       ├── openai.py
    └──     └── utils.py
```



## Demo

Change the image path in `semantic_uncertainty/data/data_utils`
```
base_image_path = "/work/images/images"
original_image_path = "/work/images/images/mmvp"
file_path = "/home/yw699/codes/MLLM-hallu/semantic_uncertainty/uncertainty/data/Questions.csv" 
```

```
conda-env update -f environment.yaml
conda activate semantic_uncertainty
```

```
python semantic_uncertainty/generate_answers_llm.py --model_name=Llama-2-7b-chat --dataset=trivia_qa
python semantic_uncertainty/generate_answers_mllm.py --dataset=mmvp
python semantic_uncertainty/generate_answers_mllm_plus.py --dataset=mmvp
```



## Completed Tasks

✅ Framework setup.

✅ LLM: Dataset module.

✅ LLM: Prompt engineering (few-shot methods).

✅ LLM: Model integration.

✅ LLM: Model answer generation.

✅ LLM: Overall `p_true` methodology, including few-shot design for `p_true`.

✅ MLLM: Add MLLM

✅MLLM: Finish the metric of the accuracy

✅ MLLM: Handle with MMVP dataset

✅ MLLM: Finish MLLM few-shot 

✅ MLLM: Finish MLLM  `p_true` method, including few-shot design for `p_true`.

✅ MLLM: Finish MLLM  generation method

✅ MLLM: Finish MLLM  generation plus method through our transformed images



## To do

- README
- polish the structure and encapsulate various components
- add some dataset 
- add some models
- do many expriments
- add some method: RepE

## Ground Truth/confidence

- Semantic Entropy
https://arxiv.org/abs/2302.09664 https://www.nature.com/articles/s41586-024-07421-0

Use the Semantic Entropy to measure the confidence of llm output


















