# MLLM-Uncertainty

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

- **YAML configuration files** to store experimental parameters.
- **wandb integration** for experiment tracking and simple data visualization.



## File Structure

Current directory structure:

```
LLM_Hallu/
│
├── configs/
│   └── experiment_config1.yaml       # Experiment configuration files
│
├── data/
│   └── prompt_templates/             
│       └── ask_templates             
│
├── src/
│   ├── dataset/                      # Dataset loading and processing
│   ├── prompt_engineer/              # Prompt engineering
│   ├── models/                       # Model-related code
│   ├── metrics/                      # Metrics computation
│   ├── uncertainty_measures/         # Uncertainty measures
│   └── utils.py                      # Utility functions (e.g., config loading, logging)
│
├── logs/                             # Experiment logs
│
├── results/                          # Experiment results
│
├── notebooks/                        # Jupyter Notebooks for data analysis and debugging
│
├── README.md                         # Project documentation
└── requirements.txt                  # Python dependencies
```



## Completed Tasks

✅ Framework setup.

✅ Dataset module.

✅ Prompt engineering (few-shot methods).

✅ Model integration.

✅ Model answer generation.

✅ Overall `p_true` methodology, including few-shot design for `p_true`.



## Todo List

- Encapsulate the sample class.
- Compile and implement common metrics for LLM hallucination.
- Implement semantic entropy calculations.
- Develop the Aggregator class.


## Ground Truth/confidence
- Semantic Entropy
https://arxiv.org/abs/2302.09664 https://www.nature.com/articles/s41586-024-07421-0

Use the Semantic Entropy to measure the confidence of llm output










