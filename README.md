<h1 align="center">Contextual Integrity in LLMs via Reasoning and Reinforcement Learning</h1>

<div align="center">
<a href='https://arxiv.org/abs/2506.04245'><img src='https://img.shields.io/badge/Paper-Arxiv-red.svg?style=for-the-badge&logo=arxiv&logoColor=white'></a> 
<a href='https://huggingface.co/papers/2506.04245'><img src='https://img.shields.io/badge/Paper-Hugging_Face-yellow.svg?style=for-the-badge&logo=huggingface&logoColor=%23FFD21E'></a>
<a href='LICENSE'><img src='https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge'></a>
</div>

> [!NOTE]
> The description of this repository is still in progress.


- **Training and Evaluation**: In ./verl-supp folder.

- **Data_Generation and PrivacyLens_Evaluation**: In ./posttraining-research-ci-supp folder.


```
└── code
    ├── posttraining-research-ci-supp
    │   ├── components
    │   │   ├── privacylens
    │   │   │   ├── assets
    │   │   │   │   └── overview.png
    │   │   │   ├── component_spec.yaml
    │   │   │   ├── data
    │   │   │   │   ├── extensibility
    │   │   │   │   │   ├── confide_subset.json
    │   │   │   │   │   └── culturebank_subset.json
    │   │   │   │   └── main_data.json
    │   │   │   ├── data_construction
    │   │   │   │   ├── format_trajectory.py
    │   │   │   │   ├── format_vignette_for_trajectory_simulation.py
    │   │   │   │   ├── seed_to_vignette.py
    │   │   │   │   ├── simulate_trajectory.py
    │   │   │   │   └── toolemu
    │   │   │   │       ├── __init__.py
    │   │   │   │       ├── agent_executor_builder.py
    │   │   │   │       ├── agents
    │   │   │   │       │   ├── __init__.py
    │   │   │   │       │   ├── agent_executor.py
    │   │   │   │       │   ├── agent_interface.md
    │   │   │   │       │   ├── virtual_agent_executor.py
    │   │   │   │       │   └── zero_shot_agent_with_toolkit.py
    │   │   │   │       ├── dataloader.py
    │   │   │   │       ├── executors
    │   │   │   │       │   ├── __init__.py
    │   │   │   │       │   ├── func_executor.py
    │   │   │   │       │   └── prompt_executor.py
    │   │   │   │       ├── prompts
    │   │   │   │       │   ├── __init__.py
    │   │   │   │       │   ├── agent
    │   │   │   │       │   │   ├── __init__.py
    │   │   │   │       │   │   ├── agent_naive.py
    │   │   │   │       │   │   ├── agent_privacy_enhanced.py
    │   │   │   │       │   │   └── shared.py
    │   │   │   │       │   ├── globals.py
    │   │   │   │       │   ├── principles.py
    │   │   │   │       │   ├── README.md
    │   │   │   │       │   └── simulator
    │   │   │   │       │       ├── __init__.py
    │   │   │   │       │       ├── privacy_adversarial.py
    │   │   │   │       │       └── shared.py
    │   │   │   │       ├── README.md
    │   │   │   │       ├── tools
    │   │   │   │       │   ├── __init__.py
    │   │   │   │       │   ├── core_virtual_tools.py
    │   │   │   │       │   ├── register.py
    │   │   │   │       │   └── tool_interface.py
    │   │   │   │       └── utils
    │   │   │   │           ├── __init__.py
    │   │   │   │           ├── agent.py
    │   │   │   │           ├── colorful.py
    │   │   │   │           ├── const.py
    │   │   │   │           ├── convertion.py
    │   │   │   │           ├── io.py
    │   │   │   │           ├── langchain_utils.py
    │   │   │   │           ├── llm.py
    │   │   │   │           ├── misc.py
    │   │   │   │           ├── my_typing.py
    │   │   │   │           ├── parallel.py
    │   │   │   │           └── tool.py
    │   │   │   ├── evaluation
    │   │   │   │   ├── evaluate_final_action.py
    │   │   │   │   ├── get_final_action.py
    │   │   │   │   ├── output
    │   │   │   │   │   └── main_gpt4_o.csv
    │   │   │   │   ├── output_action
    │   │   │   │   │   ├── main_deepseek_naive_evaluate_leakage.json
    │   │   │   │   │   ├── main_deepseek_naive_missing.json
    │   │   │   │   │   ├── main_deepseek_naive.csv
    │   │   │   │   │   ├── main_deepseek_privacy_enhanced.csv
    │   │   │   │   │   ├── main_gpt4o_mini_naive_evaluate_leakage.json
    │   │   │   │   │   ├── main_gpt4o_mini_naive.csv
    │   │   │   │   │   ├── main_gpt4o_naive_evaluate_leakage.json
    │   │   │   │   │   ├── main_gpt4o_naive.csv
    │   │   │   │   │   ├── main_o1_mini_naive_evaluate_leakage.json
    │   │   │   │   │   ├── main_o1_mini_naive.csv
    │   │   │   │   │   ├── main_o1_mini_privacy_enhanced_evaluate_leakage.json
    │   │   │   │   │   ├── main_o1_mini_privacy_enhanced.csv
    │   │   │   │   │   └── main_qwen25_7b_instruct_privacy_enhanced.csv
    │   │   │   │   ├── output_probing
    │   │   │   │   │   ├── main_gpt4_o_mini.csv
    │   │   │   │   │   ├── main_gpt4_o.csv
    │   │   │   │   │   └── qwen25_7b_instruct.csv
    │   │   │   │   └── probing.py
    │   │   │   ├── helper
    │   │   │   │   ├── inspect_data.py
    │   │   │   │   ├── quick_start.ipynb
    │   │   │   │   └── utils.py
    │   │   │   ├── probing_component_spec.yaml
    │   │   │   ├── probing_environment.yaml
    │   │   │   ├── readme.md
    │   │   │   └── requirements.txt
    │   │   └── training
    │   │       ├── changes.diff
    │   │       ├── README.md
    │   │       └── run_rl_for_contextual_integrity.sh
    │   ├── datasets
    │   │   └── synthetic
    │   │       ├── dataset.json
    │   │       └── generate_new_data_from_seeds.py
    │   ├── experiments
    │   │   ├── privacylens.sh
    │   │   └── privacylens.yaml
    │   ├── models
    │   │   ├── gpt4o
    │   │   │   └── config.json
    │   │   ├── mai-ds-r1
    │   │   │   └── config.json
    │   │   ├── o1mini
    │   │   │   └── config.json
    │   │   └── phi4-reasoning
    │   │       └── config.json
    │   ├── notebooks
    │   │   ├── confaide.ipynb
    │   │   ├── explore_single_turn_conversations.ipynb
    │   │   ├── explore_taskmaster.ipynb
    │   │   ├── inspect_outputs.ipynb
    │   │   ├── privacylens.ipynb
    │   │   ├── README.md
    │   │   ├── requirements.txt
    │   │   └── sample_inference.ipynb
    │   ├── README.md
    │   └── src
    │       ├── data
    │       │   └── agent.py
    │       └── model_engines
    │           ├── __init__.py
    │           ├── base.py
    │           ├── huggingface_engine.py
    │           ├── openai_engine.py
    │           ├── utils.py
    │           └── vllm.py
    ├── README.md
    └── verl-supp
        ├── docker
        │   ├── Dockerfile.megatron
        │   ├── Dockerfile.ngc.vllm
        │   ├── Dockerfile.ngc.vllm0.8
        │   ├── Dockerfile.ngc.vllm0.8.sagemaker
        │   ├── Dockerfile.rocm
        │   └── Dockerfile.vemlp.vllm.te
        ├── docs
        │   ├── _static
        │   │   └── logo.png
        │   ├── advance
        │   │   ├── checkpoint.rst
        │   │   ├── dpo_extension.rst
        │   │   ├── fsdp_extension.rst
        │   │   ├── megatron_extension.rst
        │   │   └── placement.rst
        │   ├── amd_tutorial
        │   │   └── amd_build_dockerfile_page.rst
        │   ├── conf.py
        │   ├── data.rst
        │   ├── examples
        │   │   ├── config.rst
        │   │   ├── gsm8k_example.rst
        │   │   └── ppo_code_architecture.rst
        │   ├── experiment
        │   │   └── ppo.rst
        │   ├── faq
        │   │   └── faq.rst
        │   ├── hybrid_flow.rst
        │   ├── index.rst
        │   ├── Makefile
        │   ├── perf
        │   │   └── perf_tuning.rst
        │   ├── preparation
        │   │   ├── prepare_data.rst
        │   │   └── reward_function.rst
        │   ├── README_vllm0.7.md
        │   ├── README_vllm0.8.md
        │   ├── README.md
        │   ├── requirements-docs.txt
        │   ├── start
        │   │   ├── install.rst
        │   │   ├── multinode.rst
        │   │   └── quickstart.rst
        │   └── workers
        │       ├── fsdp_workers.rst
        │       ├── megatron_workers.rst
        │       └── ray_trainer.rst
        ├── examples
        │   ├── checkpoint
        │   │   ├── run_deepseek_megatron_ckpt.sh
        │   │   └── run_qwen_megatron_ckpt.sh
        │   ├── data_preprocess
        │   │   ├── contextual_integrity.py
        │   │   ├── full_hh_rlhf.py
        │   │   ├── geo3k.py
        │   │   ├── gsm8k.py
        │   │   ├── hellaswag.py
        │   │   └── math_dataset.py
        │   ├── generation
        │   │   ├── run_deepseek_v2_lite_math.sh
        │   │   └── run_deepseek7b_mutli_node.sh
        │   ├── grpo_trainer
        │   │   ├── run_deepseek7b_llm_math_megatron.sh
        │   │   ├── run_deepseek7b_llm_math.sh
        │   │   ├── run_deepseek7b_llm_megatron.sh
        │   │   ├── run_deepseek7b_llm_seq_balance.sh
        │   │   ├── run_deepseek7b_llm.sh
        │   │   ├── run_qwen2_5_vl-7b.sh
        │   │   ├── run_qwen2-7b_math_megatron.sh
        │   │   ├── run_qwen2-7b_math.sh
        │   │   ├── run_qwen2-7b_megatron.sh
        │   │   ├── run_qwen2-7b_seq_balance.sh
        │   │   └── run_qwen2-7b.sh
        │   ├── ppo_trainer
        │   │   ├── run_deepseek_full_hh_rlhf.sh
        │   │   ├── run_deepseek_math_gsm8k_megatron.sh
        │   │   ├── run_deepseek_megatron.sh
        │   │   ├── run_deepseek7b_llm_modelscope.sh
        │   │   ├── run_deepseek7b_llm_sp2.sh
        │   │   ├── run_deepseek7b_llm.sh
        │   │   ├── run_gemma.sh
        │   │   ├── run_qwen2-7b_math_gsm8k_megatron.sh
        │   │   ├── run_qwen2-7b_megatron.sh
        │   │   ├── run_qwen2-7b_rm_seq_balance.sh
        │   │   ├── run_qwen2-7b_rm.sh
        │   │   ├── run_qwen2-7b_seq_balance.sh
        │   │   ├── run_qwen2.5-32b.sh
        │   │   └── verl_getting_started.ipynb
        │   ├── ray
        │   │   └── tutorial.ipynb
        │   ├── remax_trainer
        │   │   ├── run_qwen2.5-3b_seq_balance.sh
        │   │   └── run_qwen2.5-7b_seq_balance.sh
        │   ├── rloo_trainer
        │   │   └── run_qwen2-7b.sh
        │   ├── sft
        │   │   └── gsm8k
        │   │       ├── run_deepseek_6b7.sh
        │   │       ├── run_gemma_2b.sh
        │   │       ├── run_gemma_7b.sh
        │   │       ├── run_qwen_05_peft.sh
        │   │       ├── run_qwen_05_sp2_liger.sh
        │   │       └── run_qwen_05_sp2.sh
        │   ├── slurm
        │   │   └── ray_on_slurm.slurm
        │   └── split_placement
        │       ├── config
        │       │   └── ppo_trainer_split.yaml
        │       ├── main_ppo_split.py
        │       ├── README.md
        │       ├── run_deepseek7b_llm.sh
        │       └── split_monkey_patch.py
        ├── LICENSE
        ├── Notice.txt
        ├── patches
        │   └── megatron_v4.patch
        ├── pyproject.toml
        ├── README.md
        ├── recipe
        │   └── prime
        │       ├── __init__.py
        │       ├── config
        │       │   └── prime_trainer.yaml
        │       ├── main_prime.py
        │       ├── prime_core_algos.py
        │       ├── prime_dp_rm.py
        │       ├── prime_fsdp_workers.py
        │       ├── prime_ray_trainer.py
        │       └── run_prime_qwen.sh
        ├── requirements_sglang.txt
        ├── requirements.txt
        ├── scripts
        │   ├── format.sh
        │   └── model_merger.py
        ├── setup.py
        ├── tests
        │   ├── __init__.py
        │   ├── checkpoint
        │   │   ├── run_deepseek_megatron_ckpt.sh
        │   │   ├── run_qwen_megatron_ckpt.sh
        │   │   └── test_fsdp_ckpt.py
        │   ├── distributed
        │   │   ├── run_all.sh
        │   │   └── test_tensor_dict.py
        │   ├── e2e
        │   │   ├── __init__.py
        │   │   ├── arithmetic_sequence
        │   │   │   ├── data
        │   │   │   │   ├── create_dataset.py
        │   │   │   │   ├── test.parquet
        │   │   │   │   └── train.parquet
        │   │   │   ├── model
        │   │   │   │   ├── config.json
        │   │   │   │   ├── create_model_tokenizer.py
        │   │   │   │   ├── generation_config.json
        │   │   │   │   ├── model.safetensors
        │   │   │   │   └── tokenizer_config.json
        │   │   │   └── rl
        │   │   │       ├── main_trainer.py
        │   │   │       └── README.md
        │   │   ├── check_custom_rwd_fn.py
        │   │   ├── check_results.py
        │   │   ├── envs
        │   │   │   ├── __init__.py
        │   │   │   └── digit_completion
        │   │   │       ├── __init__.py
        │   │   │       ├── task.py
        │   │   │       └── tokenizer.py
        │   │   ├── run_deepseek_grpo_megatron.sh
        │   │   ├── run_deepseek_grpo.sh
        │   │   ├── run_deepseek_megatron_parallelism.sh
        │   │   ├── run_deepseek_megatron.sh
        │   │   ├── run_qwen_grpo_megatron.sh
        │   │   ├── run_qwen_grpo.sh
        │   │   ├── run_qwen_gsm8k_custom_function_rm.sh
        │   │   ├── run_qwen_gsm8k_function_rm_both_kl.sh
        │   │   ├── run_qwen_gsm8k_function_rm_grpo.sh
        │   │   ├── run_qwen_gsm8k_function_rm_no_rmpad.sh
        │   │   ├── run_qwen_gsm8k_function_rm_remax.sh
        │   │   ├── run_qwen_gsm8k_function_rm.sh
        │   │   ├── run_qwen_gsm8k_model_rm_liger_kernel.sh
        │   │   ├── run_qwen_gsm8k_model_rm_no_rmpad.sh
        │   │   ├── run_qwen_gsm8k_model_rm_seq_balance.sh
        │   │   ├── run_qwen_gsm8k_model_rm_ulysses.sh
        │   │   ├── run_qwen_gsm8k_model_rm.sh
        │   │   ├── run_qwen_gsm8k_prime.sh
        │   │   ├── run_qwen_megatron_parallelism.sh
        │   │   ├── run_qwen_megatron.sh
        │   │   ├── run_qwen2vl_geo3k_function_rm.sh
        │   │   ├── run_ray_trainer_fire_sampling.sh
        │   │   ├── run_ray_trainer_rmpad.sh
        │   │   └── run_ray_trainer.sh
        │   ├── generation
        │   │   └── run_gen_qwen05.sh
        │   ├── gpu_utility
        │   │   ├── test_memory_buffers.py
        │   │   ├── test_ops.py
        │   │   └── test_torch_functional.py
        │   ├── kill_github_tests.sh
        │   ├── model
        │   │   ├── test_transformer.py
        │   │   └── test_transformers_ulysses.py
        │   ├── ray
        │   │   ├── check_worker_alive
        │   │   │   └── main.py
        │   │   ├── detached_worker
        │   │   │   ├── client.py
        │   │   │   ├── README.md
        │   │   │   ├── run.sh
        │   │   │   └── server.py
        │   │   ├── test_check_worker_alive.py
        │   │   ├── test_colocated_workers.py
        │   │   ├── test_data_transfer.py
        │   │   ├── test_driverfunc_to_worker.py
        │   │   ├── test_high_level_scheduling_api.py
        │   │   ├── test_ray_local_envs.py
        │   │   ├── test_rvdz.py
        │   │   ├── test_worker_group_basics.py
        │   │   └── test_worker_group_torch.py
        │   ├── rollout
        │   │   ├── run_fsdp_vllm.py
        │   │   ├── test_sglang_spmd.py
        │   │   ├── test_vllm_hf_loader.py
        │   │   └── test_vllm_spmd.py
        │   ├── sandbox
        │   │   └── test_sandbox.py
        │   ├── sanity
        │   │   ├── check_license.py
        │   │   └── test_import.py
        │   ├── sft
        │   │   ├── run_sft_qwen05_peft.sh
        │   │   ├── run_sft_qwen05_sp2_liger.sh
        │   │   ├── run_sft_sp_loss_match.sh
        │   │   ├── run_sft.sh
        │   │   └── test_sp_loss_match.py
        │   ├── utility
        │   │   └── test_tensor_dict_utilities.py
        │   └── verl
        │       └── utils
        │           └── dataset
        │               ├── test_rl_dataset.py
        │               ├── test_rm_dataset.py
        │               └── test_sft_dataset.py
        └── verl
            ├── __init__.py
            ├── models
            │   ├── __init__.py
            │   ├── llama
            │   │   ├── __init__.py
            │   │   └── megatron
            │   │       ├── __init__.py
            │   │       ├── checkpoint_utils
            │   │       │   ├── __init__.py
            │   │       │   ├── llama_loader_depracated.py
            │   │       │   ├── llama_loader.py
            │   │       │   └── llama_saver.py
            │   │       ├── layers
            │   │       │   ├── __init__.py
            │   │       │   ├── parallel_attention.py
            │   │       │   ├── parallel_decoder.py
            │   │       │   ├── parallel_linear.py
            │   │       │   ├── parallel_mlp.py
            │   │       │   └── parallel_rmsnorm.py
            │   │       └── modeling_llama_megatron.py
            │   ├── mcore
            │   │   ├── __init__.py
            │   │   ├── gpt_model.py
            │   │   ├── loader.py
            │   │   └── saver.py
            │   ├── qwen2
            │   │   ├── __init__.py
            │   │   └── megatron
            │   │       ├── __init__.py
            │   │       ├── checkpoint_utils
            │   │       │   ├── __init__.py
            │   │       │   ├── qwen2_loader_depracated.py
            │   │       │   ├── qwen2_loader.py
            │   │       │   └── qwen2_saver.py
            │   │       ├── layers
            │   │       │   ├── __init__.py
            │   │       │   ├── parallel_attention.py
            │   │       │   ├── parallel_decoder.py
            │   │       │   ├── parallel_linear.py
            │   │       │   ├── parallel_mlp.py
            │   │       │   └── parallel_rmsnorm.py
            │   │       └── modeling_qwen2_megatron.py
            │   ├── README.md
            │   ├── registry.py
            │   ├── transformers
            │   │   ├── __init__.py
            │   │   ├── llama.py
            │   │   ├── monkey_patch.py
            │   │   ├── qwen2_vl.py
            │   │   └── qwen2.py
            │   └── weight_loader_registry.py
            ├── protocol.py
            ├── single_controller
            │   ├── __init__.py
            │   ├── base
            │   │   ├── __init__.py
            │   │   ├── decorator.py
            │   │   ├── megatron
            │   │   │   ├── __init__.py
            │   │   │   ├── worker_group.py
            │   │   │   └── worker.py
            │   │   ├── register_center
            │   │   │   ├── __init__.py
            │   │   │   └── ray.py
            │   │   ├── worker_group.py
            │   │   └── worker.py
            │   └── ray
            │       ├── __init__.py
            │       ├── base.py
            │       └── megatron.py
            ├── third_party
            │   ├── __init__.py
            │   ├── sglang
            │   │   ├── __init__.py
            │   │   └── parallel_state.py
            │   └── vllm
            │       ├── __init__.py
            │       ├── vllm_v_0_3_1
            │       │   ├── __init__.py
            │       │   ├── arg_utils.py
            │       │   ├── config.py
            │       │   ├── llm_engine_sp.py
            │       │   ├── llm.py
            │       │   ├── model_loader.py
            │       │   ├── model_runner.py
            │       │   ├── parallel_state.py
            │       │   ├── tokenizer.py
            │       │   ├── weight_loaders.py
            │       │   └── worker.py
            │       ├── vllm_v_0_4_2
            │       │   ├── __init__.py
            │       │   ├── arg_utils.py
            │       │   ├── config.py
            │       │   ├── dtensor_weight_loaders.py
            │       │   ├── hf_weight_loader.py
            │       │   ├── llm_engine_sp.py
            │       │   ├── llm.py
            │       │   ├── megatron_weight_loaders.py
            │       │   ├── model_loader.py
            │       │   ├── model_runner.py
            │       │   ├── parallel_state.py
            │       │   ├── spmd_gpu_executor.py
            │       │   ├── tokenizer.py
            │       │   └── worker.py
            │       ├── vllm_v_0_5_4
            │       │   ├── __init__.py
            │       │   ├── arg_utils.py
            │       │   ├── config.py
            │       │   ├── dtensor_weight_loaders.py
            │       │   ├── hf_weight_loader.py
            │       │   ├── llm_engine_sp.py
            │       │   ├── llm.py
            │       │   ├── megatron_weight_loaders.py
            │       │   ├── model_loader.py
            │       │   ├── model_runner.py
            │       │   ├── parallel_state.py
            │       │   ├── spmd_gpu_executor.py
            │       │   ├── tokenizer.py
            │       │   └── worker.py
            │       └── vllm_v_0_6_3
            │           ├── __init__.py
            │           ├── arg_utils.py
            │           ├── config.py
            │           ├── dtensor_weight_loaders.py
            │           ├── hf_weight_loader.py
            │           ├── llm_engine_sp.py
            │           ├── llm.py
            │           ├── megatron_weight_loaders.py
            │           ├── model_loader.py
            │           ├── model_runner.py
            │           ├── parallel_state.py
            │           ├── spmd_gpu_executor.py
            │           ├── tokenizer.py
            │           └── worker.py
            ├── trainer
            │   ├── __init__.py
            │   ├── config
            │   │   ├── evaluation.yaml
            │   │   ├── generation.yaml
            │   │   ├── ppo_megatron_trainer.yaml
            │   │   ├── ppo_trainer.yaml
            │   │   └── sft_trainer.yaml
            │   ├── fsdp_sft_trainer.py
            │   ├── main_eval.py
            │   ├── main_generation.py
            │   ├── main_ppo.py
            │   ├── ppo
            │   │   ├── __init__.py
            │   │   ├── core_algos.py
            │   │   ├── metric_utils.py
            │   │   └── ray_trainer.py
            │   └── runtime_env.yaml
            ├── utils
            │   ├── __init__.py
            │   ├── checkpoint
            │   │   ├── __init__.py
            │   │   ├── checkpoint_manager.py
            │   │   ├── fsdp_checkpoint_manager.py
            │   │   └── megatron_checkpoint_manager.py
            │   ├── config.py
            │   ├── dataset
            │   │   ├── __init__.py
            │   │   ├── README.md
            │   │   ├── rl_dataset.py
            │   │   ├── rm_dataset.py
            │   │   └── sft_dataset.py
            │   ├── debug
            │   │   ├── __init__.py
            │   │   ├── performance.py
            │   │   └── trajectory_tracker.py
            │   ├── distributed.py
            │   ├── flops_counter.py
            │   ├── fs.py
            │   ├── fsdp_utils.py
            │   ├── hdfs_io.py
            │   ├── import_utils.py
            │   ├── logger
            │   │   ├── __init__.py
            │   │   └── aggregate_logger.py
            │   ├── logging_utils.py
            │   ├── megatron
            │   │   ├── __init__.py
            │   │   ├── memory.py
            │   │   ├── optimizer.py
            │   │   ├── pipeline_parallel.py
            │   │   ├── sequence_parallel.py
            │   │   └── tensor_parallel.py
            │   ├── megatron_utils.py
            │   ├── memory_buffer.py
            │   ├── model.py
            │   ├── py_functional.py
            │   ├── ray_utils.py
            │   ├── rendezvous
            │   │   ├── __init__.py
            │   │   └── ray_backend.py
            │   ├── reward_score
            │   │   ├── __init__.py
            │   │   ├── contextual_integrity_reward.py
            │   │   ├── geo3k.py
            │   │   ├── gsm8k.py
            │   │   ├── math_verify.py
            │   │   ├── math.py
            │   │   ├── prime_code
            │   │   │   ├── __init__.py
            │   │   │   ├── testing_util.py
            │   │   │   └── utils.py
            │   │   └── prime_math
            │   │       ├── __init__.py
            │   │       ├── grader.py
            │   │       └── math_normalize.py
            │   ├── seqlen_balancing.py
            │   ├── tokenizer.py
            │   ├── torch_dtypes.py
            │   ├── torch_functional.py
            │   ├── tracking.py
            │   └── ulysses.py
            ├── version
            │   └── version
            └── workers
                ├── __init__.py
                ├── actor
                │   ├── __init__.py
                │   ├── base.py
                │   ├── dp_actor.py
                │   └── megatron_actor.py
                ├── critic
                │   ├── __init__.py
                │   ├── base.py
                │   ├── dp_critic.py
                │   └── megatron_critic.py
                ├── fsdp_workers.py
                ├── megatron_workers.py
                ├── reward_manager
                │   ├── __init__.py
                │   ├── naive.py
                │   └── prime.py
                ├── reward_model
                │   ├── __init__.py
                │   ├── base.py
                │   └── megatron
                │       ├── __init__.py
                │       └── reward_model.py
                ├── rollout
                │   ├── __init__.py
                │   ├── base.py
                │   ├── hf_rollout.py
                │   ├── naive
                │   │   ├── __init__.py
                │   │   └── naive_rollout.py
                │   ├── sglang_rollout
                │   │   ├── __init__.py
                │   │   └── sglang_rollout.py
                │   ├── tokenizer.py
                │   └── vllm_rollout
                │       ├── __init__.py
                │       ├── fire_vllm_rollout.py
                │       ├── vllm_rollout_spmd.py
                │       └── vllm_rollout.py
                └── sharding_manager
                    ├── __init__.py
                    ├── base.py
                    ├── fsdp_sglang.py
                    ├── fsdp_ulysses.py
                    ├── fsdp_vllm.py
                    └── megatron_vllm.py
```
