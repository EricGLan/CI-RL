#!/bin/bash

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.max_prompt_length=768 \
    data.max_response_length=2048 \
    data.truncation=error \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path="Qwen/Qwen2.5-3B-Instruct" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.n=16 \
    +actor_rollout_ref.rollout.stop_token_ids=[151645] \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=contextual_integrity \
    trainer.experiment_name=Qwen2.5-3B-Instruct_grpo_final_action_only_bsz32_n16_mini128_kl0.001 \
    trainer.n_gpus_per_node=8 \
    trainer.log_val_generations=1 \
    trainer.nnodes=1 \
    trainer.save_freq=240 \
    trainer.test_freq=240 \
    trainer.default_local_dir=outputs/output_dir \
    trainer.total_epochs=150 \
    "$@"
