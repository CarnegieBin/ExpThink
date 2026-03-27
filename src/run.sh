#!/bin/bash
set -x
export HYDRA_FULL_ERROR=1



export PROJECT_NAME="ExpThink"
export q_ratio=0.3
export max_response_length=6144
export EXP_NAME="deepseek_llm_1.5b-max_token-${max_response_length}-q_ratio-${q_ratio}"
export MODEL_PATH="/ssd2/llm_models/DeepSeek-R1-Distill-Qwen-1.5B"
export data_root="./data"
export TRAIN_FILE="${data_root}/deepscaler.parquet"
export TEST_FILE="['${data_root}/aime_16.parquet','${data_root}/amc.parquet','${data_root}/math.parquet','${data_root}/minerva.parquet','${data_root}/olympiad.parquet']"

#export TEST_FILE="['${data_root}/aime_16.parquet']"

export CKPTS_DIR="/home/work/tcbian/ExpThink/ckpts/${PROJECT_NAME}/${EXP_NAME}"


# Train over a single node, 4 A800-80GB GPUs.
python3 -m src.main_dapo \
    +reward_model.q_ratio=${q_ratio} \
    +trainer.credit=True \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=1024 \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=256 \
    data.train_batch_size=512 \
    actor_rollout_ref.rollout.n=16 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric="seq_final_reward" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20000 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20000 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20000 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_model_len=17408 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=20000 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k="-1" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k="-1" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.max_tokens=16384 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.total_epochs=10 \
    trainer.resume_mode=auto \
    custom_reward_function.path="./src/custom_think_rm.py" \
    custom_reward_function.name="verify_think_rm" \
    2>&1 | tee "${PROJECT_NAME}-${EXP_NAME}.log"