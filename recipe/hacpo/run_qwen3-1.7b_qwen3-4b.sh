set -x

math_train_path="Your own path"
math500_test_path="Your own path"

train_files="['$math_train_path']"
test_files="['$math500_test_path']"

clip_ratio_low=0.0003
clip_ratio_high=0.0004


n_gpus_per_node=8

tensor_model_parallel_size=8
mirco_bacth_size=8
gpu_memory_utilization=0.7
offload=True

project_name="hacpo"
experiment_name="qwen3_1.7b_4b"


main_model=Qwen/Qwen3-1.7B-Base
aux_model=Qwen/Qwen3-4B-Base


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=mapo \
    algorithm.model_source_baseline=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=mapo_clip \
    actor_rollout_ref.actor.alpha=1.0 \
    actor_rollout_ref.actor.accuracy_window_size=5 \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${main_model} \
    aux_model.enable=True \
    aux_model.model.path=${aux_model} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64  \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${mirco_bacth_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.aux_clip_ratio_low=0.8 \
    actor_rollout_ref.actor.aux_clip_ratio_step=0.025 \
    actor_rollout_ref.actor.model_source_performance=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${mirco_bacth_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${mirco_bacth_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger='["wandb"]' \
    trainer.val_before_train=True \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=1 \
    trainer.save_freq=3 \
    trainer.test_freq=3 \
    trainer.total_epochs=1@