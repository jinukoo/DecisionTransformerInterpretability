ls trajectories
python src/run_decision_transformer.py \
    --exp_name "MiniGrid-Dynamic-Obstacles-8x8-v0" \
    --trajectory_path "trajectories/MiniGrid-Dynamic-Obstacles-8x8-v0bd60729d-dc0b-4294-9110-8d5f672aa82c.pkl" \
    --d_model 128 \
    --n_heads 4 \
    --d_mlp 256 \
    --n_layers 2 \
    --learning_rate 0.0001 \
    --batch_size 128 \
    --batches 101 \
    --n_ctx 3 \
    --pct_traj 1 \
    --weight_decay 0.001 \
    --seed 1 \
    --wandb_project_name "DecisionTransformerInterpretability" \
    --test_frequency 100 \
    --test_batches 10 \
    --eval_frequency 100 \
    --eval_episodes 10 \
    --initial_rtg 1 \
    --prob_go_from_end 0.1 \
    --eval_max_time_steps 1000 \
    --cuda True \
    --track False 