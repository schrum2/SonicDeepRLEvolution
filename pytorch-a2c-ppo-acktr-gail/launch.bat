REM Specify number of processes at command line, or default to 1
if [%1]==[] (SET /A num_proc=1) else (SET /A num_proc=%1)
REM Specify environment at command line, or default to "SonicTheHedgehog-Genesis"
if [%2]==[] (SET env="SonicTheHedgehog-Genesis") else (SET env="%2")

python main.py --env-name %env% --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes %num_proc% --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --save-dir "save" --save-interval 5