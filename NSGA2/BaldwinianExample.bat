REM %* Allows for any additional parameters
python NSGAII.py --evol-mode baldwin --num-gens 1000 --pop-size 10 --use-proper-time-limits --watch-frequency 1 --use-gae --num-processes 1 --num-updates 64 --lr 2e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-steps 1024 --num-mini-batch 16 --use-linear-lr-decay --entropy-coef 0.01 --gamma 0.99 --gae-lambda 0.95 --ppo-epoch 4 --final-pt %*