@echo off
echo Starting training with 50%% MAE + 50%% Autocorrelation loss (Phase 5 of 500-epoch strategy)...
echo Output directory: D:\neural network\50mae_50auto_401_500

REM Kill any existing Python processes that might be running
taskkill /f /im python.exe 2>nul

python train_combined_mae_autocorr.py ^
    --data_dir "D:/sample3" ^
    --output_dir "D:/neural network/50mae_50auto_401_500" ^
    --epochs 100 ^
    --batch_size 32 ^
    --hidden_size 256 ^
    --num_layers 2 ^
    --dropout 0.3 ^
    --max_lag 20 ^
    --start_lr 0.00001 ^
    --min_lr 0.000005 ^
    --weight_decay 0.002 ^
    --clip_grad_norm 0.6 ^
    --teacher_forcing ^
    --teacher_forcing_start 0.3 ^
    --teacher_forcing_end 0.2 ^
    --noise_level 0.01 ^
    --mixed_precision ^
    --save_every 5 ^
    --seed 42 ^
    --existing_model "D:/neural network/50mae_50auto_301_400/best_model.pth"

echo Training complete!
echo.
pause
