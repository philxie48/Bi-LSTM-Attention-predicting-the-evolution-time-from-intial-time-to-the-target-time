@echo off
echo Starting training with 100%% autocorrelation loss (Phase 2 of 500-epoch strategy)...
echo Output directory: D:\neural network\100auto_101_200

REM Kill any existing Python processes that might be running
taskkill /f /im python.exe 2>nul

python train_pure_autocorr.py ^
    --data_dir "D:/sample3" ^
    --output_dir "D:/neural network/100auto_101_200" ^
    --epochs 100 ^
    --batch_size 32 ^
    --hidden_size 256 ^
    --num_layers 2 ^
    --dropout 0.3 ^
    --max_lag 15 ^
    --start_lr 0.0005 ^
    --min_lr 0.0001 ^
    --weight_decay 0.008 ^
    --clip_grad_norm 1.0 ^
    --teacher_forcing ^
    --teacher_forcing_start 0.6 ^
    --teacher_forcing_end 0.5 ^
    --noise_level 0.05 ^
    --mixed_precision ^
    --save_every 5 ^
    --seed 42 ^
    --existing_model "D:/neural network/100auto/best_model.pth"

echo Training complete!
echo.
echo To continue training for the next 100 epochs (Phase 3), use:
echo python train_pure_autocorr.py --existing_model "D:/neural network/100auto_101_200/best_model.pth" --epochs 100 --start_lr 0.0001 --min_lr 0.00005 --output_dir "D:/neural network/100auto_201_300" --max_lag 15 --teacher_forcing_start 0.5 --teacher_forcing_end 0.4 --noise_level 0.04 --weight_decay 0.005
echo.
pause
