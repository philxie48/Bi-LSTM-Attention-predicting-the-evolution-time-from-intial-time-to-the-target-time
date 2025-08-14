@echo off
echo Starting training with 100%% MAE loss (Phase 3 of 500-epoch strategy)...
echo Output directory: D:\neural network\100mae_201_300

REM Kill any existing Python processes that might be running
taskkill /f /im python.exe 2>nul

python train_pure_mae.py ^
    --data_dir "D:/sample3" ^
    --output_dir "D:/neural network/100mae_201_300" ^
    --epochs 100 ^
    --batch_size 32 ^
    --hidden_size 256 ^
    --num_layers 2 ^
    --dropout 0.3 ^
    --start_lr 0.0001 ^
    --min_lr 0.00005 ^
    --weight_decay 0.005 ^
    --clip_grad_norm 0.8 ^
    --teacher_forcing ^
    --teacher_forcing_start 0.5 ^
    --teacher_forcing_end 0.4 ^
    --noise_level 0.03 ^
    --mixed_precision ^
    --save_every 5 ^
    --seed 42 ^
    --existing_model "D:/neural network/100mae_101_200/best_model.pth"

echo Training complete!
echo.
echo To continue training for the next 100 epochs (Phase 4), use:
echo python train_pure_mae.py --existing_model "D:/neural network/100mae_201_300/best_model.pth" --epochs 100 --start_lr 0.00005 --min_lr 0.00001 --output_dir "D:/neural network/100mae_301_400" --teacher_forcing_start 0.4 --teacher_forcing_end 0.3 --noise_level 0.02 --weight_decay 0.003
echo.
pause
