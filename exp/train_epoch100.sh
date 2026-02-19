# [12:43, 2/13/2026] Professor Alexander: 1.  Prithvi Unet (Tversky α=0.3, β=0.7, γ=0.75, E=100) dengan seed 48
# [12:52, 2/13/2026] Professor Alexander: 2.  Unet (Tversky α=0.3, β=0.7, γ=2, E=100) dengan seed 12
# 3. Prithvi (Tversky α=0.3, β=0.7, γ=2, E=100) dengan seed 124

# python ./exp/train.py --loss_func tversky --epochs 100 --torch_seed 48 --tv_alpha 0.3 --tv_beta 0.7 --tv_gamma 0.75 --model_name 'prithvi_unet'

# python ./exp/train.py --loss_func tversky --epochs 100 --torch_seed 12 --tv_alpha 0.3 --tv_beta 0.7 --tv_gamma 2 --model_name 'unet'

# python ./exp/train.py --loss_func tversky --epochs 100 --torch_seed 124 --tv_alpha 0.3 --tv_beta 0.7 --tv_gamma 2 --model_name 'prithvi'

python ./exp/multimodal_trainer.py --loss_func tversky --epochs 100 --batch_size 6 --torch_seed 124 --finetune_ratio 1