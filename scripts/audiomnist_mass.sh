OUTPUT_DIR='./icml_results/audiomnist/'

# Table 3 (10)

python3 -m torch.distributed.run --nproc_per_node 1 --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 train_mass.py --data_dir ./data/audiomnist --dataset audiomnist  \
--epochs 2000 --batch-size 256 \
--loss-m gender=0.0 accent=0.0 age=0.0 id=0.0 \
--eval-attributes digit \
--gen_model audiomnist_gen_mlp \
--disc_model audiomnist_mlp \
--disc_infonce_model audiomnist_infonce_mlp \
--attribute_models_weights id=icml_results/audiomnist/id/checkpoint.pth gender=icml_results/audiomnist/gender/checkpoint.pth age=icml_results/audiomnist/age/checkpoint.pth accent=icml_results/audiomnist/accent/checkpoint.pth gender=icml_results/audiomnist/gender/checkpoint.pth digit=icml_results/audiomnist/digit/checkpoint.pth infonce=icml_results/audiomnist/infonce/checkpoint.pth \
--new_dis_attribute_models_weights digit=icml_results/audiomnist/digit_seed1/checkpoint.pth \
--lr 0.0001 \
--seed 0 --output_dir $OUTPUT_DIR/mass/


# Table 4 (11)

python3 -m torch.distributed.run --nproc_per_node 1 --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 train_mass.py --data_dir ./data/audiomnist --dataset audiomnist  \
--epochs 2000 --batch-size 256 \
--loss-m gender=0.0 accent=0.0 age=0.0 \
--loss-n digit=2.3 \
--eval-attributes id \
--gen_model audiomnist_gen_mlp \
--disc_model audiomnist_mlp \
--disc_infonce_model audiomnist_infonce_mlp \
--attribute_models_weights id=icml_results/audiomnist/id/checkpoint.pth gender=icml_results/audiomnist/gender/checkpoint.pth age=icml_results/audiomnist/age/checkpoint.pth accent=icml_results/audiomnist/accent/checkpoint.pth gender=icml_results/audiomnist/gender/checkpoint.pth digit=icml_results/audiomnist/digit/checkpoint.pth infonce=icml_results/audiomnist/infonce/checkpoint.pth \
--new_dis_attribute_models_weights id=icml_results/audiomnist/id_seed1/checkpoint.pth digit=icml_results/audiomnist/digit_seed1/checkpoint.pth \
--lr 0.0001 \
--seed 0 --output_dir $OUTPUT_DIR/mass/


