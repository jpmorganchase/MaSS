OUTPUT_DIR='./icml_results/motionsense/'

# Table 2 (8)

python3 -m torch.distributed.run --nproc_per_node 1 --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 train_mass.py --data_dir ./data/motionsense --dataset motionsense  \
--epochs 200 --batch-size 128 \
--gen_model motionsense_gen_mlp \
--disc_model motionsense_mlp \
--disc_infonce_model motionsense_mlp \
--attribute_models_weights id=icml_results/motionsense/id/checkpoint.pth gender=icml_results/motionsense/gender/checkpoint.pth act=icml_results/motionsense/act/checkpoint.pth infonce=icml_results/motionsense/infonce/checkpoint.pth \
--new_dis_attribute_models_weights act=icml_results/motionsense/act/checkpoint.pth \
--lr 0.0001 \
--seed 0 \
--loss-m gender=0.0 id=0.0 \
--eval-attributes act \
--output_dir $OUTPUT_DIR/mass/


# Table 9

python3 -m torch.distributed.run --nproc_per_node 1 --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 train_mass.py --data_dir ./data/motionsense --dataset motionsense  \
--epochs 200 --batch-size 128 \
--gen_model motionsense_gen_mlp \
--disc_model motionsense_mlp \
--disc_infonce_model motionsense_mlp \
--attribute_models_weights id=icml_results/motionsense/id/checkpoint.pth gender=icml_results/motionsense/gender/checkpoint.pth act=icml_results/motionsense/act/checkpoint.pth infonce=icml_results/motionsense/infonce/checkpoint.pth \
--new_dis_attribute_models_weights act=icml_results/motionsense/act/checkpoint.pth \
--lr 0.0001 \
--seed 0 \
--loss-m gender=0.0 \
--loss-n id=1.6 \
--eval-attributes act \
--output_dir $OUTPUT_DIR/mass/
