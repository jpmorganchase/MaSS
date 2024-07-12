# train single attribute classifiers
OUTPUT_DIR='./icml_results/motionsense/'
python3 -m torch.distributed.run --nproc_per_node 1 --master_port 22222 train_sinlge_attribute.py --data_dir ./data/motionsense/ --dataset motionsense --model motionsense_mlp --batch-size 256 --attributes gender --lr 0.001 --weight-decay 0.0001 --epochs 100  --output_dir $OUTPUT_DIR/gender/ --deterministic
python3 -m torch.distributed.run --nproc_per_node 1 --master_port 22222 train_sinlge_attribute.py --data_dir ./data/motionsense/ --dataset motionsense --model motionsense_mlp --batch-size 256 --attributes id --lr 0.001  --weight-decay 0.0001 --epochs 100  --output_dir $OUTPUT_DIR/id/ --deterministic
python3 -m torch.distributed.run --nproc_per_node 1 --master_port 22222 train_sinlge_attribute.py --data_dir ./data/motionsense/ --dataset motionsense --model motionsense_mlp --batch-size 256 --attributes act --lr 0.001  --weight-decay 0.0001 --epochs 100  --output_dir $OUTPUT_DIR/act/ --deterministic

# pre-train another utiliy classifier for further fine-tuning after MaSS
python3 -m torch.distributed.run --nproc_per_node 1 --master_port 22222 train_sinlge_attribute.py --data_dir ./data/motionsense/ --dataset motionsense --model motionsense_mlp --batch-size 256 --attributes act --lr 0.001  --weight-decay 0.0001 --epochs 100  --output_dir $OUTPUT_DIR/act_seed1/ --seed 1 --deterministic

# pre-train infonce
python3 train_infonce.py  --data_dir ./data/motionsense/ --dataset motionsense --model motionsense_mlp --epochs 100 --learning_rate 1e-3 --weight_decay 1e-4  --output_dir $OUTPUT_DIR/infonce/  --deterministic

# Table 2 (8)

python3 -m torch.distributed.run --nproc_per_node 1 --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 train_mass.py --data_dir ./data/motionsense --dataset motionsense  \
--epochs 200 --batch-size 128 \
--gen_model motionsense_gen_mlp \
--disc_model motionsense_mlp \
--disc_infonce_model motionsense_mlp \
--attribute_models_weights id=$OUTPUT_DIR/id/checkpoint.pth gender=$OUTPUT_DIR/gender/checkpoint.pth infonce=$OUTPUT_DIR/infonce/checkpoint.pth \
--new_dis_attribute_models_weights act=$OUTPUT_DIR/act_seed1/checkpoint.pth \
--lr 0.0001 \
--seed 0 \
--loss-m gender=0.0 id=0.0 \
--eval-attributes act \
--output_dir $OUTPUT_DIR/mass/ --deterministic
