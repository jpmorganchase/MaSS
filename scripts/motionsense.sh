# train single attribute classifiers
OUTPUT_DIR='./icml_results/motionsense/'
python3 -m torch.distributed.run --nproc_per_node 1 --master_port 22222 train_sinlge_attribute.py --data_dir ./data/motionsense/ --dataset motionsense --model motionsense_mlp --batch-size 256 --attributes gender --lr 0.001 --weight-decay 0.0001 --epochs 100  --output_dir $OUTPUT_DIR/gender/
python3 -m torch.distributed.run --nproc_per_node 1 --master_port 22222 train_sinlge_attribute.py --data_dir ./data/motionsense/ --dataset motionsense --model motionsense_mlp --batch-size 256 --attributes id --lr 0.001  --weight-decay 0.0001 --epochs 100  --output_dir $OUTPUT_DIR/id/
python3 -m torch.distributed.run --nproc_per_node 1 --master_port 22222 train_sinlge_attribute.py --data_dir ./data/motionsense/ --dataset motionsense --model motionsense_mlp --batch-size 256 --attributes act --lr 0.001  --weight-decay 0.0001 --epochs 100  --output_dir $OUTPUT_DIR/act/

# pre-train another utiliy classifier for further fine-tuning after MaSS
python3 -m torch.distributed.run --nproc_per_node 1 --master_port 22222 train_sinlge_attribute.py --data_dir ./data/motionsense/ --dataset motionsense --model motionsense_mlp --batch-size 256 --attributes act --lr 0.001  --weight-decay 0.0001 --epochs 100  --output_dir $OUTPUT_DIR/act_seed1/ --seed 1
python3 -m torch.distributed.run --nproc_per_node 1 --master_port 22222 train_sinlge_attribute.py --data_dir ./data/motionsense/ --dataset motionsense --model motionsense_mlp --batch-size 256 --attributes id --lr 0.001  --weight-decay 0.0001 --epochs 100  --output_dir $OUTPUT_DIR/id_seed1/ --seed 1

# pre-train infonce
python3 train_infonce.py  --data_dir ./data/motionsense/ --dataset motionsense --model motionsense_mlp --epochs 100 --learning_rate 1e-3 --weight_decay 1e-4  --output_dir $OUTPUT_DIR/infonce/

