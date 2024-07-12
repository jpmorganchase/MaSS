# train single attribute classifiers
OUTPUT_DIR='./icml_results/audiomnist/'
python3 -m torch.distributed.run --nproc_per_node 1 --master_port 11111 train_sinlge_attribute.py --data_dir ./data/audiomnist/ --dataset audiomnist --model audiomnist_mlp --attributes accent --batch-size 256 --lr 0.01 --weight-decay 0.05 --epochs 100  --output_dir $OUTPUT_DIR/accent/
python3 -m torch.distributed.run --nproc_per_node 1 --master_port 11111 train_sinlge_attribute.py --data_dir ./data/audiomnist/ --dataset audiomnist --model audiomnist_mlp --attributes age --batch-size 256 --lr 0.01 --weight-decay 0.05 --epochs 100  --output_dir $OUTPUT_DIR/age/
python3 -m torch.distributed.run --nproc_per_node 1 --master_port 11111 train_sinlge_attribute.py --data_dir ./data/audiomnist/ --dataset audiomnist --model audiomnist_mlp --attributes gender --batch-size 256 --lr 0.01 --weight-decay 0.05 --epochs 100  --output_dir $OUTPUT_DIR/gender/
python3 -m torch.distributed.run --nproc_per_node 1 --master_port 11111 train_sinlge_attribute.py --data_dir ./data/audiomnist/ --dataset audiomnist --model audiomnist_mlp --attributes id --batch-size 256 --lr 0.01 --weight-decay 0.05 --epochs 100  --output_dir $OUTPUT_DIR/id/
python3 -m torch.distributed.run --nproc_per_node 1 --master_port 11111 train_sinlge_attribute.py --data_dir ./data/audiomnist/ --dataset audiomnist --model audiomnist_mlp --attributes digit --batch-size 256 --lr 0.01 --weight-decay 0.05 --epochs 100  --output_dir $OUTPUT_DIR/digit/

# pre-train another utiliy classifier for further fine-tuning after MaSS
python3 -m torch.distributed.run --nproc_per_node 1 --master_port 11111 train_sinlge_attribute.py --data_dir ./data/audiomnist/ --dataset audiomnist --model audiomnist_mlp --attributes digit --batch-size 256 --lr 0.01 --weight-decay 0.05 --epochs 100  --output_dir $OUTPUT_DIR/digit_seed1/ --seed 1
python3 -m torch.distributed.run --nproc_per_node 1 --master_port 11111 train_sinlge_attribute.py --data_dir ./data/audiomnist/ --dataset audiomnist --model audiomnist_mlp --attributes id --batch-size 256 --lr 0.01 --weight-decay 0.05 --epochs 100  --output_dir $OUTPUT_DIR/id_seed1/ --seed 1

# pre-train the feature extractor for unannotated attributes
python3 train_infonce.py --data_dir ./data/audiomnist/ --dataset audiomnist --model audiomnist_infonce_mlp --learning_rate 0.05 --weight_decay 1e-4  --epochs 100  --output_dir $OUTPUT_DIR/infonce/


