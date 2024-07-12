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


# Table 3 (10)

python3 -m torch.distributed.run --nproc_per_node 1 --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 train_mass.py --data_dir ./data/audiomnist --dataset audiomnist  \
--epochs 2000 --batch-size 256 \
--loss-m gender=0.0 accent=0.0 age=0.0 id=0.0 \
--eval-attributes digit \
--gen_model audiomnist_gen_mlp \
--disc_model audiomnist_mlp \
--disc_infonce_model audiomnist_infonce_mlp \
--attribute_models_weights id=$OUTPUT_DIR/id/checkpoint.pth gender=$OUTPUT_DIR/gender/checkpoint.pth age=$OUTPUT_DIR/age/checkpoint.pth accent=$OUTPUT_DIR/accent/checkpoint.pth gender=$OUTPUT_DIR/gender/checkpoint.pth infonce=$OUTPUT_DIR/infonce/checkpoint.pth \
--new_dis_attribute_models_weights digit=$OUTPUT_DIR/digit_seed1/checkpoint.pth \
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
--attribute_models_weights gender=$OUTPUT_DIR/gender/checkpoint.pth age=$OUTPUT_DIR/age/checkpoint.pth accent=$OUTPUT_DIR/accent/checkpoint.pth gender=$OUTPUT_DIR/gender/checkpoint.pth digit=$OUTPUT_DIR/digit/checkpoint.pth infonce=$OUTPUT_DIR/infonce/checkpoint.pth \
--new_dis_attribute_models_weights id=$OUTPUT_DIR/id_seed1/checkpoint.pth digit=$OUTPUT_DIR/digit_seed1/checkpoint.pth \
--lr 0.0001 \
--seed 0 --output_dir $OUTPUT_DIR/mass/


