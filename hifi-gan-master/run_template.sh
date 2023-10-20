python3 train.py  --config config_v1.json --training_epochs 300 --checkpoint_interval 1000 --input_wavs_dir JSUT/wavs --input_mels_dir JSUT/mels --input_training_file JSUT/training.txt --input_validation_file JSUT/validation.txt

python3 inference_e2e.py --checkpoint_file cp_hifigan/g_00030000

python3 inference.py --checkpoint_file cp_hifigan/g_

