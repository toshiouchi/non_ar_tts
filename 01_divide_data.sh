echo "stage 1: Data preparation"
echo "train/dev/eval split"
mkdir -p data
find ./downloads/jsut_ver1.1/basic5000/wav -name "*.wav" -exec basename {} .wav \; | sort > data/utt_list.txt
head -n 4700 data/utt_list.txt > data/train.list
tail -300 data/utt_list.txt > data/deveval.list
head -n 200 data/deveval.list > data/dev.list
tail -n 100 data/deveval.list > data/eval.list
rm -f data/deveval.list