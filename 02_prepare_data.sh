echo "stage 2: Feature generation for Tacotron"
python new_preprocess.py data/eval.list downloads/jsut_ver1.1/basic5000/wav downloads/jsut-label/labels/basic5000 dump/jsut_sr22050/norm/eval hifi-gan-master --n_jobs 4
python new_preprocess.py data/dev.list downloads/jsut_ver1.1/basic5000/wav downloads/jsut-label/labels/basic5000 dump/jsut_sr22050/norm/dev hifi-gan-master --n_jobs 4
python new_preprocess.py data/train.list downloads/jsut_ver1.1/basic5000/wav downloads/jsut-label/labels/basic5000 dump/jsut_sr22050/norm/train hifi-gan-master --n_jobs 4
