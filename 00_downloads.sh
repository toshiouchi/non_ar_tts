echo "stage 0: Data download"
mkdir -p downloads
if [ ! -d downloads/jsut_ver1.1 ]; then
    cd downloads
    curl -LO http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip
    unzip -o jsut_ver1.1.zip
    cd -
fi
if [ ! -d downloads/jsut-lab ]; then
    cd downloads
    curl -LO https://github.com/sarulab-speech/jsut-label/archive/v0.0.2.zip
    unzip -o v0.0.2.zip
    ln -s jsut-label-0.0.2 jsut-label
    cd -
fi