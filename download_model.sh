echo "Downloading exif_final.zip"

echo "Saving to ./ckpt/exif_final.zip"

URL=http://people.eecs.berkeley.edu/~owens/consistency/exif_final.zip
MODEL_FILE=exif_final.zip
wget -N $URL -O $MODEL_FILE

mkdir -p ./ckpt/
unzip exif_final.zip -d ./ckpt/

rm exif_final.zip
