#!/bin/bash

echo "Downloading exif_final.zip"

# Google Drive link to exif_final.zip
gdown https://drive.google.com/uc?id=1X6b55rwZzU68Mz1m68WIX_G2idsEw3Qh
echo "Unzipping to ./ckpt/exif_final.zip"

mkdir -p ./ckpt/
unzip exif_final.zip -d ./ckpt/

rm exif_final.zip