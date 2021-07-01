#!/bin/bash

# Install download package for google drive
pip install gdown

# Create data directory
mkdir data
cd data

# Library, Sidgwick, Maths data
gdown -O lsm.zip "https://drive.google.com/uc?export=download&id=1OP-2m0nApOUYqW3knGbe7-ey6Ke_hPJM"

unzip lsm.zip -d lsm

echo "Downloaded Library/Sidgwick/Maths data."

read -p "Do you wish to download all other data? (Type Y/N)" y

if [ "$y" = "Y" ]; then
  gdown -O west.zip "https://drive.google.com/uc?export=download&id=1AXoHWmgDuckD4u-SJJN9HXp_6xEdao1U"
  unzip west.zip -d west
  
  gdown -O centre.zip "https://drive.google.com/uc?export=download&id=1Cyz7CQDlcCEcRXyxF6QhJUz0lutj2u-2"
  unzip centre.zip -d centre
fi

echo "Finished Downloading!"
