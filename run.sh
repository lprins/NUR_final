#!/bin/bash

echo "Creating directories"
mkdir -p plots DataFiles output movies/2D_zeldovich movies/3D_zeldovich/xy movies/3D_zeldovich/yz movies/3D_zeldovich/xz

wget -P DataFiles https://home.strw.leidenuniv.nl/~nobels/coursedata/randomnumbers.txt
wget -P DataFiles https://home.strw.leidenuniv.nl/~nobels/coursedata/GRBs.txt
wget -P DataFiles https://home.strw.leidenuniv.nl/~nobels/coursedata/colliding.hdf5

python3 ex1.py
python3 ex2.py
python3 ex3.py
python3 ex4.py > output/ex4.txt
python3 ex5.py > output/ex5.txt
python3 ex7.py > output/ex7.txt

cd movies/2D_zeldovich/
ffmpeg -framerate 30 -pattern_type glob -i "*.png" -s:v 640x480 -c:v libx264 -profile:v high -level 4.0 -crf 10 -tune animation -preset slow -pix_fmt yuv420p -r 25 -threads 12 -f mp4 ICs.mp4
cd ../3D_zeldovich/xy
ffmpeg -framerate 30 -pattern_type glob -i "*.png" -s:v 640x480 -c:v libx264 -profile:v high -level 4.0 -crf 10 -tune animation -preset slow -pix_fmt yuv420p -r 25 -threads 12 -f mp4 ICs.mp4
cd ../xz
ffmpeg -framerate 30 -pattern_type glob -i "*.png" -s:v 640x480 -c:v libx264 -profile:v high -level 4.0 -crf 10 -tune animation -preset slow -pix_fmt yuv420p -r 25 -threads 12 -f mp4 ICs.mp4
cd ../yz
ffmpeg -framerate 30 -pattern_type glob -i "*.png" -s:v 640x480 -c:v libx264 -profile:v high -level 4.0 -crf 10 -tune animation -preset slow -pix_fmt yuv420p -r 25 -threads 12 -f mp4 ICs.mp4
cd ../../../texfiles/
pdflatex -shell-escape main
cp main.pdf ../main.pdf
