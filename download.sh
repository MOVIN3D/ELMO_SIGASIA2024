#!/bin/bash

if [ -z "$1" ]; then
    echo "Please provide an argument: ELMO, MOVIN, or EVAL."
    exit 1
fi

FILE="$1"

mkdir -p "./datasets"

case "$FILE" in
    "ELMO")
        URL="https://www.dropbox.com/scl/fi/u1eiczhp6183unasp5uyf/ELMO_dataset.zip?rlkey=d56imgt9z1np5fc4rmt9o9bu9&st=5y9w7p4z&dl=1"
        ZIP_FILE="./datasets/ELMO_dataset.zip"
        ;;
    "MOVIN")
        URL="https://www.dropbox.com/scl/fi/ab394n5p1ovijn01bkbz2/MOVIN_dataset.zip?rlkey=zufn5t6apewrsgt5vi0f30fi7&st=1myot7pd&dl=1"
        ZIP_FILE="./datasets/MOVIN_dataset.zip"
        ;;
    "EVAL")
        URL="https://www.dropbox.com/scl/fi/fgv64i91kw2rkiacpvjir/evaluation_dataset.zip?rlkey=1105x27uqa4hl3jdvccykwl8d&st=le7m6pob&dl=1"
        ZIP_FILE="./datasets/evaluation_dataset.zip"
        ;;
    *)
        echo "Available arguments are ELMO, MOVIN, and EVAL."
        exit 1
        ;;
esac

wget "$URL" -O "$ZIP_FILE"
unzip "$ZIP_FILE" -d "./datasets"
rm "$ZIP_FILE"