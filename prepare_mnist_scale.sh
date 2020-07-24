# !/bin/bash
# MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja

MNIST_DIR="${MNIST_DIR:-./datasets}"
MNIST_SCALE_DIR="${MNIST_SCALE_DIR:-./datasets}"


VALID_HASH=(
    "00f60f1d99234b8fe8e5e1fe79306759" 
    "f13da5cde237fb224846b8e0dd8188b2" 
    "f06ad4b20d3f7d33f7b417cf7a1eb811" 
    "eeb05f67093efae9f9b3f5a5617b2444" 
    "5aa583af4e3c08a32f633bcb565823e5" 
    "35ec90ad3cd9543123208cbb4a311233" 
)


echo "Preparing datasets..."
for i in {0..5}
do 
    echo ""
    echo "Dataset [$((i+1))/6]"

    python prepare_datasets.py --source "$MNIST_DIR" --dest "$MNIST_SCALE_DIR" --min_scale 0.3 --download --seed $i
    HASH="$(python prepare_datasets.py --source "$MNIST_DIR" --dest "$MNIST_SCALE_DIR" --min_scale 0.3 --seed $i --validate)"

    
    if [ "$HASH" != "${VALID_HASH[i]}" ]
    then
        echo "!!! Dataset is invalid. Its MD5 does not match the original one."
        echo "!!! Valid MD5  : ${VALID_HASH[i]}"
        echo "!!! Current MD5: $HASH"
        exit 1
    fi
    
done

echo "All datasets are generated and validated."
