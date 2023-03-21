"""
This script transforms the icdar2015, syntext1, syntext2, mlt2017, totaltext datasets into the format of the MindSpore OCR project.
These datasets were used for pretraining
"""

DATASETS_DIR="../ade_datasets" 
############################## Download Polygon Annotations ##############################
ANNOT_FILE="$DATASETS_DIR/polys.tar.gz"
if  [ "$(ls -A $ANNOT_FILE)"  ]; then
  echo "Download Skipped."
else
  wget --no-check-certificate "https://ucsdcloud-my.sharepoint.com/:u:/g/personal/xiz102_ucsd_edu/ES4aqkvamlJAgiPNFJuYkX4BLo-5cDx9TD_6pnMJnVhXpw?e=tu9D8t&download=1" -O $ANNOT_FILE
fi

ANNOT_DIR=DATASETS_DIR
tar -xvzf $ANNOT_FILE -C $ANNOT_DIR

############################## Convert Annotations ##############################
#########################icdar2015#########################

DIR="$DATASETS_DIR/icdar2015"
if test -f "$DIR/train_det_gt.txt"; then
    echo "$DIR/train_det_gt.txt exists."
else
    python tools/dataset_converters/convert.py \
        --dataset_name  coco \
        --task det \
        --image_dir $DIR/train_images/ \
        --label_dir  $DIR/train_poly.json \
        --output_path $DIR/train_det_gt.txt
fi
# pretraining only used train set

#########################syntext150k#########################

DIR="$DATASETS_DIR/syntext1"

if test -f "$DIR/train_det_gt.txt"; then
    echo "$DIR/train_det_gt.txt exists."
else
    python tools/dataset_converters/convert.py \
        --dataset_name  coco \
        --task det \
        --image_dir $DIR/images/ \
        --label_dir  $DIR/annotations/train_poly.json \
        --output_path $DIR/train_det_gt.txt
fi

DIR="$DATASETS_DIR/syntext2"
if test -f "$DIR/train_det_gt.txt"; then
    echo "$DIR/train_det_gt.txt exists."
else
    python tools/dataset_converters/convert.py \
        --dataset_name  coco \
        --task det \
        --image_dir $DIR/images/ \
        --label_dir  $DIR/annotations/train_poly.json \
        --output_path $DIR/train_det_gt.txt
fi
##########################mlt2017#########################
DIR="$DATASETS_DIR/mlt2017"
if test -f "$DIR/train_det_gt.txt"; then
    echo "$DIR/train_det_gt.txt exists."
else
    python tools/dataset_converters/convert.py \
        --dataset_name  coco \
        --task det \
        --image_dir $DIR/images/MLT_train_images/ \
        --label_dir $DIR/annotations/train_poly.json \
        --output_path $DIR/train_det_gt.txt
fi
# pretraining only used train set

##########################total_text#########################
DIR="$DATASETS_DIR/totaltext"
if test -f "$DIR/train_det_gt.txt"; then
    echo "$DIR/train_det_gt.txt exists."
else
    python tools/dataset_converters/convert.py \
        --dataset_name  coco \
        --task det \
        --image_dir $DIR/train_images \
        --label_dir $DIR/train_poly.json \
        --output_path $DIR/train_det_gt.txt
fi
if test -f "$DIR/test_det_gt.txt"; then
    echo "$DIR/test_det_gt.txt exists."
else
    python tools/dataset_converters/convert.py \
        --dataset_name  coco \
        --task det \
        --image_dir $DIR/test_images \
        --label_dir $DIR/test_poly.json \
        --output_path $DIR/test_det_gt.txt
fi

  # pretraining only used train set,  but also evaluated on the test set

###################################################