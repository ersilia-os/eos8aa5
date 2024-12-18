absolute_path=$(realpath "$1")
current_dir=$(pwd)
echo "Absolute path: $absolute_path"
temp_dir=$(mktemp -d)
echo "Temporary directory created: $temp_dir"
echo "Preprocessing data"
python $absolute_path/code/preprocess.py $2 $temp_dir
cd $absolute_path/code/scripts
python preprocess_downstream_dataset.py --data_path $temp_dir --dataset data
python extract_features.py --config base --model_path $absolute_path/../checkpoints/pretrained/base/base.pth --data_path $temp_dir --dataset data
cd $current_dir
python $absolute_path/code/postprocess.py $temp_dir $3
rm -rf "$temp_dir"
echo "Temporary directory removed"