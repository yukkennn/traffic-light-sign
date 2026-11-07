set -x
python3 -m tlcv.train_cls --model a1_resnet18
for s in 42 7; do
python3 -m tlcv.train_cls --model yolo11s --seed $s
python3 -m tlcv.train_cls --model yolo11s --condsim --seed $s
python3 -m tlcv.train_cls --model resnet18 --condsim --seed $s
python3 -m tlcv.train_cls --model resnet18 --seed $s
done
echo ALLDONE
