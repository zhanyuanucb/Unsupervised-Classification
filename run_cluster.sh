python moco.py --config_env configs/env.yml --config_exp configs/pretext/moco_fashion.yml 
python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_fashion.yml
python selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_fashion.yml

python eval.py --config_exp configs/selflabel/selflabel_fashion.yml \
               --model fashion_results/fashion/selflabel/model.pth.tar \
               --download ./cluster_results