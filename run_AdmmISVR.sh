#! /bin/sh
python main.py ./datas ./results/logistics_AdmmISVR --epochs 10 --batch_size 100 --l2_lambda 0.01 --model_name logistics --dataset_name fashion --group_channels 32 --drop_rate 0.1 --last_drop_rate 0.5 --loglevel INFO --train_data_length 10000 --cuda_device_no 0 --skip_plots --scheduler StepLR --step_size 410 --gamma 0.9 --sleep_factor 0 AdmmISVR node0 ./conf/node_list.n8.doublering.round8.json ./conf/hosts.n8.json --lr 0.002 --swap_timeout 10 --async_step > ./results/logistics_AdmmISVR/node0.txt 2>&1 &
python main.py ./datas ./results/logistics_AdmmISVR --epochs 10 --batch_size 100 --l2_lambda 0.01 --model_name logistics --dataset_name fashion --group_channels 32 --drop_rate 0.1 --last_drop_rate 0.5 --loglevel INFO --train_data_length 10000 --cuda_device_no 1 --skip_plots --scheduler StepLR --step_size 410 --gamma 0.9 --sleep_factor 0 AdmmISVR node1 ./conf/node_list.n8.doublering.round8.json ./conf/hosts.n8.json --lr 0.002 --swap_timeout 10 --async_step > ./results/logistics_AdmmISVR/node1.txt 2>&1 &
python main.py ./datas ./results/logistics_AdmmISVR --epochs 10 --batch_size 100 --l2_lambda 0.01 --model_name logistics --dataset_name fashion --group_channels 32 --drop_rate 0.1 --last_drop_rate 0.5 --loglevel INFO --train_data_length 10000 --cuda_device_no 2 --skip_plots --scheduler StepLR --step_size 410 --gamma 0.9 --sleep_factor 0 AdmmISVR node2 ./conf/node_list.n8.doublering.round8.json ./conf/hosts.n8.json --lr 0.002 --swap_timeout 10 --async_step > ./results/logistics_AdmmISVR/node2.txt 2>&1 &
python main.py ./datas ./results/logistics_AdmmISVR --epochs 10 --batch_size 100 --l2_lambda 0.01 --model_name logistics --dataset_name fashion --group_channels 32 --drop_rate 0.1 --last_drop_rate 0.5 --loglevel INFO --train_data_length 10000 --cuda_device_no 3 --skip_plots --scheduler StepLR --step_size 410 --gamma 0.9 --sleep_factor 0 AdmmISVR node3 ./conf/node_list.n8.doublering.round8.json ./conf/hosts.n8.json --lr 0.002 --swap_timeout 10 --async_step > ./results/logistics_AdmmISVR/node3.txt 2>&1 &
python main.py ./datas ./results/logistics_AdmmISVR --epochs 10 --batch_size 100 --l2_lambda 0.01 --model_name logistics --dataset_name fashion --group_channels 32 --drop_rate 0.1 --last_drop_rate 0.5 --loglevel INFO --train_data_length 10000 --cuda_device_no 4 --skip_plots --scheduler StepLR --step_size 410 --gamma 0.9 --sleep_factor 0 AdmmISVR node4 ./conf/node_list.n8.doublering.round8.json ./conf/hosts.n8.json --lr 0.002 --swap_timeout 10 --async_step > ./results/logistics_AdmmISVR/node4.txt 2>&1 &
python main.py ./datas ./results/logistics_AdmmISVR --epochs 10 --batch_size 100 --l2_lambda 0.01 --model_name logistics --dataset_name fashion --group_channels 32 --drop_rate 0.1 --last_drop_rate 0.5 --loglevel INFO --train_data_length 10000 --cuda_device_no 5 --skip_plots --scheduler StepLR --step_size 410 --gamma 0.9 --sleep_factor 0 AdmmISVR node5 ./conf/node_list.n8.doublering.round8.json ./conf/hosts.n8.json --lr 0.002 --swap_timeout 10 --async_step > ./results/logistics_AdmmISVR/node5.txt 2>&1 &
python main.py ./datas ./results/logistics_AdmmISVR --epochs 10 --batch_size 100 --l2_lambda 0.01 --model_name logistics --dataset_name fashion --group_channels 32 --drop_rate 0.1 --last_drop_rate 0.5 --loglevel INFO --train_data_length 10000 --cuda_device_no 6 --skip_plots --scheduler StepLR --step_size 410 --gamma 0.9 --sleep_factor 0 AdmmISVR node6 ./conf/node_list.n8.doublering.round8.json ./conf/hosts.n8.json --lr 0.002 --swap_timeout 10 --async_step > ./results/logistics_AdmmISVR/node6.txt 2>&1 &
python main.py ./datas ./results/logistics_AdmmISVR --epochs 10 --batch_size 100 --l2_lambda 0.01 --model_name logistics --dataset_name fashion --group_channels 32 --drop_rate 0.1 --last_drop_rate 0.5 --loglevel INFO --train_data_length 10000 --cuda_device_no 7 --skip_plots --scheduler StepLR --step_size 410 --gamma 0.9 --sleep_factor 0 AdmmISVR node7 ./conf/node_list.n8.doublering.round8.json ./conf/hosts.n8.json --lr 0.002 --swap_timeout 10 --async_step > ./results/logistics_AdmmISVR/node7.txt 2>&1 &