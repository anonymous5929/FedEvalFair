This project is further developed based on the following code:
https://github.com/cuis15/FCFL

## Adult Dataset
You can perform federated learning training without any constraints and calculate the bootstrap value by using the following command:
```bash
python main.py --step_size 0.13 --max_epoch_stage1 800 --max_epoch_stage2 1000 --eps_g 1.0 --sensitive_attr race --seed 1 --target_dir_name race_specific_1-0 --uniform --uniform_eps
python main.py --step_size 0.13 --max_epoch_stage1 800 --max_epoch_stage2 1000 --eps_g 1.0 --sensitive_attr race --seed 1 --target_dir_name race_specific_1-0 --uniform --uniform_eps --valid True --bootstrap 100 --load_epoch 1799 
```
To train models with different fairness based on the Adult dataset, train them using the following command, each with different weights optimized:
>If you want to obtain the result under bootstrap, you need to add parameter `--valid True --bootstrap 100 --load_epoch 1799` after each command as above and execute it again.
```bash
python main.py --weight_eps 0.9 --max_epoch_stage1 800 --max_epoch_stage2 1000 --step_size 0.10 --sensitive_attr race --seed 1 --target_dir_name race_specific_0-9
python main.py --weight_eps 0.7 --max_epoch_stage1 800 --max_epoch_stage2 1000 --step_size 0.06 --sensitive_attr race --seed 1 --target_dir_name race_specific_0-7
python main.py --weight_eps 0.6 --max_epoch_stage1 800 --max_epoch_stage2 1000 --step_size 0.06 --sensitive_attr race --seed 1 --target_dir_name race_specific_0-6
python main.py --weight_eps 0.5 --max_epoch_stage1 800 --max_epoch_stage2 1000 --step_size 0.05 --sensitive_attr race --seed 1 --target_dir_name race_specific_0-5
python main.py --weight_eps 0.4 --max_epoch_stage1 800 --max_epoch_stage2 1000 --step_size 0.04 --sensitive_attr race --seed 1 --target_dir_name race_specific_0-4
python main.py --weight_eps 0.3 --max_epoch_stage1 800 --max_epoch_stage2 1000 --step_size 0.04 --sensitive_attr race --seed 1 --target_dir_name race_specific_0-3
python main.py --weight_eps 0.2 --max_epoch_stage1 800 --max_epoch_stage2 1000 --step_size 0.03 --sensitive_attr race --seed 1 --target_dir_name race_specific_0-2
python main.py --weight_eps 0.1 --max_epoch_stage1 800 --max_epoch_stage2 1000 --step_size 0.02 --sensitive_attr race --seed 1 --target_dir_name race_specific_0-1
```
## Eicu Dataset
Eicu dataset can obtain from the following website: https://eicu-crd.mit.edu/gettingstarted/access/
Need preprocess patient.xlsx using FUEL/dataset_generate_eicu.py.
### Origin Model Train
```bash
python main.py --step_size 0.50 --max_epoch_stage1 800 --max_epoch_stage2 1500 --eps_g 1.0 --dataset eicu --sensitive_attr race --seed 1 --target_dir_name eicu_race_specific_1-0 --uniform --uniform_eps
python main.py --step_size 0.50 --max_epoch_stage1 800 --max_epoch_stage2 1500 --eps_g 1.0 --dataset eicu --sensitive_attr race --seed 1 --target_dir_name eicu_race_specific_1-0 --uniform --uniform_eps --valid True --bootstrap 100 --load_epoch 2299
```

### Bias experiment
- Purpose: To compare the stability of traditional methods and our method by adding bias in the dataset.
- Bias：0，0.1，0.2，0.3，0.4，0.45，0.5
- Each bias 200 models
```bash
bias: 0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5
seed: 0, 1, ..., 10
model_count: 0, 1, ..., 20
python main.py --step_size 0.50 --max_epoch_stage1 800 --max_epoch_stage2 1500 --dataset eicu --sensitive_attr race --target_dir_name new_trial/bias/${bias}/eicu_DP_0-01_0-50_seed_${seed} --eps_g 0.01 --uniform_eps --new_trial ${model_count} --seed $seed --new_trial_method bias --new_trial_bias_rate $bias
```