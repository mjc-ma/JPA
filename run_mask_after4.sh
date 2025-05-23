#!/bin/bash

# 模拟 $i 的循环
for i in {0..142}; do
    CUDA_VISIBLE_DEVICES=7 python /home/majc/Attack/src/execs/attack.py --config-file /home/majc/Attack/configs/nudity/mask/text_mask_after4.json --attacker.attack_idx "$i" --logger.name "attack_idx_$i"
done
