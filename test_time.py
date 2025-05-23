import os
import json
import json
import os
import re
import matplotlib.pyplot as plt
###################################### test_time ############################
# root_dir = 'files/results/random_text/random_esd_nudity11'
# root_dir = 'files/results/violence/text_2sim_esd_violence_4.0'
# root_dir = 'files/results/random_esd_nudity'
# root_dir = 'results/text_similarity_SLD_nudity_classifier'
# # BPS_nude 
###################################### write to clip score #############@#################
## new_baseline:mask/before/after
# root = "/home/majc/Attack/results/SLD/Scon_sdnp"
# methed_name = "Scon_Medium"
# root = '/root/data/Attack/results/Checker/JPA/ESD3'
# root = '/root/data/Attack/results/Checker/SLD_undiff/Scon_max'
root = "/data/majc/Attack1/random/max"
methed_name = "Scon_sdnp"
# root = "/home/majc/Attack/results/text_grad_esd_nudity_classifier_new_baseline_FMN"
# methed_name = "text_strong"
# root = "/home/majc/Attack/results/insert/text_mask_after_insertk"
# methed_name = "insertk"

# root = "/home/majc/Attack/results/mask/text_mask_after6"
# methed_name = "mask_after6"

# ##old_baseline
# root = "results/text_2similarity_esd_nudity_classifier"
# methed_name = "BPS_nude"

# root = "/home/majc/Attack/results/text_2similarity_esd_nudity_classifier_lr0.01r"
# methed_name = "BPS_nude"

# BPS
# root = "results/text_similarity_esd_nudity_classifier"
# methed_name = "BPS"

# # grad
# root = "results/text_grad_esd_nudity_classifier"
# methed_name = "grad"

# grad_new_baseline
# root = "results/text_grad_esd_nudity_classifier_new_baseline"
# methed_name = "grad"

# P4D
# root = "/home/majc/Attack/files/results/P4D_new_baseline"
# methed_name = "P4D_new_baseline"

# random_text
# root = "results/random_text/random_esd_nudity"
# methed_name = "random_text"

# root = "files/results/random_text/random_esd_nudity11"
# methed_name = "random_text11"

dataset_path = 'files/dataset/i2p_nude'
adv_dataset_path = 'files/dataset/adv_i2p_nude'
# 读取JSON文件
def visual_loss():
    loss_values = []
    main_folder = root
    num_files = 0  # 用于记录文件总数
    for subdir, _, files in os.walk(main_folder):
        for file in files:
            if file == "log.json":
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r') as f:
                    log_data = json.load(f)
                    ## 把log写进list里
                    for log_entry in log_data:
                        loss = log_entry.get('loss')
                        if loss is not None:
                            loss_values.append(loss)
                # 绘制loss值的图表
                plt.figure(figsize=(8, 6))
                plt.plot(loss_values)
                plt.xlabel('Entry Index')
                plt.ylabel('Loss Value')
                plt.title('Loss Values Over Entries')
                plt.savefig(file_path.replace('log.json', 'loss_plot.png'))  # 在同一文件夹中保存图表
                loss_values = []

#########################@############### Exp4 测试result里面acc是多少##########Z#################

def test_time(root_dir):
    # 定义全局变量
    success_count = 0
    no_attack_count = 0
    success_time = 0.0
    total_time = 0.0
    # 主文件夹路径
    main_folder = root_dir
    num_files = 0  # 用于记录文件总数
    for subdir, _, files in os.walk(main_folder):
        for file in files:
            if file == "log.json":
                num_files += 1
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r') as f:
                    log_data = json.load(f)

                    ## 计算首次check就是true的个数，并退出
                    for log_entry in log_data:
                        success = log_entry.get('success', False)
                        if success:
                            no_attack_count += 1
                        else:
                            break
                    ## 计算有true就行的个数
                    for log_entry in log_data:
                        success = log_entry.get('success', False)
                        if success:
                            success_count += 1
                            relative_time = log_entry.get('relative_time')
                            if relative_time:
                                time_parts = relative_time.split(':')
                                if len(time_parts) == 3:
                                    try:
                                        hours, minutes, seconds = map(float, time_parts)
                                        total_seconds = hours * 3600 + minutes * 60 + seconds
                                        success_time += total_seconds
                                        total_time += total_seconds
                                    except ValueError:
                                        pass
                            break

                    if not any(entry.get('success', False) for entry in log_data):
                        last_relative_time = log_data[-1].get('relative_time')
                        if last_relative_time:
                            time_parts = last_relative_time.split(':')
                            if len(time_parts) == 3:
                                try:
                                    hours, minutes, seconds = map(float, time_parts)
                                    total_seconds = hours * 3600 + minutes * 60 + seconds
                                    total_time += total_seconds
                                except ValueError:
                                    pass

    # 打印全局变量的具体值
    print(num_files)
    print("No Attack Count:", no_attack_count)
    print("No_Attack_acc:", no_attack_count/num_files)
    print("Success Count:", success_count)
    print("BPS_Attack_acc:", success_count/num_files)
    print("Success Time:", success_time / 60)  # 转换为分钟
    print("Total Time:", total_time / 60)  # 转换为分钟

def test_nude_type(root_dir):
    ## 定义检测到的暴露类型
    Feet_exposed = 0
    Armpits_exposed = 0
    Belly_exposed = 0
    FEMALE_BREAST_EXPOSED = 0
    FEMALE_GENITALIA_EXPOSED = 0
    MALE_BREAST_EXPOSED = 0
    MALE_GENITALIA_EXPOSED = 0
    BUTTOCKS_EXPOSED = 0
    ANUS_EXPOSED = 0


    no_attack_Feet_exposed = 0
    no_attack_Armpits_exposed = 0
    no_attack_Belly_exposed = 0
    no_attack_count = 0
    main_folder = root_dir
    num_files = 0  # 用于记录文件总数
    for subdir, _, files in os.walk(main_folder):
        for file in files:
            if file == "log.json":
                num_files += 1
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r') as f:
                    log_data = json.load(f)
                    ## 计算首次check就是true的个数，并退出
                    # for log_entry in log_data:
                    #     success = log_entry.get('success', False)
                    #     if success:
                    #         no_attack_count += 1
                    #         nude_keys = list(log_entry.get('nude', {}).keys())
                    #         if len(nude_keys) == 1 and nude_keys[0] == "ARMPITS_EXPOSED":
                    #             no_attack_Armpits_exposed += 1
                    #         if len(nude_keys) == 1 and nude_keys[0] == "FEET_EXPOSED":
                    #             no_attack_Feet_exposed += 1 
                    #         if len(nude_keys) == 1 and nude_keys[0] == "BELLY_EXPOSED":
                    #             no_attack_Belly_exposed += 1 
                                              
                    #     else:
                    #         break
                    ## 计算有true就行的个数
                    for log_entry in log_data:
                        success = log_entry.get('success', False)
                        if success:
                            nude_keys = list(log_entry.get('nude', {}).keys())
                            Armpits_exposed += sum(1 for key in nude_keys if key == "ARMPITS_EXPOSED") if len(nude_keys) == 1 else 0
                            Feet_exposed += sum(1 for key in nude_keys if key == "FEET_EXPOSED") if len(nude_keys) == 1 else 0
                            Belly_exposed += sum(1 for key in nude_keys if key == "BELLY_EXPOSED") if len(nude_keys) == 1 else 0
                            FEMALE_BREAST_EXPOSED += sum("FEMALE_BREAST_EXPOSED" in key for key in nude_keys) if len(nude_keys) == 1 else 0
                            FEMALE_GENITALIA_EXPOSED+= sum(1 for key in nude_keys if key == "FEMALE_GENITALIA_EXPOSED") if len(nude_keys) == 1 else 0
                            MALE_BREAST_EXPOSED += sum(1 for key in nude_keys if key == "MALE_BREAST_EXPOSED") if len(nude_keys) == 1 else 0
                            MALE_GENITALIA_EXPOSED += sum(1 for key in nude_keys if key == "MALE_GENITALIA_EXPOSED") if len(nude_keys) == 1 else 0
                            BUTTOCKS_EXPOSED += sum(1 for key in nude_keys if key == "BUTTOCKS_EXPOSED") if len(nude_keys) == 1 else 0
                            ANUS_EXPOSED += sum(1 for key in nude_keys if key == "ANUS_EXPOSED") if len(nude_keys) == 1 else 0

                            # Armpits_exposed += sum(1 for key in nude_keys if key == "ARMPITS_EXPOSED")
                            # Feet_exposed += sum(1 for key in nude_keys if key == "FEET_EXPOSED")
                            # Belly_exposed += sum(1 for key in nude_keys if key == "BELLY_EXPOSED")
                            # FEMALE_BREAST_EXPOSED += sum("FEMALE_BREAST_EXPOSED" in key for key in nude_keys)
                            # FEMALE_GENITALIA_EXPOSED+= sum(1 for key in nude_keys if key == "FEMALE_GENITALIA_EXPOSED")
                            # MALE_BREAST_EXPOSED += sum(1 for key in nude_keys if key == "MALE_BREAST_EXPOSED")
                            # MALE_GENITALIA_EXPOSED += sum(1 for key in nude_keys if key == "MALE_GENITALIA_EXPOSED")
                            # BUTTOCKS_EXPOSED += sum(1 for key in nude_keys if key == "BUTTOCKS_EXPOSED")
                            # ANUS_EXPOSED += sum(1 for key in nude_keys if key == "ANUS_EXPOSED")

    print("NO_ATTACK_SUCCESS:",no_attack_count)                            
    print("TOTAL_FILES:",num_files)
    print("\nARMPITS_EXPOSED:",Armpits_exposed)
    print("FEET_EXPOSED:",Feet_exposed)
    print("BEELY_EXPOSED:",Belly_exposed)

    print("FEMALE_BREAST_EXPOSED:",FEMALE_BREAST_EXPOSED)
    print("FEMALE_GENITALIA_EXPOSED:",FEMALE_GENITALIA_EXPOSED)
    print("MALE_BREAST_EXPOSED:",MALE_BREAST_EXPOSED)
    print("MALE_GENITALIA_EXPOSED:",MALE_GENITALIA_EXPOSED)
    print("BUTTOCKS_EXPOSED:",BUTTOCKS_EXPOSED)
    print("ANUS_EXPOSED:",ANUS_EXPOSED)


    print("\nNO_ATTACK_ARMPITS_EXPOSED:",no_attack_Armpits_exposed)
    print("NO_ATTACK_FEET_EXPOSED:",no_attack_Feet_exposed)
    print("NO_ATTACK_BEELY_EXPOSED:",no_attack_Belly_exposed)
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"

## ########################################  Exp3: extract clip score###########################################
from PIL import Image
from transformers import AutoProcessor, CLIPVisionModel,CLIPModel

from src.tasks.utils.datasets import get as get_dataset
import numpy as np
from sklearn.decomposition import PCA

def read_image_extract_features(model,processor,dataset_path):
    dataset=get_dataset(dataset_path)
    img_list = [dataset[i][0] for i in range(len(dataset.idxs))]
    images = torch.stack(img_list, dim=0)
    inputs = processor(images=images, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    features = outputs.pooler_output # pooled CLS states
    return features
    ## [bs,1024]

def plot_PCA(features1,features2):

    merged_tensor = np.concatenate((features1, features2), axis=0)
    # 创建 PCA 模型并拟合数据
    pca = PCA(n_components=2)  # 保留前两个主成分
    pca.fit(merged_tensor)
    # 获取主成分和投影的图像
    components = pca.components_  # 主成分
    projected1 = pca.transform(features1)  # 图像在主成分上的投影
    projected2 = pca.transform(features2)  # 图像在主成分上的投影
    # 绘制图像投影
    plt.figure(figsize=(8, 6))
    plt.scatter(projected1[:, 0], projected1[:, 1], alpha=0.5,label='Nude_Prompts')
    plt.scatter(projected2[:, 0], projected2[:, 1], alpha=0.5,label='Adv_Prompts')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('SD nude/adv images distribute')
    plt.show()
    plt.savefig('plot.png')  # 指定保存路径和文件名，可以根据需要修改文件格式


def visual_clip_score(dataset_path,adv_dataset_path):
    model = CLIPVisionModel.from_pretrained("/home/majc/openai/clip-vit-large-patch14")
    processor = AutoProcessor.from_pretrained("/home/majc/openai/clip-vit-large-patch14")
    features2 = read_image_extract_features(model,processor,adv_dataset_path)
    features1 = read_image_extract_features(model,processor,dataset_path)
    fea1 = features1.detach().numpy()
    fea2 = features2.detach().numpy()

    np.savez('SD_img_features.npz', features1=fea1, features2=fea2)
    loaded_data = np.load('SD_img_features.npz')
    # 从加载的数据中获取保存的数组
    loaded_data1 = loaded_data['features1']
    loaded_data2 = loaded_data['features2']

    plot_PCA(fea1,fea2)

############################# Exp2 读取ruslts里的success adv_prompts并保存######################
import re
import pandas as pd
def extract_number(directory):
    # print(directory,int(directory.split("_")[-1]))
    return int(directory.split("_")[-1])

## measure cos sim版本的df
from transformers import CLIPTextModel, CLIPTokenizer
import torch
def measure_cos(prompt1,prompt2,tokenizer,text_encoder,device):
    adv_ids = tokenizer(prompt1,padding=True, return_tensors="pt").to(device)
    if adv_ids['input_ids'].shape[1]<=77:
        adv_output = text_encoder(**adv_ids)
    else:
        zero_tensor = torch.tensor(0.0)
        return zero_tensor.to(device)
    pooled_output = adv_output.pooler_output
    nude_ids = tokenizer(prompt2,padding=True, return_tensors="pt").to(device)
    if nude_ids["input_ids"].shape[1]<=77:
        nude_output = text_encoder(**nude_ids)
    else:
        zero_tensor = torch.tensor(0.0)
        return zero_tensor.to(device)
    nude_pooled_output = nude_output.pooler_output
    ## calculate cos
    pooled_output = pooled_output / pooled_output.norm(dim=1, keepdim=True)
    nude_pooled_output = nude_pooled_output / nude_pooled_output.norm(dim=1, keepdim=True)

    cos = pooled_output @ nude_pooled_output.t()
    return cos

def process_logs(root_dir):
    model_name = "/root/data/stable-diffusion-v1-4"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(device)
    # 初始化一个空的 DataFrame
    df = pd.DataFrame()
    df_prompts = pd.read_csv("prompts/nudity.csv")

    print(os.listdir(root_dir))
    folders = sorted(os.listdir(root_dir),key=extract_number)
    print(folders)  
    for directory in folders:
        if directory.startswith("attack_idx_"):
            current_dir = os.path.join(root_dir, directory)
            log_file = os.path.join(current_dir, "log.json")
            # 检查是否存在 log.json 文件
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
                    found_success = False
                    for idx, log_entry in enumerate(logs):
                        if idx == 0 or not log_entry.get("success", False):
                            continue  # 跳过第一个或者不含有成功字段的项
                        found_success = True

                        prompt = log_entry.get("prompt", None)
                        nude_keys = list(log_entry.get('nude', {}).keys())

                        ## 攻击成功的原来的prompt和nudity的cos
                        if prompt :
                            # if len(nude_keys) == 1 and nude_keys[0] == "BUTTOCKS_EXPOSED":
                            #     break
                            # else:
                            index = int(directory.split("_")[-1])
                            # 将成功的 prompt 值存入 DataFrame
                            df.at[index, methed_name] = prompt
                            nudity_prompt = df_prompts.loc[index, 'prompt']
                            eval_seed = str(df_prompts.loc[index, 'evaluation_seed'])
                            df.at[index, "ORI"] = nudity_prompt
                            df.at[index, "seed"] = eval_seed
                            df.at[index, 'attack_index'] = index
                            cos = measure_cos(nudity_prompt,"nudity",tokenizer=tokenizer,text_encoder=text_encoder,device=device)
                            df.at[index, f'{methed_name}_cos'] = cos.detach().item()

                            break
                    # 如果 log 文件中没有成功的记录或者是第一个，标记为 False
                    if not found_success:
                        index = int(directory.split("_")[-1])
                        df.at[index, 'attack_index'] = index
                    # 将成功的 prompt 值存入 DataFrame
                        # nudity_prompt = df_prompts.loc[index, 'prompt']
                        # cos = measure_cos(nudity_prompt,"nudity",tokenizer=tokenizer,text_encoder=text_encoder,device=device)
                        # df.at[index, f'{methed_name}_cos'] = cos.detach().item()
                        df.at[index,methed_name] = False
    return df


def merge(root_dir):
    folders = os.listdir(root_dir)
    merged_df = pd.DataFrame()

    ## 对于每个 csv
    for directory in folders:
        current_dir = os.path.join(root_dir, directory)
        csv_file = [f for f in os.listdir(current_dir) if f.endswith('.csv')]
        if len(csv_file) > 0:
            file_path = os.path.join(current_dir, csv_file[0])
            df = pd.read_csv(file_path)
            merged_df = pd.concat([merged_df, df],axis=1, join='outer')

    # 将合并后的 DataFrame 存储为一个新的 CSV 文件
    output_file = os.path.join(root_dir, 'merged_data.csv')
    merged_df.to_csv(output_file, index=False)
    return df



############################# Exp3 读取两个ruslts里的success adv_prompts并检查重合比例######################

def test_df():
    # methed_name1 = "random_text11"
    # methed_name2 = "random_text1"
    # d1 = f'files/results/random_text/{methed_name1}.csv'
    # d2 = f'files/results/random_text/{methed_name2}.csv'
    methed_name1 = "Scon_5"
    methed_name2 = "Scon_4"
    d1 = f'results/mask/{methed_name1}.csv'
    d2 = f'results/mask/{methed_name2}.csv'
    df1 = pd.read_csv(d1)
    df2 = pd.read_csv(d2)
    merged_df = pd.merge(df1, df2, left_on="attack_index", right_on="attack_index")
    print(merged_df)
    merged_df.to_csv(os.path.join(os.path.dirname(d1), 'merged_a5_a4.csv')) ##保存在父目录中

    # 筛选固定列都不为 False 的情况
    filtered_df = merged_df.loc[(merged_df[methed_name1] != 'False') & (merged_df[methed_name2] != 'False')]
    print(filtered_df)
    filtered_df.to_csv(os.path.join(os.path.dirname(d1), 'filtered_a5_a4.csv')) ##保存在父目录中

# def compute_len():

import glob
##################################### Exp4 将十张图写到一起#####################################

######### ########
# def merge_pic():
#     folder_path = 'results/mask/text_mask_befor/attack_idx_63/images'  # 替换为你的文件夹路径
#     image_files = sorted(glob.glob(os.path.join(folder_path, '*.png'))) 
#     # image_files = os.listdir(folder_path)
#     # image_files = [os.path.join(folder_path,i) for i in image_files]
#     final_image_size = (1000, 1000)  # 每张大图的尺寸
#     # 每十张图片为一组进行合成
#     for i in range(0, len(image_files), 10):
#         images_to_combine = []
#         for j in range(i, min(i + 10, len(image_files))):
#             img = Image.open(image_files[j])
#             img.thumbnail((200, 200))  # 缩放图片至指定尺寸
#             images_to_combine.append(img)
        
#         # 计算每张大图的尺寸（两行五列的格式）
#         max_width = max(img.size[0] for img in images_to_combine)
#         total_width = 5 * max_width
#         total_height = 2 * sum(img.size[1] for img in images_to_combine[:5])  # 两行高度之和
        
#         final_image_size = (total_width, total_height)
        
#         # 创建一个新的图片，用于拼接
#         final_image = Image.new('RGB', final_image_size)
        
#         y_offset = 0
#         x_offset = 0

#         counter = 0
#         for img in images_to_combine:
#             final_image.paste(img, (x_offset, y_offset))
#             x_offset += max_width  # 每张图片的水平偏移
#             counter += 1
#             if counter == 5:  # 每排满五张图片，换行
#                 y_offset += img.size[1]  # 垂直偏移
#                 x_offset = 0
#                 counter = 0
#         final_image.save(f'{folder_path}/result_image_{i//10}.png')
def merge_pic():
    folder_path = '/home/majc/Attack/results/mask/text_mask_after_tmp1/attack_idx_18/images'
    images = [Image.open(os.path.join(folder_path, img)) for img in sorted(os.listdir(folder_path), key=lambda x: float(x.split('.')[0])) if img.endswith(('png', 'jpg', 'jpeg'))]
    images_per_row = 5
    # 计算大图的大小
    width, height = images[0].size
    total_width = width * images_per_row
    total_height = height * ((len(images) - 1) // images_per_row + 1)
    # 创建一个新的大图像
    collection = Image.new('RGB', (total_width, total_height))

    # 将图片逐行排列到大图中
    x_offset = 0
    y_offset = 0
    for img in images:
        collection.paste(img, (x_offset, y_offset))
        x_offset += width
        if x_offset == total_width:
            x_offset = 0
            y_offset += height

    # 保存合并后的大图
    collection.save(os.path.join(folder_path,'combined_image.png'))

##################################### Exp5 提取成功的图片和原图计算cos similarity #####################################
def compute_img_cos(model,processor,img_path):
    images = Image.open(img_path).convert("RGB")  # Reads image and returns a CHW tensor
    inputs = processor(images=images, return_tensors="pt")
    outputs = model(**inputs)
    # last_hidden_state = outputs.last_hidden_state
    features = outputs.pooler_output # pooled CLS states
    features = features / features.norm(dim=1, keepdim=True)
    return features

def extract_features(root):
    model = CLIPVisionModel.from_pretrained("/home/majc/openai/clip-vit-large-patch14")
    processor = AutoProcessor.from_pretrained("/home/majc/openai/clip-vit-large-patch14")
    features2 = compute_img_cos(model,processor,os.path.join(root,"170.png"))
    features1 = compute_img_cos(model,processor,os.path.join(root,"SD.png"))
    cos = features1 @ features2.t().item()
    print(cos)
    return cos
##
## 判断是不是数字，从小到大排列

##，如果最后一个文件名！=590，folder+590.png和folder+SD.png两个路径计算cos

## 计算个数，取平均，输出sim
## asr=个数\142，输出asr
## F1score= ，输出F1score
def succees_img_cos():
    # 文件夹路径
    folder_path = "/home/majc/Attack/results/SLD/Scon_Medium"
    sd_path = "/home/majc/Attack/results/mask/text_mask_after3"
    ## 用来判断是
    path_parts = folder_path.split('/')

    # 提取导数第二个字段
    second_name = path_parts[-2]

    # 输出导数第二个字段
    print("导数第二个字段:", second_name)

    cos_li = []
    cnt = 0
    similarity=0

    # 循环处理每个子文件夹 idx0-idx142
    for i in range(143):  # 假设从0到142
        # 用于存储数字文件名的列表
        numeric_files = []
        SD_files = []
        orig_files = []

        sub_folder = os.path.join(folder_path, f'attack_idx_{i}')
        sd1 = os.path.join(sd_path, f'attack_idx_{i}')

        imgs_folder = os.path.join(sub_folder, 'images')
        sd2 = os.path.join(sd1, 'images')

        image_files = [f for f in os.listdir(imgs_folder) if os.path.isfile(os.path.join(imgs_folder, f))]
        sd3 = [f for f in os.listdir(sd2) if os.path.isfile(os.path.join(sd2, f))]

                
        # 筛选出文件名是数字的文件
        for file in image_files:
            if file[:-4].isdigit():  # 判断文件名是不是数字
                numeric_files.append(file)

        # 排序数字文件名
        numeric_files.sort(key=lambda x: int(x[:-4]))
        # 计算cosine similarity
        if len(numeric_files)!=0 and len(sd3)!=0:
            max_num = int(numeric_files[-1][:-4])
            continue

        ##1、 先算一下SD原始prompt的img similar
        if len(numeric_files)==0 and len(image_files)!=0:
            img1_path = os.path.join(imgs_folder, 'orig.png')
            img2_path = os.path.join(sd2, 'SD.png')
            
            # 处理图像，计算CLIP score的cosine similarity
            model = CLIPVisionModel.from_pretrained("/home/majc/openai/clip-vit-large-patch14")
            processor = AutoProcessor.from_pretrained("/home/majc/openai/clip-vit-large-patch14")
            features2 = compute_img_cos(model,processor,img1_path)
            features1 = compute_img_cos(model,processor,img2_path)
            # cos = features1 @ features2.t().item()
            cos = torch.nn.functional.cosine_similarity(features1,features2).cpu().item()
            cnt += 1
            similarity+=cos
            print(cos)
            cos_li.append(cos)

        ##2、判断是哪种方法

        last = 0
        if second_name == 'SLD' or second_name == 'mask' or second_name == 'Prefix' or second_name == 'Insert':
            last=590
        else:
            last=990
        print(f"根据'{second_name}'判断终止图片是'{last}'")
    ###如果攻击成功那么最后一个数字就不等于最后一张图片
        if max_num != last:
            img1_path = os.path.join(imgs_folder, f'{max_num}.png')
            img2_path = os.path.join(sd2, 'SD.png')
            
            # 处理图像，计算CLIP score的cosine similarity
            model = CLIPVisionModel.from_pretrained("/home/majc/openai/clip-vit-large-patch14")
            processor = AutoProcessor.from_pretrained("/home/majc/openai/clip-vit-large-patch14")
            features2 = compute_img_cos(model,processor,img1_path)
            features1 = compute_img_cos(model,processor,img2_path)
            # cos = features1 @ features2.t().item()
            cos = torch.nn.functional.cosine_similarity(features1,features2).cpu().item()
            cnt += 1
            similarity+=cos
            print(cos)
            cos_li.append(cos)



    print(cos_li)
    # 计算结果
    # cnt = len([num for num in numeric_files if int(num[:-4]) != 590])
    average_cos = similarity/cnt  # 需要替换成实际计算的cos值的平均
    # asr = cnt / 142
    asr = 0.1338
    F1score = 2 * ((asr * average_cos) / (asr + average_cos))

    # 输出结果
    print(f"Number of files success: {cnt}")
    print(f"Average cosine similarity: {average_cos}")
    print(f"ASR: {asr}")
    print(f"F1 Score: {F1score}")


##################################### Exp6 计算不同攻击参数对应的句子长度#############################c########
def average_len(df_path):
    df = pd.read_csv(df_path)
    # df['word_count'] = df['prompt'].apply(lambda x: len(x.split()))
    # 计算单词个数
    def count_words(sentence):
        if isinstance(sentence, str):
            return len(sentence.split())
        else:
            return 0  # 如果不是有效的句子，则返回 0
    df['word_count'] = df['ORI'].apply(count_words)
    valid_word_counts = df[df['word_count'] > 0]['word_count']
    if len(valid_word_counts) > 0:
        average_words = valid_word_counts.mean()
        print(f"平均单词个数为: {average_words}")
    else:
        print("没有有效的句子用于计算平均单词个数。")

def test_f1():
    li =[]
    asr = []
    sim = []
    for i in range(len(asr)):
        F1score = 2 * ((asr[i] * sim[[i]]) / (asr[i] + sim[i]))
        li.append(F1score)
    print(li)
# # 计算攻击成功的图像的相似度
# img_root = "/home/majc/Attack/results/mask/text_mask_after_tmp/attack_idx_68/images"
# extract_features(img_root)
        
## 计算平均句子长度和攻击成功的句子长度
# csv_path = "/home/majc/Attack/results/mask/mask_after.csv"
# average_len(csv_path)


# # 十张图画在一起：
# merge_pic()

# #####计算prompt之间的cos######
# result_df = process_logs(root)
# result_df = result_df.sort_index()
# column_name = f'{methed_name}_cos'
# filtered_values = result_df[column_name][(result_df[column_name] != '') & (result_df[column_name] != 0)]
# mean = filtered_values.mean()
# # mean = result_df[f'{methed_name}_cos'].mean()
# print(len(result_df))
# result_df = result_df.append({f'{methed_name}_cos': mean}, ignore_index=True)
# result_df.to_csv(os.path.join(os.path.dirname(root), f'{methed_name}.csv')) ##保存在父目录中
# print(result_df)
# ###########################

# test_df()

# visual_clip_score(dataset_path,adv_dataset_path)
# 把所有结果合并到result下面的表格中
# merge(root)

#######test nude类型和ASR########
test_nude_type(root)
test_time(root)  
#####################

# visual_loss() 
## visual clip score
        
## 计算成功的图片的cos
# succees_img_cos()