from .base import Attacker
import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr, spearmanr
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math,os
from transformers import AutoProcessor, CLIPVisionModel,CLIPModel

class TextSimilarity(Attacker):
    def __init__(
                self,
                lr=1e-2,
                weight_decay=0.1,
                rand_init=False,
                **kwargs
                ):
        super().__init__(**kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.rand_init = rand_init

    def bisection(self,a, eps, xi = 1e-5, ub=1):
        '''
        bisection method to find the root for the projection operation of $u$
        '''
        pa = torch.clip(a, 0, ub)
        if np.abs(torch.sum(pa).item() - eps) <= xi:
            upper_S_update = pa
        else:
            mu_l = torch.min(a-1).item()
            mu_u = torch.max(a).item()
            while np.abs(mu_u - mu_l)>xi:
                mu_a = (mu_u + mu_l)/2
                gu = torch.sum(torch.clip(a-mu_a, 0, ub)) - eps
                gu_l = torch.sum(torch.clip(a-mu_l, 0, ub)) - eps + 1e-8
                gu_u = torch.sum(torch.clip(a-mu_u, 0, ub)) - eps
                if gu == 0: 
                    break
                elif gu_l == 0:
                    mu_a = mu_l
                    break
                elif gu_u == 0:
                    mu_a = mu_u
                    break
                if gu * gu_l < 0:  
                    mu_l = mu_l
                    mu_u = mu_a
                elif gu * gu_u < 0:  
                    mu_u = mu_u
                    mu_l = mu_a
                else:
                    print(a)
                    print(gu, gu_l, gu_u)
                    raise Exception()
            upper_S_update = torch.clip(a-mu_a, 0, ub)
        return upper_S_update 

    
    def post_filter(self,task,adv_embeddings, sensitive_id_list):
        adv_embeddings.grad[0, :,sensitive_id_list] = 1e9
        return adv_embeddings

    def projection(self,curr_var,xi=1e-5):
        var_list = []
        curr_var = torch.squeeze(curr_var,dim=0)
        for i in range(curr_var.size(0)):
            projected_var = self.bisection(curr_var[i], eps=1, xi=xi, ub=1)
            var_list.append(projected_var)
        projected_var = torch.stack(var_list, dim=0).unsqueeze(0)
        return projected_var
    
    def init_adv(self, task, orig_prompt_len):
        vocab_dict = task.tokenizer.get_vocab()
        tmp = torch.zeros([1, self.k, len(vocab_dict)]).fill_(1/len(vocab_dict))
        adv_embedding = torch.nn.Parameter(tmp).to(task.device)
        if self.rand_init:
            torch.nn.init.uniform_(tmp, 0, 1)
        tmp_adv_embedding = self.projection(adv_embedding)
        # FIXME: 为社么要用.data进行赋值，而不直接赋值
        adv_embedding.data = tmp_adv_embedding.data
        # FIXME: 这个地方为什么要detach（跟前面的计算图分离）
        self.adv_embedding = adv_embedding.detach().requires_grad_(True)

    def init_opt(self):
        self.optimizer = torch.optim.Adam([self.adv_embedding],lr = self.lr,weight_decay=self.weight_decay)
        # self.optimizer = torch.optim.SGD([self.adv_embedding],lr = self.lr,weight_decay=self.weight_decay)


    def split_embd(self,input_embed,orig_prompt_len):
        sot_embd, mid_embd, _, eot_embd = torch.split(input_embed, [1, orig_prompt_len, self.k, 76-orig_prompt_len-self.k ], dim=1)
        self.sot_embd = sot_embd
        self.mid_embd = mid_embd
        self.eot_embd = eot_embd
        return sot_embd, mid_embd, eot_embd
    
    def split_id(self,input_ids,orig_prompt_len):
        sot_id, mid_id,_, eot_id = torch.split(input_ids, [1, orig_prompt_len,self.k, 76-orig_prompt_len-self.k], dim=1)
        return sot_id, mid_id, eot_id
    
    def construct_embd(self,adv_embedding):
        if self.insertion_location == 'prefix_k':     # Prepend k words before the original prompt
            embedding = torch.cat([self.sot_embd,adv_embedding,self.mid_embd,self.eot_embd],dim=1)
        elif self.insertion_location == 'suffix_k':   # Append k words after the original prompt
            embedding = torch.cat([self.sot_embd,self.mid_embd,adv_embedding,self.eot_embd],dim=1)
        elif self.insertion_location == 'mid_k':      # Insert k words in the middle of the original prompt
            embedding = [self.sot_embd,]
            total_num = self.mid_embd.size(1)
            embedding.append(self.mid_embd[:,:total_num//2,:])
            embedding.append(adv_embedding)
            embedding.append(self.mid_embd[:,total_num//2:,:])
            embedding.append(self.eot_embd)
            embedding = torch.cat(embedding,dim=1)
        elif self.insertion_location == 'insert_k':   # seperate k words into the original prompt with equal intervals
            embedding = [self.sot_embd,]
            total_num = self.mid_embd.size(1)
            internals = total_num // (self.k+1)
            for i in range(self.k):
                embedding.append(self.mid_embd[:,internals*i:internals*(i+1),:])
                embedding.append(adv_embedding[:,i,:].unsqueeze(1))
            embedding.append(self.mid_embd[:,internals*(i+1):,:])
            embedding.append(self.eot_embd)
            embedding = torch.cat(embedding,dim=1)
            
        elif self.insertion_location == 'per_k_words':
            embedding = [self.sot_embd,]
            for i in range(adv_embedding.size(1) - 1):
                embedding.append(adv_embedding[:,i,:].unsqueeze(1))
                embedding.append(self.mid_embd[:,3*i:3*(i+1),:])
            embedding.append(adv_embedding[:,-1,:].unsqueeze(1))
            embedding.append(self.mid_embd[:,3*(i+1):,:])
            embedding.append(self.eot_embd)
            embedding = torch.cat(embedding,dim=1)
        return embedding
    
    def construct_id(self,adv_id,sot_id,eot_id,mid_id):
        if self.insertion_location == 'prefix_k':
            input_ids = torch.cat([sot_id,adv_id,mid_id,eot_id],dim=1)
        elif self.insertion_location == 'suffix_k':
            input_ids = torch.cat([sot_id,mid_id,adv_id,eot_id],dim=1)
            
        elif self.insertion_location == 'mid_k':
            input_ids = [sot_id,]
            total_num = mid_id.size(1)
            input_ids.append(mid_id[:,:total_num//2])
            input_ids.append(adv_id)
            input_ids.append(mid_id[:,total_num//2:])
            input_ids.append(eot_id)
            input_ids = torch.cat(input_ids,dim=1)
            
        elif self.insertion_location == 'insert_k':
            input_ids = [sot_id,]
            total_num = mid_id.size(1)
            internals = total_num // (self.k+1)
            for i in range(self.k):
                input_ids.append(mid_id[:,internals*i:internals*(i+1)])
                input_ids.append(adv_id[:,i].unsqueeze(1))
            input_ids.append(mid_id[:,internals*(i+1):])
            input_ids.append(eot_id)
            input_ids = torch.cat(input_ids,dim=1)
            
        elif self.insertion_location == 'per_k_words':
            input_ids = [sot_id,]
            for i in range(adv_id.size(1) - 1):
                input_ids.append(adv_id[:,i].unsqueeze(1))
                input_ids.append(mid_id[:,3*i:3*(i+1)])
            input_ids.append(adv_id[:,-1].unsqueeze(1))
            input_ids.append(mid_id[:,3*(i+1):])
            input_ids.append(eot_id)
            input_ids = torch.cat(input_ids,dim=1)
        return input_ids
    
    def argmax_project(self,adv_embedding,all_embeddings,tokenizer):
        input = torch.squeeze(adv_embedding,dim=0)
        num_classes = input.size(-1)
        text = torch.argmax(input, dim=-1)
        out = text.view(-1)
        out = F.one_hot(out, num_classes = num_classes).float()
        adv_embedding = torch.unsqueeze(out,dim=0) @ all_embeddings
        return adv_embedding,torch.unsqueeze(text,0)    

    def plot_list(self,data,logger):
        plt.plot(data)  # 绘制数值列表的折线图

        plt.xlabel('iter num')
        plt.ylabel('Loss')
        plt.title('Loss——Trendancy')
        path = os.path.join(logger.img_root, 'loss_fig.png')
        plt.savefig(path)  # 保存图形为指定文件名

    def measure_cos(self,task,prompt1,prompt2,tokenizer,text_encoder):
        adv_ids = tokenizer(prompt1,padding=True, return_tensors="pt").to(task.device)
        adv_output = text_encoder(**adv_ids)
        pooled_output = adv_output.pooler_output
        nude_ids = tokenizer(prompt2,padding=True, return_tensors="pt").to(task.device)
        nude_output = text_encoder(**nude_ids)
        nude_pooled_output = nude_output.pooler_output
        ## calculate cos
        pooled_output = pooled_output / pooled_output.norm(dim=1, keepdim=True)
        nude_pooled_output = nude_pooled_output / nude_pooled_output.norm(dim=1, keepdim=True)

        cos = pooled_output @ nude_pooled_output.t()
        return cos
    
    def get_pooled_emb(self,task,prompt1,tokenizer,text_encoder):
        adv_ids = tokenizer(prompt1,padding=True, return_tensors="pt").to(task.device)
        adv_output = text_encoder(**adv_ids)
        pooled_output = adv_output.pooler_output
        return pooled_output
    
    def get_emb(self,task,prompt1,tokenizer,text_encoder):
        adv_ids = tokenizer(prompt1,padding=True, return_tensors="pt").to(task.device)
        adv_output = text_encoder(**adv_ids)
        output = adv_output.last_hidden_state
        return output


    def run(self, task, logger):
  

        image,img_path,prompt, seed, guidance = task.dataset[self.attack_idx]
        # prompt, seed, guidance = task.dataset[self.attack_idx]

        ## test用

        # prompt = "A photo of a King with crown"
        # prompt = "A photo of a red apple"
        # prompt = "A photo of a red apple"
        # prompt = "A photo of a dog"
        # prompt = "A photo of a red apple"

        print(prompt)
        if seed is None:
            seed = self.eval_seed
            
        task.tokenizer.pad_token = task.tokenizer.eos_token
        ## Get original i        
        viusalize_prompt_id = task.str2id(prompt)
        visualize_orig_prompt_len = (viusalize_prompt_id == 49407).nonzero(as_tuple=True)[1][0]-1

        # if visualize_orig_prompt_len>68:
        #     viusalize_prompt_id[0][69:] = 49407
        #     visualize_orig_prompt_len = 68
            
        visualize_embedding = task.id2embedding(viusalize_prompt_id)

        print((viusalize_prompt_id == 49407).nonzero(as_tuple=True))
        print(visualize_orig_prompt_len)



        self.k = min(76-visualize_orig_prompt_len,self.k)

            
        print(self.k)
        self.init_adv(task, visualize_orig_prompt_len.item())
        self.init_opt() 
        visualize_sot_id, visualize_mid_id, visualize_eot_id = self.split_id(viusalize_prompt_id,visualize_orig_prompt_len)

        ### Visualization for the original prompt:
        t = -1
        results = task.eval(logger,t,viusalize_prompt_id,prompt,seed=seed,guidance_scale=guidance)
        results['prompt'] = prompt
        logger.save_img('orig', results.pop('image'))
        to_pil = transforms.ToPILImage()
        pil_img = to_pil(image)
        logger.save_img('SD', pil_img)
        logger.log(results)    
        # if results.get('success') is not None and results['success']:
        #     return 0  
        
        if not self.universal:
            if seed is None:
                seed = self.eval_seed
            # x0 = task.img2latent(image)
            # image_int = image.unsqueeze(0)
            input_ids = task.str2id(prompt)
            orig_prompt_len = (input_ids == 49407).nonzero(as_tuple=True)[1][0]-1
            input_embeddings_ori = task.id2embedding(input_ids)
            task.mask = task.mask.view(-1)
            input_embeddings = input_embeddings_ori
# #### defense 插件
#             id = task.str2id("nudity")
#             concept_embeddings = task.id2embedding(id)
#             _,cos = task.text_custom_cos(input_ids,input_ids,input_embeddings_ori,concept_embeddings)
#             print("cos:",cos)
#             a = 7788272950850468
#             b = -18.57116356422019
#             task.nude_weight = a*cos+b
#             print("nude_weight:",task.nude_weight)
# '''
            if task.mode =="before and after":
                ### 手动mask前self.k个字符
                # task.mask[:,1:self.k+1,:] = task.mask[:,1:2,:].expand(-1,self.k, -1)
                # print(task.mask[0,1:self.k+1,].sum())
                task.mask = task.mask.view(-1)
                ### nude embedding手动加到每个单词上面
                task.nude_embeddings[:, 1:visualize_orig_prompt_len+1, :] = task.nude_embeddings[:,1:2,:].expand(-1, visualize_orig_prompt_len, -1)
                input_embeddings =input_embeddings_ori + task.nude_weight*task.nude_embeddings
                print(task.nude_embeddings)

                # _,cos = task.text_custom_cos(input_ids,input_ids,input_embeddings_ori,input_embeddings)
                # print("cos:",cos)


                li = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7]
                for i in li:
                    sen_embed = input_embeddings_ori+i * task.nude_embeddings
                    results = task.embedding_eval(viusalize_prompt_id,sen_embed,seed=seed,guidance_scale=guidance)
                    logger.save_img(f'hard_emb{i}', results.pop('image'))
                return 0
# ####### 这两个和上面的情况唯一的区别是 ，下面的mask用的是 过了attention的mask
#             elif task.mode == "before and after":
#                 task.mask = task.mask.view(-1)
#                 task.nude_embeddings[:, 1:visualize_orig_prompt_len+1, :] = task.nude_embeddings[:,1:2,:].expand(-1, visualize_orig_prompt_len, -1)
#                 # task.nude_embeddings[:, visualize_orig_prompt_len-1:visualize_orig_prompt_len+1, :] = task.nude_embeddings[:,1:2,:].expand(-1, 1, -1)
#                 # input_embeddings = input_embeddings_ori + 4.0*task.nude_embeddings    
#             # else:

#             #     sen_embed = task.custom_text_encoder(input_ids = input_ids,inputs_embeds=input_embeddings)[0]
#                 # li = [1.5,1.9,2.4,2.8,3.2]
#             li = [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8]
#             for i in li:
#                 sen_embed = input_embeddings_ori+i * task.nude_embeddings
#                 results = task.embedding_eval(viusalize_prompt_id,sen_embed,seed=seed,guidance_scale=guidance)
#                 logger.save_img(f'soft_emb{i}', results.pop('image'))
            return 0
                # for i in li:
                #     sen_embed = (sen_embed+i * task.nude_embeddings)/(1+i)
                #     results = task.embedding_eval(viusalize_prompt_id,sen_embed,seed=seed,guidance_scale=guidance)
                #     logger.save_img(f'soft_emb{i}', results.pop('image'))
    #         ## get new embeddings
    #    #######假设3：pipeline推理prompt #######
    #         cos_list =[]
    #         output_folder = f"/home/majc/Attack/results/text_similarity_esd_vangogh_classifier/attack_idx_15"
    #         os.makedirs(output_folder, exist_ok=True)

    #         ##################compute cos ########################
    #         p1 = prompt
    #         p2 = prompt.replace("Vincent van Gogh",f"{task.inverted_tokens}")
    #         cos = self.measure_cos(task,p1,p2,tokenizer=task.pipe.tokenizer,text_encoder=task.pipe.text_encoder)
    #         print(cos)
    #         cos_list.append(cos.detach().item())
    #         #####################save image#######################
    #         seed = seed
    #         torch.manual_seed(seed)
    #         print(f'Inferencing: {p2}')
    #         im = task.pipe(p2, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=1).images[0]
    #         im.save(f"{output_folder}/inversion.jpg") 
            
            ##(repeat single token and add to origin)
            # task.nude_embeddings[:, visualize_orig_prompt_len-2:visualize_orig_prompt_len+2, :] = first_value.expand(-1, visualize_orig_prompt_len, -1)
            # input_embeddings =input_embeddings_ori + task.nude_weight* task.nude_embeddings + task.nude_class_weight*task.nude_class_embeddings

#########假设5：知识按程度注入embdding#########
            ###########假设4：用inversion token作为引导#########

            ##------假设4.2：用inversion之后的整句话作为引导
            # p2 = prompt.replace("Vincent van Gogh",f"{task.inverted_tokens}")
            ##------假设4.1 ：用inverted token作为引导
            # p2 = task.inverted_tokens
            ##------假设4.3：用inversion token 做attention然后

            # first_value = self.get_emb(task,p2,tokenizer=task.pipe.tokenizer,text_encoder=task.pipe.text_encoder)
            # len = first_value.shape[1]
            # add_embedding = first_value[:,1:len-1,:]

            ## (repeat single token and add to origin)
            # first_value = 
            # task.nude_embeddings[:, 1:visualize_orig_prompt_len+1, :] = task.nude_embeddings[:,1:2,:].expand(-1, visualize_orig_prompt_len, -1)
    #     #### Exp1：加权 不行
    #     #### Exp2：加程度
    #     #### Exp3：nude/近义词/相减
    #         # li = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    #         # li = [0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]
                

            # li = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]#0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,1.0,
            # li = [0.46,0.461,0.462,0.463,0.464,0.465,0.466,0.467,0.468,0.469,0.470,0.48,0.49,0.5]
            # li = [0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.50,0.51,0.52,0.6,0.7]
    #         x_=[]
    #         cos_=[]
    #         #0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
    #         for i in li:
    #             input_embeddings =input_embeddings_ori + i*task.nude_embeddings
    #             # _,text_cos = task.text_custom_cos(input_ids,input_ids,input_embeddings_ori,input_embeddings)
    #             # print("cos:",cos)
    # # ################## test embedding ##############
    #             results = task.embedding_eval(viusalize_prompt_id,input_embeddings,seed=seed,guidance_scale=guidance)

                # im = results["image"]
                # logger.save_img(i*100, results.pop('image'))

    ###### 取出目标路径下的图片 ####
                # folder_path = "/home/majc/Attack/results/mask/text_mask_after_tmp1/attack_idx_6/images"
                # target_filename = f'{float(i*10)}.png'  # 目标文件名
                # image_files = [img for img in os.listdir(folder_path) if img.endswith(('png', 'jpg', 'jpeg'))]
                # for img_filename in image_files:
                #     if img_filename == target_filename:
                #         img_path = os.path.join(folder_path, img_filename)
                #         im = Image.open(img_path)
    # ###### 文本图片计算相似度部分
    #             inputs = task.tokenizer("smiling", padding=True, return_tensors="pt").to(task.device)
    #             text_features = model.get_text_features(**inputs)

    #             inputs = processor(images=im, return_tensors="pt").to(task.device)
    #             image_features = model.get_image_features(**inputs)
    #             text_features = text_features / text_features.norm(dim=1, keepdim=True)
    #             image_features = image_features / image_features.norm(dim=1, keepdim=True)
    #             cos = image_features @ text_features.t()
    #             cos = cos.detach().cpu().item()
    #             x_.append(i)
    #             cos_.append(cos)
            

            # ## 画图
            # print(cos_)
            # # plt.scatter(x_, cos_)  # 创建散点图
            # plt.figure()  # 创建新的图表
            # plt.plot(x_, cos_)  # 创建散点图
            # plt.xlabel('concept')  # 设置横坐标标签
            # plt.ylabel('Clip score')  # 设置纵坐标标签
            # plt.title('SD')  # 设置图表标题
            # plt.savefig("SD_plot.png") 
            # plt.savefig("/home/majc/Attack/results/mask/text_mask_after_tmp1/attack_idx_1/SD_plot.png") 
            
            # plt.figure()  # 创建新的图表
            # plt.scatter(x_, cos_)  # 创建散点图
            # plt.xlabel('concept')  # 设置横坐标标签
            # plt.ylabel('Clip score')  # 设置纵坐标标签
            # plt.title('SD')  # 设置图表标题
            # plt.savefig("SD_scatter.png") 
            # plt.savefig("/home/majc/Attack/results/mask/text_mask_after_tmp1/attack_idx_1/SD_scatter.png")
            
            # plt.figure()  # 创建新的图表
            # plt.scatter(x_, cos_, label='SLD-Max', color='navy')
            # # 一次函数拟合
            # coefficients = np.polyfit(x_, cos_, 1)  # 一次函数拟合
            # poly_function = np.poly1d(coefficients)
            # x_fit = np.linspace(min(x_), max(x_), 100)  # 生成用于绘制拟合直线的 x 值
            # y_fit = poly_function(x_fit)  # 计算拟合直线的 y 值
            # # 绘制拟合直线
            # plt.plot(x_fit, y_fit, label='Fitted Line', color='gray')
            # # 设置标签和标题
            # plt.xlabel('Concept')
            # plt.ylabel('Clip score')
            # plt.title('SLD-Max')
            # plt.savefig("/home/majc/Attack/results/mask/text_mask_after_tmp1/SD_plot01.png") 


            # # 添加图例
            # plt.legend()
            # pearson_corr, _ = pearsonr(x_, cos_)
            # print(f"Pearson correlation coefficient: {pearson_corr}")
            # spearman_corr, _ = spearmanr(x_, cos_)
            # print(f"Spearman correlation coefficient: {spearman_corr}")

    #             if results.get('success') is not None and results['success']:
    #                 logger.log(results)
    #             #     return 0  
            
                # results = task.embedding_eval(viusalize_prompt_id, i*task.nude_embeddings,seed=seed,guidance_scale=guidance)
    #             logger.save_img(f'img_nude{i}', results.pop('image'))
    #             if results.get('success') is not None and results['success']:
    #                 logger.log(results)
    #                 return 0  
    #             _,cos = task.text_custom_cos(input_ids,input_ids,input_embeddings_ori,input_embeddings)
    #             print("cos:",cos)
            sensitive_id_list = []  
            # sensitive_word_list = ["sexual","nude","sex","18+","naked", "nsfw" ,"porn","dick","vagina","explicit","uncensored","fuck","nipples","breast","areola",]
            sensitive_word_list = [
    "sexual", "blood", "nude", "bleed", "nudity", "corpse", "sex", "hang", "18+", 
    "flesh", "naked", "creep", "nsfw", "horror", "porn", "demon", "dick", "dead", 
    "vagina", "death", "naked", "child", "deform", "explicit", "content", "pepe", 
    "unsensored", "nazi", "fuck", "confederate", "nipples", "breast", "biden", "areola", 
    "pussy", "pelosi", "penis", "jew", "vagina", "asian", "suck", "african", 
    "cock", "asia", "blowjock", "africa", "blow","zombi","nudes","anus","erotic","bloody"
]
            for i in sensitive_word_list:
                sensitive_id = task.tokenizer(i, padding=True, max_length=task.tokenizer.model_max_length, return_tensors="pt",truncation=True).input_ids[0,1].item()
                sensitive_id_list.append(sensitive_id)
            sensitive_id_list = torch.tensor(sensitive_id_list)
            print(f"==>> sensitive_id_list: {sensitive_id_list}")


            self.split_embd(input_embeddings_ori,orig_prompt_len)

            # if results.get('success') is not None and results['success']:
            #     logger.log(results)
            #     return 0  
                
            if self.sequential:
                total_loss_list = []
                for t in task.sampled_t:
                    total_loss = 0
                    loss_list = []
                    for i in range(self.iteration):
                        self.optimizer.zero_grad()
                        adv_one_hot = STERandSelect.apply(self.adv_embedding)
                        tmp_embeds = adv_one_hot @ task.all_embeddings
                        adv_input_embeddings = self.construct_embd(tmp_embeds)
                        input_arguments = {"input_ids":input_ids,"adv_input_ids":input_ids,"input_embeddings":input_embeddings,"adv_input_embeddings":adv_input_embeddings}
                        loss,t2t_sim = task.get_mask_loss(**input_arguments)
                        print("loss:",loss)
                        self.adv_embedding.grad = torch.autograd.grad(loss, [self.adv_embedding])[0]
                        torch.nn.utils.clip_grad_norm_([self.adv_embedding], max_norm=1)
                        # self.adv_embedding = self.post_filter(task,self.adv_embedding, sensitive_id_list)

                        self.optimizer.step()
                        proj_adv_embedding = self.projection(self.adv_embedding)
                        self.adv_embedding.data = proj_adv_embedding.data
                        total_loss += loss.item() 
                        loss_list.append(loss.item())
                        total_loss_list.append(loss.item())
                    ## Get final all prompt
                    _, proj_ids = self.argmax_project(self.adv_embedding,task.all_embeddings,task.tokenizer) 
                    new_visualize_id = self.construct_id(proj_ids,visualize_sot_id,visualize_eot_id,visualize_mid_id)
                    id_list = new_visualize_id[0][1:].tolist()
                    id_list = [id for id in id_list if id!=task.tokenizer.eos_token_id]
                    new_visualize_prompt = task.tokenizer.decode(id_list)


                    ## Check if attack work

                    results = task.eval(logger,t,new_visualize_id,new_visualize_prompt,seed,guidance_scale=guidance)
                    results['prompt'] = new_visualize_prompt
                    results['loss'] = total_loss
                    results['loss_list'] = loss_list
                    logger.save_img(f'{t}', results.pop('image'))
                    logger.log(results)
                    # if results.get('success') is not None and results['success']:
                    #     break 

                print(total_loss_list)
                print(total_loss_list[0])
                self.plot_list(total_loss_list,logger)

    
            else:
                raise NotImplementedError                   

        else:
            raise NotImplementedError


class STERandSelect(torch.autograd.Function):  
    @staticmethod                               
    def forward(ctx, input):
        input = torch.squeeze(input,dim=0)
        num_classes = input.size(-1)
        res = torch.multinomial(input,1)
        out = res.view(-1)
        out = F.one_hot(out, num_classes = num_classes).float()
        out = torch.unsqueeze(out,dim=0)
        return out


    @staticmethod
    def backward(ctx, grad_output):
        output = F.hardtanh(grad_output).unsqueeze(0)
        return output


def get(**kwargs):
    return TextSimilarity(**kwargs)