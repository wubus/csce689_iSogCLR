import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

import pickle
import argparse

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

from models.model_clip import CLIP
from transformers import AutoTokenizer, RobertaTokenizer

import utils
import shutil
from dataset import create_train_dataset, create_val_dataset, create_sampler, create_train_loader, create_val_loader
from scheduler import create_scheduler
from optim import create_optimizer
from zeroshot_transfer.classes import CIFAR10_CLASSES, CIFAR100_CLASSES, IMAGENET_CLASSES


from tqdm import tqdm


def train(model, data_loader, optimizer, tokenizer, epoch, max_epoch, warmup_steps, device, scheduler, grad_scaler, args):
    # train
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_temp_net', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('avg_image_tau', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('avg_text_tau', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('cur_eta', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('grad_tau_image', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('grad_tau_text', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('b_I', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('b_T', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('v', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('lamda', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('weights_image_pos', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('weights_text_pos', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size  

    for i,(image, text, idx, text_idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad()

        image = image.to(device, non_blocking=True)   
        idx = idx.to(device, non_blocking=True)
        text_idx = text_idx.to(device, non_blocking=True)   
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device)
        
        # set learning rate for temperature network
        optimizer.param_groups[2]["lr"] = optimizer.param_groups[0]["lr"] / 10.0

        if grad_scaler is None:
            loss_ita, info_dict = model(image, text_input, idx=idx, text_idx=text_idx, epoch=epoch, max_epoch=max_epoch)
            loss_ita.backward()
            optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                loss_ita, info_dict = model(image, text_input, idx=idx, text_idx=text_idx, epoch=epoch, max_epoch=max_epoch)
            grad_scaler.scale(loss_ita).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        
        metric_logger.update(loss_ita=loss_ita.item())

        if args.ita_type in ['sogclr_dro', 'isogclr_new']:
            metric_logger.update(avg_image_tau=info_dict['avg_image_tau'])
            metric_logger.update(avg_text_tau=info_dict['avg_text_tau'])
            metric_logger.update(cur_eta=info_dict['cur_eta'])
            metric_logger.update(grad_tau_image=info_dict['grad_tau_image'])
            metric_logger.update(grad_tau_text=info_dict['grad_tau_text'])
            metric_logger.update(b_I=info_dict['b_I'])
            metric_logger.update(b_T=info_dict['b_T'])
            metric_logger.update(weights_image_pos=0.0)
            metric_logger.update(weights_text_pos=0.0)
            metric_logger.update(v=0.0)
            metric_logger.update(lamda=0.0)
        elif args.ita_type == 'isogclr_new_v2':
            metric_logger.update(avg_image_tau=info_dict['avg_image_tau'])
            metric_logger.update(avg_text_tau=info_dict['avg_text_tau'])
            metric_logger.update(cur_eta=info_dict['cur_eta'])
            metric_logger.update(grad_tau_image=info_dict['grad_tau_image'])
            metric_logger.update(grad_tau_text=info_dict['grad_tau_text'])
            metric_logger.update(b_I=info_dict['b_I'])
            metric_logger.update(b_T=info_dict['b_T'])
            metric_logger.update(weights_image_pos=0.0)
            metric_logger.update(weights_text_pos=0.0)
            metric_logger.update(v=info_dict['v'])
            metric_logger.update(lamda=info_dict['lamda'])
        elif args.ita_type == 'sogclr':
            metric_logger.update(avg_image_tau=info_dict['avg_image_tau'])
            metric_logger.update(avg_text_tau=info_dict['avg_text_tau'])
            metric_logger.update(weights_image_pos=0.0)
            metric_logger.update(weights_text_pos=0.0)
            metric_logger.update(cur_eta=0.0)
            metric_logger.update(grad_tau_image=0.0)
            metric_logger.update(grad_tau_text=0.0)
            metric_logger.update(b_I=0.0)
            metric_logger.update(b_T=0.0)
            metric_logger.update(v=0.0)
            metric_logger.update(lamda=info_dict['lamda'])
        else:
            metric_logger.update(avg_image_tau=info_dict['avg_image_tau'])
            metric_logger.update(avg_text_tau=info_dict['avg_text_tau'])
            metric_logger.update(cur_eta=0.0)
            metric_logger.update(grad_tau_image=0.0)
            metric_logger.update(grad_tau_text=0.0)
            metric_logger.update(weights_image_pos=0.0)
            metric_logger.update(weights_text_pos=0.0)
            metric_logger.update(b_I=0.0)
            metric_logger.update(b_T=0.0)
            metric_logger.update(v=0.0)
            metric_logger.update(lamda=0.0)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_temp_net=optimizer.param_groups[2]["lr"])
        if epoch==0 and i%step_size==0 and i<=warmup_iterations and scheduler is not None: 
            scheduler.step(i//step_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


"""
    zero-shot transfer
    https://github.com/goel-shashank/CyCLIP/blob/52d77af2a5f1a4bff01b4c371d6b98e2d0340137/src/evaluate.py#L42
"""
def create_zeroshot_dataloader(dataset_name, data_folder, image_size):
    assert dataset_name in ['cifar10', 'cifar100', 'imagenet']

    if dataset_name == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset_name == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean=mean, std=std)

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root=data_folder, download=False, train=False, transform=val_transform)
    elif dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(root=data_folder, download=False, train=False, transform=val_transform)
    else:
        dataset = datasets.ImageFolder(root=data_folder, transform=val_transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False,
                                              num_workers=2, pin_memory=True)

    data_loader.num_samples = len(dataset)

    return data_loader



@torch.no_grad()
def zeroshot_transfer(model, data_loader, dataset_name, tokenizer, device):
    model.eval()

    if dataset_name == 'cifar10':
        config = CIFAR10_CLASSES
    elif dataset_name == 'cifar100':
        config = CIFAR100_CLASSES
    elif dataset_name == 'imagenet':
        config = IMAGENET_CLASSES
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    classes, templates = config["classes"], config["templates"]

    text_embeddings = []
    for c in classes:
        texts = [template(c) for template in templates]
        text_inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device) 
        text_outputs = model.text_encoder(text_inputs.input_ids, attention_mask=text_inputs.attention_mask, output_hidden_states=False)  
        text_embeds = F.normalize(model.text_proj(text_outputs.last_hidden_state[:,0,:]), dim=-1)
        text_embed = text_embeds.mean(dim=0)
        text_embed /= text_embed.norm()
        text_embeddings.append(text_embed)

    text_embeddings = torch.stack(text_embeddings, dim=1).to(device)

    topk = [1, 3, 5, 10]
    correct = {k: 0 for k in topk}

    for image, label in data_loader:
        image, label = image.to(device), label.to(device)
        image_feat = model.visual_encoder(image)        
        image_embed = model.vision_proj(image_feat)            
        image_embedding = F.normalize(image_embed, dim=-1)

        logits = image_embedding @ text_embeddings
        ranks = logits.topk(max(topk), 1)[1].T
        predictions = ranks == label

        for k in topk:
            correct[k] += torch.sum(torch.any(predictions[:k], dim=0)).item()

    results = {f"zeroshot_top{k}": 100.0 * correct[k] / data_loader.num_samples for k in topk}

    return results



@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, args):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_embeds = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, output_hidden_states=False)  
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]), dim=-1)
        text_embeds.append(text_embed)
    text_embeds = torch.cat(text_embeds,dim=0)
    
    image_embeds = []
    for image, img_id in data_loader: 
        image = image.to(device) 
        image_feat = model.visual_encoder(image)        
        image_embed = model.vision_proj(image_feat)            
        image_embed = F.normalize(image_embed, dim=-1)      
        image_embeds.append(image_embed)
    image_embeds = torch.cat(image_embeds,dim=0)
    
    sims_matrix = image_embeds.to(device) @ text_embeds.to(device).t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image),len(texts)),-100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)

    for i,sims in enumerate(sims_matrix[start:end]): 
        topk_sim, topk_idx = sims.topk(k=args.k_test, dim=0)
        score_matrix_i2t[start+i, topk_idx] = topk_sim
        
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    
    
    for i,sims in enumerate(sims_matrix[start:end]): 
        topk_sim, topk_idx = sims.topk(k=args.k_test, dim=0)
        score_matrix_t2i[start+i, topk_idx] = topk_sim

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


            
@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result




def main(args):
    if args.distributed:
        utils.init_distributed_mode(args)    
    else:
        args.gpu = 0
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating retrieval dataset")
    train_dataset = create_train_dataset('re', args)
    # val_coco_dataset, test_coco_dataset = create_val_dataset('re', args, args.val_coco_file, args.coco_image_root, args.test_coco_file)
    val_coco_dataset = create_val_dataset('re', args, args.val_coco_file, args.coco_image_root, None)
    # val_flickr_dataset, test_flickr_dataset = create_val_dataset('re', args, args.val_flickr_file, args.flickr_image_root, args.test_flickr_file)
    # sbu_dataset = create_val_dataset('re', args, args.sbu_file, args.sbu_image_root)
    print("len of train_dataset:", len(train_dataset))
    # print("len of coco val/test:", len(val_coco_dataset), len(test_coco_dataset))
    print("len of coco val:", len(val_coco_dataset))
    # print("len of flickr val/test:", len(val_flickr_dataset), len(test_flickr_dataset))
    # print("len of sbu data:", len(sbu_dataset))

    if args.extract_data:
        idx_list = []
        data_dir = os.path.join(args.output_dir, '')
        Path(data_dir).mkdir(parents=True, exist_ok=True)

        for idx in tqdm(idx_list):
            image, text, _, _ = train_dataset.__getitem__(idx, enable_transform=False)
            torchvision.utils.save_image(image, fp=os.path.join(data_dir, str(idx)+':'+text+'.png'))
            
        shutil.make_archive(data_dir, 'zip', data_dir)

        assert 0

    num_training = int(args.train_frac * len(train_dataset))
    train_dataset = Subset(train_dataset, list(range(num_training)))


    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]

    train_loader = create_train_loader(train_dataset, samplers[0], args.batch_size_train, 2, None)

    # val_coco_loader, test_coco_loader = create_val_loader([val_coco_dataset, test_coco_dataset], samplers[1:], 
    #                                                       [args.batch_size_test]*2, [8]*2, [None]*2)
    val_coco_loader = create_val_loader([val_coco_dataset], samplers[1:2], 
                                        [args.batch_size_test], [8], [None])[0]
    # val_flickr_loader, test_flickr_loader = create_val_loader([val_flickr_dataset, test_flickr_dataset], samplers[1:], 
    #                                                           [args.batch_size_test]*2, [8]*2, [None]*2)
    # sbu_loader= create_val_loader([sbu_dataset], [None], [args.batch_size_test], [32], [None])[0]
       
    if args.text_encoder == 'roberta-large':
        tokenizer = RobertaTokenizer.from_pretrained(args.text_encoder)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)

    #### Zero-shot transfer ####
    if args.zs_dataset:
        zeroshot_dataloader = create_zeroshot_dataloader(dataset_name=args.zs_dataset, data_folder=args.zs_datafolder, image_size=args.image_res)
    else:
        zeroshot_dataloader = None

    #### Model #### 
    print("Creating model")
    model = CLIP(image_encoder=args.image_encoder, text_encoder=args.text_encoder, embed_dim=args.embed_dim, init_model=args.init_model, bsz=args.batch_size_train*args.world_size,
                  world_size=args.world_size, ita_type=args.ita_type, sogclr_gamma=args.sogclr_gamma, rho_I=args.rho_I, rho_T=args.rho_T, tau_init=args.tau_init,
                  eta_init=args.eta_init, beta_u=args.beta_u, temp=args.temp, learnable_temp=args.learnable_temp,
                  vicreg_sim_coeff=args.vicreg_sim_coeff, vicreg_std_coeff=args.vicreg_std_coeff, personalized_tau=args.personalized_tau, 
                  use_temp_net=args.isogclr_temp_net, alpha=args.alpha, distributed=args.distributed)
    model = model.to(device)

    if args.evaluate or args.ita_type == 'isogclr_denoise':
        assert len(args.checkpoint) > 0
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']             
        model.load_state_dict(state_dict, strict=False)  
        print('load checkpoint from %s' % args.checkpoint)

    if args.check_samples_tau:
        image_tau_array = []
        text_tau_array = []

        model.eval() 
    
        with torch.no_grad():
            for image, text, idx, text_idx in tqdm(train_loader):
                image = image.to(device)
                text = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device)

                image_feat = F.normalize(model.vision_proj(model.visual_encoder(image)), dim=-1)
                text_output = model.text_encoder(text.input_ids, attention_mask=text.attention_mask, output_hidden_states=False)
                text_feat = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]), dim=-1)
            
                tau_image = model.criterion.image_temp_gen(image_feat).cpu().squeeze().numpy()
                tau_text = model.criterion.text_temp_gen(text_feat).cpu().squeeze().numpy()

                image_tau_array.append(tau_image)
                text_tau_array.append(tau_text)

            image_tau_array = np.concatenate(image_tau_array) 
            text_tau_array = np.concatenate(text_tau_array)

        with open(os.path.join(args.output_dir, "tau.pkl"), "wb") as f:
            pickle.dump({"tau_image":image_tau_array, "tau_text":text_tau_array}, f, protocol=pickle.HIGHEST_PROTOCOL)

        assert 0

    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    if args.use_amp:
        grad_scaler = torch.cuda.amp.GradScaler()
    else:
        grad_scaler = None

    max_epoch = args.epochs
    warmup_steps = args.warmup_epochs
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()    
    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, max_epoch, warmup_steps, device, lr_scheduler, 
                                grad_scaler, args)
            
        if args.evaluate:
            score_val_i2t_coco, score_val_t2i_coco = evaluation(model_without_ddp, val_coco_loader, tokenizer, device, args)
            # score_test_i2t_coco, score_test_t2i_coco = evaluation(model_without_ddp, test_coco_loader, tokenizer, device, args)

            # score_val_i2t_flickr, score_val_t2i_flickr = evaluation(model_without_ddp, val_flickr_loader, tokenizer, device, args)
            # score_test_i2t_flickr, score_test_t2i_flickr = evaluation(model_without_ddp, test_flickr_loader, tokenizer, device, args)
    
        if utils.is_main_process():  

            if args.evaluate:
                val_result_coco = itm_eval(score_val_i2t_coco, score_val_t2i_coco, val_coco_loader.dataset.txt2img, val_coco_loader.dataset.img2txt)  
                print("coco val:", val_result_coco)
                # test_result_coco = itm_eval(score_test_i2t_coco, score_test_t2i_coco, test_coco_loader.dataset.txt2img, test_coco_loader.dataset.img2txt)    
                # print("coco test:", test_result_coco)

                if args.zs_dataset:
                    zeroshot_results = zeroshot_transfer(model_without_ddp, zeroshot_dataloader, args.zs_dataset, tokenizer, device)
                    print("zeroshot:", zeroshot_results)
                else:
                    zeroshot_results = None

                # val_result_flickr = itm_eval(score_val_i2t_flickr, score_val_t2i_flickr, val_flickr_loader.dataset.txt2img, val_flickr_loader.dataset.img2txt)  
                # print("flickr val:", val_result_flickr)
                # test_result_flickr = itm_eval(score_test_i2t_flickr, score_test_t2i_flickr, test_flickr_loader.dataset.txt2img, test_flickr_loader.dataset.img2txt)    
                # print("flickr test:", test_result_flickr)

            # save tau for visualization
            if not args.evaluate and args.store_tau and (epoch+1)%10==0:
                print("saving tau...")
                tau_image = model_without_ddp.criterion.tau_I.clone().cpu().numpy()
                tau_text = model_without_ddp.criterion.tau_T.clone().cpu().numpy()

                with open(os.path.join(args.output_dir, "tau_"+str(epoch)+".pkl"), "wb") as f:
                    pickle.dump({"tau_image":tau_image, "tau_text":tau_text}, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if args.evaluate:                
                log_stats = {**{f'val_{k}': v for k, v in val_result_coco.items()},
                             # **{f'test_{k}': v for k, v in test_result_coco.items()},                  
                             'epoch': epoch,
                             'data': 'coco',
                            }
                with open(os.path.join(args.output_dir, "coco_log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")    

                # log_stats = {**{f'val_{k}': v for k, v in val_result_flickr.items()},
                #              **{f'test_{k}': v for k, v in test_result_flickr.items()},                  
                #              'epoch': epoch,
                #              'data': 'flickr',
                #             }
                # with open(os.path.join(args.output_dir, "flickr_log.txt"),"a") as f:
                #     f.write(json.dumps(log_stats) + "\n") 

                if zeroshot_results:
                    with open(os.path.join(args.output_dir, f"zeroshot_{args.zs_dataset}_log.txt"), "a") as f:
                        f.write(json.dumps(zeroshot_results) + "\n")

            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            #  **{f'val_{k}': v for k, v in val_result_coco.items()},
                             # **{f'test_{k}': v for k, v in test_result_coco.items()},                  
                             'epoch': epoch,
                             'data': 'coco',
                            }
                with open(os.path.join(args.output_dir, "coco_log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # if val_result_coco['r_mean'] > best:
                #     save_obj = {
                #         'model': model_without_ddp.state_dict(),
                #         'optimizer': optimizer.state_dict(),
                #         'lr_scheduler': lr_scheduler.state_dict(),
                #         'args': args,
                #         'epoch': epoch,
                #     }
                #     torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))  
                #     best = val_result_coco['r_mean']    
                #     best_epoch = epoch

                save_obj = {
                    'model': model_without_ddp.state_dict()
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_'+str(epoch+1)+'.pth'))
                    
        if args.evaluate: 
            break
           
        lr_scheduler.step(epoch+warmup_steps+1)  
        if args.distributed:
            dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "coco_log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)             

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data path
    parser.add_argument('--data_path', default='./datasets')
    parser.add_argument('--ann_path', default='./clip_train')
    parser.add_argument('--train_file', default='downstream/cc3m_train_new.json')
    parser.add_argument('--train_image_root', default='cc3m')

    # model config
    parser.add_argument('--bert_config', default='configs/config_bert.json')
    parser.add_argument('--image_encoder', default='resnet50')
    parser.add_argument('--text_encoder', default='distilbert-base-uncased')
    parser.add_argument('--image_res', default=256, type=int)
    parser.add_argument('--vision_width', default=768, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)

    # optimizer and schedular
    parser.add_argument('--opt', default='AdamW')
    parser.add_argument('--sched', default='cosine')
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_temp_net', default=1e-3, type=float)
    parser.add_argument('--wd_temp_net', default=1e-3, type=float,
                        help='weight decay for temperature network')
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--warmup', default=True, type=bool)
    parser.add_argument('--warmup_lr', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=0.02, type=float)
    parser.add_argument('--decay_rate', default=1, type=float)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--cooldown_epochs', default=0, type=int)

    # training & test settings
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--init_model', action='store_true')
    parser.add_argument('--batch_size_train', default=128, type=int)
    parser.add_argument('--batch_size_test', default=128, type=int)
    parser.add_argument('--k_test', default=256, type=int)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--no-distributed', dest='distributed', action='store_false')

    # output path
    parser.add_argument('--output_dir', default='./output/clip_test')  

    # loss config
    parser.add_argument('--ita_type', required=True, choices=['clip', 'cyclip', 'vicreg', 'sogclr', 'sogclr_dro', 
                        'isogclr_new_v2', 'isogclr_new_v1', 'isogclr_new', 'onlineclr'])
    parser.add_argument('--vicreg_sim_coeff', default=25.0, type=float)
    parser.add_argument('--vicreg_std_coeff', default=25.0, type=float)
    parser.add_argument('--sogclr_gamma', default=0.8, type=float)
    parser.add_argument('--rho_I', default=8.0, type=float)
    parser.add_argument('--rho_T', default=8.0, type=float)
    parser.add_argument('--eta_init', default=0.001, type=float)
    parser.add_argument('--tau_init', default=0.01, type=float)
    parser.add_argument('--beta_u', default=0.9, type=float)
    parser.add_argument('--temp', default=0.01, type=float)
    parser.add_argument('--learnable_temp', action='store_true')
    parser.add_argument('--personalized_tau', action='store_true')
    parser.add_argument('--max_norm', default=1.0, type=float)
    parser.add_argument('--store_tau', action='store_true')
    parser.add_argument('--isogclr_temp_net', action='store_true')
    parser.add_argument('--alpha', default=1.0, type=float, help='for isogclr_denoise')

    # set the fraction of data used for training
    parser.add_argument('--train_frac', default=1.0, type=float)

    # check samples with high/low temperature values
    parser.add_argument('--check_samples_tau', action='store_true')

    # extract data from the cc3m dataset
    parser.add_argument('--extract_data', action='store_true')

    # zero-shot transfer
    parser.add_argument('--zs_dataset', default="", choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--zs_datafolder', default='./datasets', type=str)

    args = parser.parse_args()

    if args.check_samples_tau:
        args.evaluate = True

    args.train_file = os.path.join(args.ann_path, args.train_file)
    args.train_image_root = os.path.join(args.data_path, args.train_image_root)

    args.val_coco_file = os.path.join(args.ann_path, 'coco_val.json')
    # args.test_coco_file = os.path.join(args.ann_path, 'coco_test.json')
    args.coco_image_root = os.path.join(args.data_path, 'mscoco_val/mscoco_val2014_subset_5k')
    # args.val_flickr_file = os.path.join(args.data_path, 'clip_train/flickr30k_val.json')
    # args.test_flickr_file = os.path.join(args.data_path, 'clip_train/flickr30k_test.json')
    # args.flickr_image_root = os.path.join(args.data_path, 'flickr30k')

    # args.sbu_file = os.path.join(args.data_path, 'clip_train/sbu_train_new.json')
    # args.sbu_image_root = os.path.join(args.data_path, 'sbu')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    json.dump(args.__dict__, open(os.path.join(args.output_dir, 'args.json'), 'w'), indent=2) 
    
    main(args)
