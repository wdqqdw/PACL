import torch
import numpy as np
import torch.nn.modules as nn
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from torch.optim import Adam, AdamW, SGD, lr_scheduler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, top_k_accuracy_score

from model.param import ModelParam

def set_optim(args, model:nn.Module, model_cfg, epoch_num):
    scene_image_param = []
    senti_image_param = []
    general_text_param = []
    senti_text_param = []
    classify_param = []
    fusion_param = []
    other_model_param = []
    for name, param in model.named_parameters():
        if 'scene_image_model' in name:
            scene_image_param.append(name)
        elif 'senti_image_model' in name:
            senti_image_param.append(name)
        elif 'general_text_model' in name:
            general_text_param.append(name)
        elif 'senti_text_model' in name:
            senti_text_param.append(name)
        elif 'classify' in name:
            classify_param.append(name)
        elif 'fusion_layer' in name:
            fusion_param.append(name)
        else:
            other_model_param.append(name)


    if args.pretrain_lr == -1:
        lr_dict = {
            'lr_scene_image': model_cfg['commonParas']['lr_scene_image'],
            'lr_senti_image': model_cfg['commonParas']['lr_senti_image'],
            'lr_general_text': model_cfg['commonParas']['lr_general_text'],
            'lr_senti_text': model_cfg['commonParas']['lr_senti_text'],
            'lr_classify': model_cfg['commonParas']['lr_classify'],
            'lr_fusion': model_cfg['commonParas']['lr_fusion'],
            'lr_other': model_cfg['commonParas']['lr_other'],
        }
    else:
        lr_dict = {
            'lr_scene_image': args.pretrain_lr,
            'lr_senti_image': args.pretrain_lr,
            'lr_general_text': 0,
            'lr_senti_text': 0,
            'lr_classify': args.pretrain_lr,
            'lr_other': args.pretrain_lr,
        }

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in scene_image_param],
            "lr": lr_dict['lr_scene_image'],
        },
        {
            "params": [p for n, p in model.named_parameters() if n in senti_image_param],
            "lr": lr_dict['lr_senti_image'],
        },
        {
            "params": [p for n, p in model.named_parameters() if n in general_text_param],
            "lr": lr_dict['lr_general_text'],
        },
        {
            "params": [p for n, p in model.named_parameters() if n in senti_text_param],
            "lr": lr_dict['lr_senti_text'],
        },
        {
            "params": [p for n, p in model.named_parameters() if n in classify_param],
            "lr": lr_dict['lr_classify'],
        },
        {
            "params": [p for n, p in model.named_parameters() if n in fusion_param],
            "lr": lr_dict['lr_fusion'],
        },
        {
            "params": [p for n, p in model.named_parameters() if n in other_model_param],
            "lr": lr_dict['lr_other'],
        },
    ]


    optimizer_name = model_cfg['commonParas']['optimizer']
    assert optimizer_name == 'AdamW' or 'Adam' or 'SGD'
    if optimizer_name == 'AdamW':
        optimizer = AdamW(optimizer_grouped_parameters, weight_decay=args.pretrain_weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epoch_num, eta_min=1e-5)
    elif optimizer_name == 'Adam':
        optimizer = Adam(optimizer_grouped_parameters)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epoch_num, eta_min=1e-5)
    else:
        optimizer = SGD(optimizer_grouped_parameters, weight_decay=args.pretrain_weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epoch_num, eta_min=1e-5)
    return optimizer, scheduler

def read_from_batch(args, data):
    id, text, image, label, ocr, senti_cluster, fact_cluster, fs_match = data
    assert args.device == torch.device('cuda')
    if args.image_encoder != 'clip':
        image = image.to(args.device)
    label = label.to(args.device)
    return text, image, label, ocr, senti_cluster, fact_cluster, fs_match

def create_cluster_map(label):
    bs = len(label)
    mask = torch.zeros((bs, bs))
    for i in range(bs):
        for j in range(bs):
            if label[i] == label[j]:
                mask[i, j] = float('-inf')
    mask.fill_diagonal_(0)
    return mask

def create_cluster_map2(match_list, cluster_label1, cluster_label2):
    num_items = len(match_list)
    mask = torch.zeros((num_items, num_items))

    for i in range(num_items):
        for j in range(num_items):
            if cluster_label1[i] == cluster_label1[j] or cluster_label2[i] == cluster_label2[j]:
                mask[i, j] = float('-inf')
            if match_list[i] == j:
                mask[i, j] = 0

    return mask

def train_epoch(args, model:nn.Module, train_loader, loss_func, optimizer, scheduler):
    y_true = [[],[]]
    y_pred = [[],[],[]]
    total_loss = [0,0]
    num_labels = 0
    batch_index = 0

    param = ModelParam()
    loss_nce = loss_func[0]
    loss_mse = loss_func[1]

    model.train()
    model.zero_grad()
    train_loader_tqdm = tqdm(train_loader, desc='Train Iteration:')
    for index, data in enumerate(train_loader_tqdm):
        batch_index += 1

        text, image, label, ocr, senti_cluster, fact_cluster, fs_match = read_from_batch(args, data)

        if args.adaptive == '1':
            text_list = []
            image_list = []
            label_list = []
            senti_cluster_list = []
            fact_cluster_list = []
            bs = len(senti_cluster)
            single_bs = int(bs / 4)
            for i in range(4):
                text_list.append({k:text[k][i*single_bs:(i+1)*single_bs] for k in text.keys()})
                image_list.append(image[i*single_bs:(i+1)*single_bs])
                label_list.append(label[i*single_bs:(i+1)*single_bs])
                senti_cluster_list.append(senti_cluster[i*single_bs:(i+1)*single_bs])
                fact_cluster_list.append(fact_cluster[i*single_bs:(i+1)*single_bs])


            #Calculate fs:11 part: bring image close to corresponding text
            param.set_data_param(text_list[0], None, image_list[0], None, label_list[0], None, None)
            text_logits, image_logits, text_cov, image_cov, itm_output = model(param)
            itm_label = torch.cat([torch.ones(single_bs, dtype=torch.long), torch.zeros(2 * single_bs, dtype=torch.long)], dim=0).to(
                args.device)
            itm_label *= args.itm_target[0]
            loss_itm = loss_nce(itm_output, itm_label)
            cl_label = torch.LongTensor(range(single_bs)).to(args.device)
            #cl_label = torch.max(image_logits, 1).indices

            #loss_text_logits = loss_nce(text_logits, cl_label)
            loss_image_logits = loss_nce(image_logits, cl_label)
            loss_logits = loss_image_logits
            loss_1_logits = loss_logits
            loss_1_itm = loss_itm
            # if args.loss == 'logits':
            #     loss_1 = loss_logits
            # elif args.loss == 'itm':
            #     loss_1 = loss_itm
            # else:
            #     loss_1 = args.weight_clip*loss_logits + loss_itm

            #record A1's result
            _, predicted_label = torch.max(itm_output, 1)
            y_true[0].extend(itm_label.cpu().numpy())
            y_pred[0].extend(predicted_label.cpu().numpy())

            _, predicted_label = torch.max(image_logits[:, :bs], 1)
            y_true[1].extend(cl_label.cpu().numpy())
            y_pred[1].extend(predicted_label.cpu().numpy())
            top_k_predicted = image_logits[:, :bs].detach().cpu().numpy()

            for j in range(len(top_k_predicted)):
                y_pred[2].append(list(top_k_predicted[j]))


            total_loss[0] += loss_itm.item()
            total_loss[1] += loss_logits.item()
            num_labels += label_list[0].size(0)


            #Calculate nfs:01 part: bring text closer to same factual cluster image
            assert args.cluster in ['2', '3', '6', '10', '15', '25']
            cluster_label = [c[args.cluster] for c in fact_cluster_list[1]]
            param.set_data_param(text_list[1], None, image_list[1], None, label_list[1], None, None, True, False, cluster_label, None)
            text_logits, image_logits, text_cov, image_cov, itm_output = model(param)
            itm_label = torch.cat(
                [torch.ones(single_bs, dtype=torch.long), torch.zeros(2 * single_bs, dtype=torch.long)], dim=0).to(
                args.device)
            itm_label *= args.itm_target[1]
            loss_itm = loss_nce(itm_output, itm_label)
            cl_label = torch.LongTensor(range(single_bs)).to(args.device)

            match_sample = torch.max(text_logits, 1).indices
            cluster_label = [cluster_label[i] for i in match_sample]

            cluster_mask = create_cluster_map(cluster_label)
            cluster_mask = cluster_mask.to(args.device)
            text_logits += cluster_mask
            loss_text_logits = loss_nce(text_logits, cl_label)
            loss_logits = loss_text_logits * args.weight_a234[0]

            #test 6.1.0.5
            # loss_image_logits = loss_nce(image_logits, cl_label)
            # loss_logits = loss_image_logits
            loss_2_logits = loss_logits
            loss_2_itm = loss_itm

            # if args.loss == 'logits':
            #     loss_2 = loss_logits
            # elif args.loss == 'itm':
            #     loss_2 = loss_itm
            # else:
            #     loss_2 = args.weight_clip*loss_logits + loss_itm

            # Calculate nfs:10 part: bring image closer to same sentiment cluster text
            cluster_label = [c[args.cluster] for c in senti_cluster_list[2]]
            param.set_data_param(text_list[2], None, image_list[2], None, label_list[2], None, None, False, True, None, cluster_label)
            text_logits, image_logits, text_cov, image_cov, itm_output = model(param)
            itm_label = torch.cat(
                [torch.ones(single_bs, dtype=torch.long), torch.zeros(2 * single_bs, dtype=torch.long)], dim=0).to(
                args.device)
            itm_label *= args.itm_target[2]
            loss_itm = loss_nce(itm_output, itm_label)
            cl_label = torch.LongTensor(range(single_bs)).to(args.device)

            match_sample = torch.max(image_logits, 1).indices
            cluster_label = [cluster_label[i] for i in match_sample]

            cluster_mask = create_cluster_map(cluster_label)
            cluster_mask = cluster_mask.to(args.device)
            image_logits += cluster_mask
            loss_image_logits = loss_nce(image_logits, cl_label)
            loss_logits = loss_image_logits * args.weight_a234[1]
            # if args.loss == 'logits':
            #     loss_3 = loss_logits
            # elif args.loss == 'itm':
            #     loss_3 = loss_itm
            # else:
            #     loss_3 = args.weight_clip*loss_logits + loss_itm
            loss_3_logits = loss_logits
            loss_3_itm = loss_itm

            # Calculate nfs:00 part: bring image closer to the best match text
            cluster_label1 = [c[args.cluster] for c in senti_cluster_list[3]]
            cluster_label2 = [c[args.cluster] for c in fact_cluster_list[3]]
            param.set_data_param(text_list[3], None, image_list[3], None, label_list[3], None, None, True, True, cluster_label2, cluster_label1)
            text_logits, image_logits, text_cov, image_cov, itm_output = model(param)
            itm_label = torch.cat(
                [torch.ones(single_bs, dtype=torch.long), torch.zeros(2 * single_bs, dtype=torch.long)], dim=0).to(
                args.device)
            itm_label *= args.itm_target[3]
            loss_itm = loss_nce(itm_output, itm_label)
            cl_label = torch.max(image_logits, 1).indices

            i2t_match_sample = torch.max(image_logits, 1).indices
            cluster_label1 = [cluster_label1[i] for i in i2t_match_sample]
            t2i_match_sample = torch.max(text_logits, 1).indices
            cluster_label2 = [cluster_label2[i] for i in t2i_match_sample]

            cluster_mask = create_cluster_map2(match_sample, cluster_label1, cluster_label2)
            cluster_mask = cluster_mask.to(args.device)
            image_logits += cluster_mask

            loss_image_logits = loss_nce(image_logits, cl_label)
            loss_logits = loss_image_logits * args.weight_a234[2]

            loss_4_logits = loss_logits
            loss_4_itm = loss_itm
            # if args.loss == 'logits':
            #     loss_4 = loss_logits
            # elif args.loss == 'itm':
            #     loss_4 = loss_itm
            # else:
            #     loss_4 = args.weight_clip*loss_logits + loss_itm

            if args.loss == 'all_itm':
                loss = loss_1_itm + loss_2_itm + loss_3_itm + loss_4_itm
            elif args.loss == 'itm':
                if args.cur_epoch < args.warmup_epoch1:
                    loss = loss_1_itm
                elif args.cur_epoch >= args.warmup_epoch1 and args.cur_epoch < args.warmup_epoch2:
                    loss = loss_1_itm + loss_2_itm + loss_3_itm
                else:
                    loss = loss_1_itm + loss_2_itm + loss_3_itm + loss_4_itm
            elif args.loss == 'logits':
                if args.cur_epoch < args.warmup_epoch1:
                    loss = loss_1_logits
                elif args.cur_epoch >= args.warmup_epoch1 and args.cur_epoch < args.warmup_epoch2:
                    loss = loss_1_logits + loss_2_logits + loss_3_logits
                else:
                    loss = loss_1_logits + loss_2_logits + loss_3_logits + loss_4_logits
            elif args.loss == 'all_itm_clip':
                total_loss_itm = loss_1_itm + loss_2_itm + loss_3_itm + loss_4_itm
                if args.cur_epoch < args.warmup_epoch1:
                    total_loss_logits = loss_1_logits
                elif args.cur_epoch >= args.warmup_epoch1 and args.cur_epoch < args.warmup_epoch2:
                    total_loss_logits = loss_1_logits + loss_2_logits + loss_3_logits
                else:
                    total_loss_logits = loss_1_logits + loss_2_logits + loss_3_logits + loss_4_logits
                loss = args.weight_clip * total_loss_logits + total_loss_itm
            else:
                assert args.loss == 'all'
                if args.cur_epoch < args.warmup_epoch1:
                    total_loss_itm = loss_1_itm
                    total_loss_logits = loss_1_logits
                elif args.cur_epoch >= args.warmup_epoch1 and args.cur_epoch < args.warmup_epoch2:
                    total_loss_itm = loss_1_itm + loss_2_itm + loss_3_itm
                    total_loss_logits = loss_1_logits + loss_2_logits + loss_3_logits
                else:
                    total_loss_itm = loss_1_itm + loss_2_itm + loss_3_itm + loss_4_itm
                    total_loss_logits = loss_1_logits + loss_2_logits + loss_3_logits + loss_4_logits
                loss = args.weight_clip * total_loss_logits + total_loss_itm


            #loss = loss_1 + loss_2 + loss_3 + loss_4
            loss.backward()

        else:
            param.set_data_param(text, None, image, None, label, None, None)
            text_logits, image_logits, text_cov, image_cov, itm_output = model(param)

            bs = text_logits.size(0)
            itm_label = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(args.device)
            #loss_cov = loss_mse(image_cov, text_cov)

            #cl_label = torch.LongTensor(range(bs)).to(args.device)
            cl_label = torch.max(image_logits, 1).indices
            # Contrastive pretraining

            #loss_text_logits = loss_nce(text_logits, cl_label)
            loss_image_logits = loss_nce(image_logits, cl_label)
            loss_logits = loss_image_logits

            loss_itm = loss_nce(itm_output, itm_label)

            loss = (loss_logits + loss_itm) * 1/2
            loss.backward()

            _, predicted_label = torch.max(itm_output, 1)
            y_true[0].extend(itm_label.cpu().numpy())
            y_pred[0].extend(predicted_label.cpu().numpy())

            _, predicted_label = torch.max(image_logits[:,:bs], 1)
            y_true[1].extend(cl_label.cpu().numpy())
            y_pred[1].extend(predicted_label.cpu().numpy())

            top_k_predicted = image_logits[:,:bs].detach().cpu().numpy()

            for j in range(len(top_k_predicted)):
                y_pred[2].append(list(top_k_predicted[j]))

            total_loss[0] += loss_itm.item()
            total_loss[1] += loss_logits.item()
            num_labels += label.size(0)

        if batch_index % args.acc_grad == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    y_true = [np.array(y_true[0]), np.array(y_true[1])]
    y_pred = [np.array(y_pred[0]), np.array(y_pred[1]), y_pred[2]]

    return y_true, y_pred,  total_loss,  num_labels


def dev_epoch(args, model:nn.Module, dev_loader, loss_func, mode):
    y_true = [[],[]]
    y_pred = [[],[],[]]
    total_loss = [0,0]
    num_labels = 0
    batch_index = 0

    param = ModelParam()
    loss_nce = loss_func[0]
    loss_mse = loss_func[1]

    with torch.no_grad():
        model.eval()
        dev_loader_tqdm = tqdm(dev_loader, desc=mode + ' Iteration:')
        for index, data in enumerate(dev_loader_tqdm):
            batch_index += 1
            text, image, label, ocr, senti_cluster, fact_cluster, fs_match = read_from_batch(args, data)
            param.set_data_param(text, None, image, None, label, ocr, None)
            text_logits, image_logits, text_cov, image_cov, itm_output = model(param)

            bs = text_logits.size(0)
            itm_label = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(
                args.device)

            cl_label = torch.LongTensor(range(bs)).to(args.device)

            # # Contrastive pretraining
            #loss_text_logits = loss_nce(text_logits, cl_label)
            loss_image_logits = loss_nce(image_logits, cl_label)
            loss_logits = loss_image_logits

            loss_itm = loss_nce(itm_output, itm_label)

            loss = (loss_logits + loss_itm) * 1 / 2

            _, predicted_label = torch.max(itm_output, 1)
            y_true[0].extend(itm_label.cpu().numpy())
            y_pred[0].extend(predicted_label.cpu().numpy())

            _, predicted_label = torch.max(image_logits[:,:bs], 1)
            y_true[1].extend(cl_label.cpu().numpy())
            y_pred[1].extend(predicted_label.cpu().numpy())

            top_k_predicted = image_logits[:,:bs].detach().cpu().numpy()

            for j in range(len(top_k_predicted)):
                y_pred[2].append(list(top_k_predicted[j]))

            total_loss[0] += loss_itm.item()
            total_loss[1] += loss_logits.item()
            #total_loss[1] += loss_logits.item()
            num_labels += label.size(0)


    y_true = [np.array(y_true[0]), np.array(y_true[1])]
    y_pred = [np.array(y_pred[0]), np.array(y_pred[1]), y_pred[2]]

    return y_true, y_pred,  total_loss,  num_labels


def evaluate_epoch1(y_true, y_pred, loss, num_samples):
    train_accuracy = []
    train_F1_weighted = []
    train_F1_micro = []
    batch_size = len(y_pred[2][0])
    for i in range(len(loss)):
        loss[i] /= num_samples/batch_size
        loss[i] = float(loss[i])

    #assert len(y_true) == len(y_pred)
    for i in range(len(y_true)):
        train_accuracy.append(accuracy_score(y_true[i], y_pred[i]))
        train_F1_weighted.append(f1_score(y_true[i], y_pred[i], average='weighted'))
        train_F1_micro.append(f1_score(y_true[i], y_pred[i], average='micro'))

    max_aligned_length = num_samples - num_samples % batch_size
    y_true_truc = y_true[1][:max_aligned_length]
    y_pred_truc = np.array(y_pred[2][:max_aligned_length])

    y_pred_truc = np.nan_to_num(y_pred_truc)
    train_accuracy.append(top_k_accuracy_score(y_true_truc, y_pred_truc, k=3))
    train_accuracy.append(top_k_accuracy_score(y_true_truc, y_pred_truc, k=5))
    return loss, train_accuracy, train_F1_weighted, train_F1_micro

# mode = Train/Dev/Test/Hard
def proceed_epoch(args, model:nn.Module, dataloader, loss_func, optimizer, scheduler, mode):
    if mode == 'Train':
        #y_true, y_pred, total_loss, num_labels = dt.debug_epoch(args, model, dataloader, loss_func, optimizer)
        y_true, y_pred, loss, num_labels = train_epoch(args, model, dataloader, loss_func, optimizer, scheduler)
        #check_model_param(args, model)
    else:
        assert mode in ['Dev', 'Test', 'Hard']
        y_true, y_pred, loss, num_labels = dev_epoch(args, model, dataloader, loss_func, mode)

    loss, accuracy, F1_weighted, F1_micro = evaluate_epoch1(y_true, y_pred, loss, num_labels)

    aligned_mode = {'Train': 'Train', 'Dev': 'Dev  ', 'Test': 'Test ', 'Hard': 'Hard '}
    save_content = aligned_mode[mode] + ': ITM Accuracy: %.4f,                        F1(weighted): %.4f, F1(macro): %.4f, loss: %.4f \n' \
                                       '    SAMPLE Accuracy: %.4f(1) %.4f(3) %.4f(5), F1(weighted): %.4f, F1(macro): %.4f, loss: %.4f' % \
                   (accuracy[0], F1_weighted[0] , F1_micro[0], loss[0], accuracy[1], accuracy[2], accuracy[3], F1_weighted[1] , F1_micro[1], loss[1])
    output_to_log(args, save_content)
    return [accuracy, F1_weighted , F1_micro, loss]

def check_model_param(args, model):
    for name, param in model.named_parameters():
        if 'classify' in name:
            print("breakpoint")
            #classify_param.append(name)
        elif 'fusion_layer' in name:
            print("breakpoint")
            #fusion_param.append(name)


def output_to_log(args, content):
    print(content)
    with open(args.log_name, 'a+', encoding='utf-8') as f:
        f.write(content + '\n')

def output_to_error_log(args, content):
    print(content)
    with open(args.error_log, 'a+', encoding='utf-8') as f:
        f.write(content + '\n')

def train_model(args, epoch_num, model, dataloader_list, loss_func, optimizer, scheduler):
    [train_loader, dev_loader, test_loader] = dataloader_list

    cls_eval_metric_name = ("dev_acc", "dev_F1", "test_acc", "test_F1", "loss_sum")
    clip_eval_metric_name = ("dev_acc1", "dev_acc3", "dev_acc5", "dev_F1", "test_acc1", "test_acc3", "test_acc5", "test_F1", "loss_sum")
    dataloader_name = ("Train", "Dev", "Test")
    metric_curve_name = {'acc':('itm_acc', 'sample_acc1', 'sample_acc3', 'sample_acc5'), 'F1_weighted':('itm_F1_weighted', 'sample_F1_weighted'),
                          'F1_micro':('itm_F1_micro', 'sample_F1_micro'), 'loss':('itm_loss', 'sample_loss')}

    #num of loss has to equal to num of acc
    best_F1_acc = [[0, 0, 0, 0, -10], [0, 0, 0, 0, 0, 0, 0, 0, -10]]
    best_epoch = [[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1, -1]]


    loss_list = [[[] for _ in range(len(metric_curve_name['loss']))] for _ in range(len(dataloader_list))]
    acc_list = [[[] for _ in range(len(metric_curve_name['acc']))] for _ in range(len(dataloader_list))]
    F1_weighted_list = [[[] for _ in range(len(metric_curve_name['F1_weighted']))] for _ in range(len(dataloader_list))]
    F1_micro_list = [[[] for _ in range(len(metric_curve_name['F1_micro']))] for _ in range(len(dataloader_list))]

    for epoch in range(epoch_num):
        args.cur_epoch = epoch
        epoch_id = 'Pretrain epoch: %d/%d:' %(args.cur_epoch + 1, epoch_num)
        output_to_log(args, epoch_id)

        #metric = [acc, F1_weighted, F1_micro, total_loss]
        train_metric = proceed_epoch(args, model, train_loader, loss_func, optimizer, scheduler, 'Train')
        dev_metric = proceed_epoch(args, model, dev_loader, loss_func, optimizer, scheduler, 'Dev')
        test_metric = proceed_epoch(args, model, test_loader, loss_func, optimizer, scheduler, 'Test')
        metric_list = [train_metric, dev_metric, test_metric]

        for i in range(len(metric_list)):
            for j in range(len(acc_list[i])):
                acc_list[i][j].append(metric_list[i][0][j])

            for j in range(len(F1_weighted_list[i])):
                F1_weighted_list[i][j].append(metric_list[i][1][j])

            for j in range(len(F1_micro_list[i])):
                F1_micro_list[i][j].append(metric_list[i][2][j])

            for j in range(len(loss_list[i])):
                loss_list[i][j].append(metric_list[i][3][j])

        best_flag = False
        subtask_id = ('text', 'image')
        epoch_dscrpt = 'New records: \n'
        for k in range(len(subtask_id)):
            loss_sum = test_metric[3][k] + dev_metric[3][k]
            if k == 0:
                cur_F1_acc = [dev_metric[0][0], dev_metric[1][k], test_metric[0][0], test_metric[1][k], -loss_sum]
            else:
                cur_F1_acc = [dev_metric[0][1], dev_metric[0][2], dev_metric[0][3], dev_metric[1][k],
                              test_metric[0][1], test_metric[0][2], test_metric[0][3], test_metric[1][k], -loss_sum]
            for i in range(len(cur_F1_acc)):
                if cur_F1_acc[i] > best_F1_acc[k][i]:
                    best_flag = True
                    acc_improve = cur_F1_acc[i] - best_F1_acc[k][i]
                    best_F1_acc[k][i] = cur_F1_acc[i]
                    best_epoch[k][i] = epoch
                    if k == 0:
                        epoch_dscrpt += subtask_id[k] + ' ' + cls_eval_metric_name[i] + '(+%.4f) ' % (acc_improve)
                    else:
                        epoch_dscrpt += subtask_id[k] + ' ' + clip_eval_metric_name[i] + '(+%.4f) ' % (acc_improve)
            epoch_dscrpt += '\n'

        with open(args.log_name, 'a+', encoding='utf-8') as f:
            if best_flag == False:
                f.write("Model is not best up to current epoch.\n\n")
            else:
                f.write(epoch_dscrpt + '\n')

    if args.pretrain == '1':
        torch.save(model.state_dict(), args.full_model_path)

    draw_metric_curve(epoch_num, args.metric_dir, acc_list, F1_weighted_list, F1_micro_list, loss_list, dataloader_name, metric_curve_name)
    record_best_epochs(args, best_F1_acc, best_epoch, cls_eval_metric_name, acc_list, F1_weighted_list)


def draw_metric_curve(epoch_num, save_path, acc_list, F1_weighted_list, F1_micro_list, loss_list, dataloader_name, metric_curve_name):
    x = range(epoch_num)

    acc_curve_name = ('itm_acc', 'sample_acc')
    plt.figure()
    for k in range(len(acc_curve_name)):
        plt.clf()
        if k == 0:
            y_up_bound = max([max([acc_list[i][k][j] for j in range(len(acc_list[i][k]))]) for i in range(len(acc_list))]) * 1.05
            y_low_bound = min([min([acc_list[i][k][j] for j in range(len(acc_list[i][k]))]) for i in range(len(acc_list))]) * 0.95
        else:
            y_up_bound = max([max([max([acc_list[i][l][j] for j in range(len(acc_list[i][l]))]) for i in range(len(acc_list))]) for l in range(1,4)]) * 1.05
            y_low_bound = min([min([min([acc_list[i][l][j] for j in range(len(acc_list[i][l]))]) for i in range(len(acc_list))]) for l in range(1,4)]) * 0.95
        for i in range(len(acc_list)):
            assert len(acc_list) <= 4
            plt.subplot(2, 2, i + 1)
            if k == 0:
                plt.plot(x, acc_list[i][k], '.-')
            else:
                plt.plot(x, acc_list[i][1], '.-')
                plt.plot(x, acc_list[i][2], '.-')
                plt.plot(x, acc_list[i][3], '.-')
            plt.ylim(y_low_bound, y_up_bound)
            plt.title(dataloader_name[i])
            #plt.ylabel('Accuracy')
        plt.tight_layout()
        plt.savefig(save_path + "/pretrain_" + acc_curve_name[k] + ".jpg")

    for k in range(len(metric_curve_name['F1_weighted'])):
        plt.clf()
        y_up_bound = max([max([F1_weighted_list[i][k][j] for j in range(len(F1_weighted_list[i][k]))]) for i in range(len(F1_weighted_list))]) * 1.05
        y_low_bound = min([min([F1_weighted_list[i][k][j] for j in range(len(F1_weighted_list[i][k]))]) for i in range(len(F1_weighted_list))]) * 0.95
        for i in range(len(F1_weighted_list)):
            assert len(F1_weighted_list) <= 4
            plt.subplot(2, 2, i + 1)
            plt.plot(x, F1_weighted_list[i][k], '.-')
            plt.ylim(y_low_bound, y_up_bound)
            plt.title(dataloader_name[i])
            #plt.ylabel('F1_weighted')
        plt.tight_layout()
        plt.savefig(save_path + "/pretrain_" + metric_curve_name['F1_weighted'][k] + ".jpg")

    for k in range(len(metric_curve_name['F1_micro'])):
        plt.clf()
        y_up_bound = max([max([F1_micro_list[i][k][j] for j in range(len(F1_micro_list[i][k]))]) for i in range(len(F1_micro_list))]) * 1.05
        y_low_bound = min([min([F1_micro_list[i][k][j] for j in range(len(F1_micro_list[i][k]))]) for i in range(len(F1_micro_list))]) * 0.95
        for i in range(len(F1_micro_list)):
            assert len(F1_micro_list) <= 4
            plt.subplot(2, 2, i + 1)
            plt.plot(x, F1_micro_list[i][k], '.-')
            plt.ylim(y_low_bound, y_up_bound)
            plt.title(dataloader_name[i])
            #plt.ylabel('F1_micro')
        plt.tight_layout()
        plt.savefig(save_path + "/pretrain_" + metric_curve_name['F1_micro'][k] + ".jpg")

    for k in range(len(metric_curve_name['loss'])):
        plt.clf()
        y_up_bound = max([max([loss_list[i][k][j] for j in range(len(loss_list[i][k]))]) for i in range(len(loss_list))]) * 1.05
        y_low_bound = min([min([loss_list[i][k][j] for j in range(len(loss_list[i][k]))]) for i in range(len(loss_list))]) * 0.95
        for i in range(len(loss_list)):
            assert len(loss_list) <= 4
            plt.subplot(2, 2, i + 1)
            plt.plot(x, loss_list[i][k], '.-')
            plt.ylim(y_low_bound, y_up_bound)
            plt.title(dataloader_name[i])
            #plt.ylabel('F1_micro')
        plt.tight_layout()
        plt.savefig(save_path + "/pretrain_" + metric_curve_name['loss'][k] + ".jpg")



def record_best_epochs(args, best_F1_acc, best_epoch, eval_metric_name, acc_list, F1_weighted_list):
    subtask_id = ('ITM', 'Sample')
    best_F1_acc[1] = [best_F1_acc[1][0], best_F1_acc[1][3], best_F1_acc[1][4], best_F1_acc[1][7], best_F1_acc[1][8]]
    best_epoch[1] = [best_epoch[1][0], best_epoch[1][3], best_epoch[1][4], best_epoch[1][7], best_epoch[1][8]]
    with open(args.log_name, 'a+', encoding='utf-8') as f:
        for k in range(len(subtask_id)):
            save_content = [subtask_id[k] + ":\n"]
            for i in range(len(best_F1_acc[k])):
                acc_list_index = int(i // 2) + 1
                epoch = best_epoch[k][i]
                if i % 2 == 1:
                    if epoch == best_epoch[k][i - 1]:
                        save_content.pop()
                        save_content.append("Best %s and %s at epoch %d: Acc (%.4f), F1 (%.4f)\n" \
                                            % (eval_metric_name[i - 1], eval_metric_name[i], epoch + 1,
                                               acc_list[acc_list_index][k][epoch], F1_weighted_list[acc_list_index][k][epoch]))
                        continue
                if i != 4:
                    save_content.append("Best %s at epoch %d: Acc (%.4f), F1 (%.4f)\n" \
                                    % (eval_metric_name[i], epoch + 1, acc_list[acc_list_index][k][epoch],
                                       F1_weighted_list[acc_list_index][k][epoch]))
                else:
                    save_content.append("Best %s at epoch %d: Sum (%.4f)\n" %(eval_metric_name[i], epoch + 1, -best_F1_acc[k][i]))
                # save_content += "Best %s: %.4f epoch: %d\n" %(eval_metric_name[i], best_F1_acc[i], best_epoch[i] + 1)
            f.write("".join(save_content))
            f.write("\n")
        f.write("\n")
