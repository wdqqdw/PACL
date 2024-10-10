import torch
import json
import numpy as np
import torch.nn.modules as nn
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from torch.optim import Adam, AdamW, lr_scheduler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, mean_squared_error, r2_score

from model.param import ModelParam
from PACL.config_PACL import Config

def set_optim(args, model:nn.Module, model_cfg, epoch_num):
    scene_image_param = []
    senti_image_param = []
    general_text_param = []
    senti_text_param = []
    classify_param = []
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
        else:
            other_model_param.append(name)

    lr_dict = {
        'lr_scene_image_ft': model_cfg['commonParas']['lr_scene_image_ft'],
        'lr_senti_image_ft': model_cfg['commonParas']['lr_senti_image_ft'],
        'lr_general_text_ft': model_cfg['commonParas']['lr_general_text_ft'],
        'lr_senti_text_ft': model_cfg['commonParas']['lr_senti_text_ft'],
        'lr_classify_ft': model_cfg['commonParas']['lr_classify_ft'],
        'lr_other_ft': model_cfg['commonParas']['lr_other_ft'],
    }


    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in scene_image_param],
            "lr": lr_dict['lr_scene_image_ft'],
        },
        {
            "params": [p for n, p in model.named_parameters() if n in senti_image_param],
            "lr": lr_dict['lr_senti_image_ft'],
        },
        {
            "params": [p for n, p in model.named_parameters() if n in general_text_param],
            "lr": lr_dict['lr_general_text_ft'],
        },
        {
            "params": [p for n, p in model.named_parameters() if n in senti_text_param],
            "lr": lr_dict['lr_senti_text_ft'],
        },
        {
            "params": [p for n, p in model.named_parameters() if n in classify_param],
            "lr": lr_dict['lr_classify_ft'],
        },
        {
            "params": [p for n, p in model.named_parameters() if n in other_model_param],
            "lr": lr_dict['lr_other_ft'],
        },
    ]


    optimizer_name = model_cfg['commonParas']['optimizer']
    assert optimizer_name == 'AdamW' or 'Adam'
    if optimizer_name == 'AdamW':
        optimizer = AdamW(optimizer_grouped_parameters)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epoch_num)
    elif optimizer_name == 'Adam':
        optimizer = Adam(optimizer_grouped_parameters)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epoch_num)
    else:
        optimizer = None
        scheduler = None
    return optimizer, scheduler


def read_from_batch(args, data):
    assert args.device == torch.device('cuda')
    if args.dataset_name in ['Emotic', 'Emotic-reg']:
        image, cat_label, vad_label = data
        image = image.to(args.device)
        cat_label = cat_label.to(args.device)
        vad_label = vad_label.to(args.device)
        return image, cat_label, vad_label

    elif args.dataset_name == 'WEBEmo':
        image, label1, label2, label3 = data
        image = image.to(args.device)
        label1 = label1.to(args.device)
        label2 = label2.to(args.device)
        label3 = label3.to(args.device)
        return image, label1, label2, label3

    elif args.dataset_name in ['FI', 'Emotion6', 'UnbiasedEmo']:
        id, image, label = data
        image = image.to(args.device)
        label = label.to(args.device)
        return id, image, label

    elif args.dataset_name in ['IMDB']:
        id, text, label = data
        label = label.to(args.device)
        return text, label

    else:
        id, text, image, label, ocr, _, _, _ = data
        if args.image_encoder != 'clip':
            image = image.to(args.device)
        label = label.to(args.device)
        return text, image, label, ocr


def train_epoch(args, model:nn.Module, dataloader, loss_func, optimizer, scheduler):
    if args.dataset_name in ['Emotic', 'Emotic-reg']:
        return train_epoch_emotic(args, model, dataloader, loss_func, optimizer, scheduler)
    elif args.dataset_name == 'WEBEmo':
        return train_epoch_webemo(args, model, dataloader, loss_func, optimizer, scheduler)
    elif args.dataset_name in ['FI', 'Emotion6', 'UnbiasedEmo']:
        return train_epoch_fi(args, model, dataloader, loss_func, optimizer, scheduler)
    elif args.dataset_name == 'IMDB':
        return train_epoch_text(args, model, dataloader, loss_func, optimizer, scheduler)
    else:
        return train_epoch_imagetext(args, model, dataloader, loss_func, optimizer, scheduler)

def train_epoch_emotic(args, model:nn.Module, dataloader, loss_func, optimizer, scheduler):
    y_true = [[], [], []]
    y_pred = [[], [], []]
    y_pred_value = [[],[],[]]
    total_loss = 0
    num_labels = 0
    batch_index = 0
    loss_func = nn.MSELoss()

    param = ModelParam()

    model.train()
    model.zero_grad()
    train_loader_tqdm = tqdm(dataloader, desc='Train Iteration:')
    for index, data in enumerate(train_loader_tqdm):
        batch_index += 1

        image, cat_label, vad_label = read_from_batch(args, data)
        label = vad_label
        param.set_data_param(None, None, image, None, label, None, None)
        output = model(param)

        loss = loss_func(output, label.float())

        loss.backward()
        for i in range(0,3):
            y_pred_value[i].extend(output[:,i].cpu().detach().numpy())
            y_true[i].extend(label[:,i].cpu().numpy())

        #output = torch.sigmoid(output)
        #y_pred_value.extend(output.cpu().detach().numpy())
        #predicted_label = output.ge(0.5).int()
        #y_true.extend(label.cpu().numpy())
        #y_pred.extend(predicted_label.cpu().numpy())
        total_loss += loss.item()
        num_labels += label.size(0)

        if batch_index % args.acc_grad == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    return np.array(y_true), [np.array(y_pred), np.array(y_pred_value)], total_loss,  num_labels

def train_epoch_webemo(args, model, dataloader, loss_func, optimizer, scheduler):
    y_true = [[], [], []]
    y_pred = [[], [], []]
    total_loss = 0
    num_labels = 0
    batch_index = 0

    param = ModelParam()

    model.train()
    model.zero_grad()
    train_loader_tqdm = tqdm(dataloader, desc='Train Iteration:')
    for index, data in enumerate(train_loader_tqdm):
        batch_index += 1

        image, label1, label2, label3 = read_from_batch(args, data)
        param.set_data_param(None, None, image, None, label1, None, None)
        output1, output2, output3 = model(param, dataset_name='WEBEmo')

        loss = loss_func(output1, label1) + loss_func(output2, label2) + loss_func(output3, label3)
        loss.backward()

        _, predicted_label = torch.max(output1, 1)
        y_pred[0].extend(predicted_label.cpu())
        y_true[0].extend(label1.cpu())

        _, predicted_label = torch.max(output2, 1)
        y_pred[1].extend(predicted_label.cpu())
        y_true[1].extend(label2.cpu())

        _, predicted_label = torch.max(output3, 1)
        y_pred[2].extend(predicted_label.cpu())
        y_true[2].extend(label3.cpu())

        total_loss += loss.item()
        num_labels += label1.size(0)

        if batch_index % args.acc_grad == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    y_true = [np.array(y_true[0]), np.array(y_true[1]), np.array(y_true[2])]
    y_pred = [np.array(y_pred[0]), np.array(y_pred[1]), np.array(y_pred[2])]

    return y_true, y_pred, total_loss, num_labels

def train_epoch_fi(args, model:nn.Module, train_loader, loss_func, optimizer, scheduler):
    y_true = []
    y_pred = []
    total_loss = 0
    num_labels = 0
    batch_index = 0

    param = ModelParam()

    model.train()
    model.zero_grad()
    train_loader_tqdm = tqdm(train_loader, desc='Train Iteration:')
    for index, data in enumerate(train_loader_tqdm):
        batch_index += 1

        id, image, label = read_from_batch(args, data)
        param.set_data_param(None, None, image, None, label, None, None)
        output, image_embed = model(param)

        loss = loss_func(output, label)

        loss.backward()

        _, predicted_label = torch.max(output, 1)
        y_true.extend(label.cpu())
        y_pred.extend(predicted_label.cpu())
        total_loss += loss.item()
        num_labels += label.size(0)

        if batch_index % args.acc_grad == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    return np.array(y_true), np.array(y_pred), total_loss,  num_labels


def train_epoch_imagetext(args, model:nn.Module, train_loader, loss_func, optimizer, scheduler):
    y_true = []
    y_pred = []
    total_loss = 0
    num_labels = 0
    batch_index = 0

    param = ModelParam()

    model.train()
    model.zero_grad()
    train_loader_tqdm = tqdm(train_loader, desc='Train Iteration:')
    for index, data in enumerate(train_loader_tqdm):
        batch_index += 1

        text, image, label, ocr = read_from_batch(args, data)
        param.set_data_param(text, None, image, None, label, ocr, None)
        output = model(param)

        loss = loss_func(output, label)

        loss.backward()

        _, predicted_label = torch.max(output, 1)
        y_true.extend(label.cpu())
        y_pred.extend(predicted_label.cpu())
        total_loss += loss.item()
        num_labels += label.size(0)

        if batch_index % args.acc_grad == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    return np.array(y_true), np.array(y_pred), total_loss,  num_labels

def train_epoch_text(args, model:nn.Module, train_loader, loss_func, optimizer, scheduler):
    y_true = []
    y_pred = []
    total_loss = 0
    num_labels = 0
    batch_index = 0

    param = ModelParam()

    model.train()
    model.zero_grad()
    train_loader_tqdm = tqdm(train_loader, desc='Train Iteration:')
    for index, data in enumerate(train_loader_tqdm):
        batch_index += 1

        text, label = read_from_batch(args, data)
        param.set_data_param(text, None, None, None, label, None, None)
        output = model(param)

        loss = loss_func(output, label)

        loss.backward()

        _, predicted_label = torch.max(output, 1)
        y_true.extend(label.cpu())
        y_pred.extend(predicted_label.cpu())
        total_loss += loss.item()
        num_labels += label.size(0)

        if batch_index % args.acc_grad == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    return np.array(y_true), np.array(y_pred), total_loss,  num_labels

def dev_epoch(args, model:nn.Module, dataloader, loss_func, mode):
    if args.dataset_name in ['Emotic', 'Emotic-reg']:
        return dev_epoch_image(args, model, dataloader, loss_func, mode)
    elif args.dataset_name == 'WEBEmo':
        return dev_epoch_webemo(args, model, dataloader, loss_func, mode)
    elif args.dataset_name in ['FI', 'Emotion6', 'UnbiasedEmo']:
        return dev_epoch_fi(args, model, dataloader, loss_func, mode)
    elif args.dataset_name == 'IMDB':
        return dev_epoch_text(args, model, dataloader, loss_func, mode)
    else:
        return dev_epoch_imagetext(args, model, dataloader, loss_func, mode)

def dev_epoch_image(args, model:nn.Module, dev_loader, loss_func, mode):
    y_true = [[], [], []]
    y_pred = [[], [], []]
    y_pred_value = [[], [], []]
    total_loss = 0
    num_labels = 0
    batch_index = 0
    loss_func = nn.MSELoss()

    param = ModelParam()

    with torch.no_grad():
        model.eval()
        dev_loader_tqdm = tqdm(dev_loader, desc=mode + ' Iteration:')
        for index, data in enumerate(dev_loader_tqdm):
            batch_index += 1

            image, cat_label, vad_label = read_from_batch(args, data)
            label = vad_label
            param.set_data_param(None, None, image, None, label, None, None)
            output = model(param)

            loss = loss_func(output, label.float())

            for i in range(0, 3):
                y_pred_value[i].extend(output[:, i].cpu().detach().numpy())
                y_true[i].extend(label[:, i].cpu().numpy())

            # output = torch.sigmoid(output)
            # y_pred_value.extend(output.cpu().detach().numpy())
            # predicted_label = output.ge(0.5).int()
            # y_true.extend(label.cpu().numpy())
            # y_pred.extend(predicted_label.cpu().numpy())
            total_loss += loss.item()
            num_labels += label.size(0)

    return np.array(y_true), [np.array(y_pred), np.array(y_pred_value)], total_loss,  num_labels

def dev_epoch_webemo(args, model, dev_loader, loss_func, mode):
    y_true = [[], [], []]
    y_pred = [[], [], []]
    total_loss = 0
    num_labels = 0
    batch_index = 0

    param = ModelParam()

    with torch.no_grad():
        model.eval()
        dev_loader_tqdm = tqdm(dev_loader, desc=mode + ' Iteration:')
        for index, data in enumerate(dev_loader_tqdm):
            batch_index += 1

            image, label1, label2, label3 = read_from_batch(args, data)
            param.set_data_param(None, None, image, None, label1, None, None)
            output1, output2, output3 = model(param, dataset_name='WEBEmo')

            loss = loss_func(output1, label1) + loss_func(output2, label2) + loss_func(output3, label3)

            _, predicted_label = torch.max(output1, 1)
            y_pred[0].extend(predicted_label.cpu())
            y_true[0].extend(label1.cpu())

            _, predicted_label = torch.max(output2, 1)
            y_pred[1].extend(predicted_label.cpu())
            y_true[1].extend(label2.cpu())

            _, predicted_label = torch.max(output3, 1)
            y_pred[2].extend(predicted_label.cpu())
            y_true[2].extend(label3.cpu())

            total_loss += loss.item()
            num_labels += label1.size(0)

    y_true = [np.array(y_true[0]), np.array(y_true[1]), np.array(y_true[2])]
    y_pred = [np.array(y_pred[0]), np.array(y_pred[1]), np.array(y_pred[2])]

    return y_true, y_pred, total_loss, num_labels

def dev_epoch_fi(args, model:nn.Module, dev_loader, loss_func, mode):
    y_true = []
    y_pred = []
    total_loss = 0
    num_labels = 0
    batch_index = 0
    correct = {}
    param = ModelParam()
    embed_list = []
    label_list = []


    with torch.no_grad():
        model.eval()
        dev_loader_tqdm = tqdm(dev_loader, desc=mode + ' Iteration:')
        for index, data in enumerate(dev_loader_tqdm):
            batch_index += 1
            id, image, label = read_from_batch(args, data)
            param.set_data_param(None, None, image, None, label, None, None)
            output, image_embed = model(param)
            embed_list.append(output.clone().cpu())
            label_list.append(label.cpu())

            loss = loss_func(output, label)

            _, predicted_label = torch.max(output, 1)

            for i in range(len(predicted_label)):
                correct[id[i]] = {'predicted': int(predicted_label[i]), 'real': int(label[i])}


            y_true.extend(label.cpu())
            y_pred.extend(predicted_label.cpu())
            total_loss += loss.item()
            num_labels += label.size(0)

    path = args.save_dir + '/' + str(args.cur_epoch) + '.json'
    json_data = json.dumps(correct, indent=2)
    with open(path, 'w') as f:
        f.write(json_data)

    embed_list = torch.cat(embed_list, dim=0)
    label_list = torch.cat(label_list, dim=0)
    embed_list = embed_list.numpy().astype(float).tolist()
    label_list = label_list.numpy().tolist()

    if args.cur_epoch >= args.epoch/2:
        info = {'embed_list': embed_list, 'label_list': label_list}
        json_data = json.dumps(info, indent=2)
        with open(args.save_dir + '/' + str(args.cur_epoch) + '_' + mode + '_tsne.json', 'w') as f:
            f.write(json_data)

    return np.array(y_true), np.array(y_pred), total_loss, num_labels

def dev_epoch_imagetext(args, model:nn.Module, dev_loader, loss_func, mode):
    y_true = []
    y_pred = []
    total_loss = 0
    num_labels = 0
    batch_index = 0

    param = ModelParam()

    with torch.no_grad():
        model.eval()
        dev_loader_tqdm = tqdm(dev_loader, desc=mode + ' Iteration:')
        for index, data in enumerate(dev_loader_tqdm):
            batch_index += 1
            text, image, label, ocr = read_from_batch(args, data)
            param.set_data_param(text, None, image, None, label, ocr, None)
            output = model(param)

            loss = loss_func(output, label)

            _, predicted_label = torch.max(output, 1)
            y_true.extend(label.cpu())
            y_pred.extend(predicted_label.cpu())
            total_loss += loss.item()
            num_labels += label.size(0)

    return np.array(y_true), np.array(y_pred), total_loss, num_labels

def dev_epoch_text(args, model:nn.Module, dev_loader, loss_func, mode):
    y_true = []
    y_pred = []
    total_loss = 0
    num_labels = 0
    batch_index = 0

    param = ModelParam()

    with torch.no_grad():
        model.eval()
        dev_loader_tqdm = tqdm(dev_loader, desc=mode + ' Iteration:')
        for index, data in enumerate(dev_loader_tqdm):
            batch_index += 1
            text, label = read_from_batch(args, data)
            param.set_data_param(text, None, None, None, label, None, None)
            output = model(param)

            loss = loss_func(output, label)

            _, predicted_label = torch.max(output, 1)
            y_true.extend(label.cpu())
            y_pred.extend(predicted_label.cpu())
            total_loss += loss.item()
            num_labels += label.size(0)

    return np.array(y_true), np.array(y_pred), total_loss, num_labels


def evaluate_epoch(args, y_true, y_pred, loss, num_samples):
    #if args.dataset_name == 'Emotic':
        # y_pred_value = y_pred[1]
        # y_pred = y_pred[0]
        #
        # loss /= num_samples
        # train_accuracy = []
        # train_F1_weighted = []
        # train_F1_macro = []
        # auc = -1
        # auc = roc_auc_score(y_true, y_pred_value, average='weighted', multi_class='ovo')
        # assert len(y_true[:,0]) == num_samples
        # for i in range(len(y_true[0, :])):
        #     y_true_label = y_true[:, i]
        #     y_pred_label = y_pred[:, i]
        #     train_accuracy.append(precision_score(y_true_label, y_pred_label, zero_division=0))
        #     train_F1_weighted.append(f1_score(y_true_label, y_pred_label, average='weighted', zero_division=0))
        #     train_F1_macro.append(f1_score(y_true_label, y_pred_label, average='macro', zero_division=0))
        #
        # return float(loss), train_accuracy, train_F1_weighted, auc

    if args.dataset_name in ['Emotic', 'Emotic-reg']:
        mse = 0
        r2 = 0
        loss /= num_samples
        y_pred = y_pred[1]
        for i in range(3):
            mse += mean_squared_error(y_true[i], y_pred[i])/3
            r2 += r2_score(y_true[i], y_pred[i])/3

        return float(loss), mse, r2, 0


    elif args.dataset_name == 'WEBEmo':
        train_accuracy = []
        train_F1_weighted = []
        train_F1_macro = []
        loss /= num_samples
        for i in range(len(y_true)):
            train_accuracy.append(accuracy_score(y_true[i], y_pred[i]))
            train_F1_weighted.append(f1_score(y_true[i], y_pred[i], average='weighted'))
            train_F1_macro.append(f1_score(y_true[i], y_pred[i], average='macro'))
        return float(loss), train_accuracy, train_F1_weighted, train_F1_macro

    else:
        assert args.dataset_name in (args.MSA_dataset + ['FI', 'Emotion6', 'UnbiasedEmo', 'IMDB'])
        loss /= num_samples
        train_accuracy = accuracy_score(y_true, y_pred)
        train_F1_weighted = f1_score(y_true, y_pred, average='weighted')
        train_F1_macro = f1_score(y_true, y_pred, average='macro')

        return float(loss), train_accuracy, train_F1_weighted, train_F1_macro

def output_epoch_result(args, mode, accuracy, F1_weighted , F1_micro, loss):
    if args.dataset_name == 'Emotic-reg':
        average_precision = float(np.mean(np.array(accuracy)))
        average_F1 = float(np.mean(np.array(F1_weighted)))
        auc = float(F1_micro)
        save_content = mode + ': AP: %.4f, AUC: %.4f, total loss: %.4f' % \
                       (average_precision, auc, loss)
        output_to_log(args, save_content)
    elif args.dataset_name == 'WEBEmo':
        save_content = mode + ': total loss: %.4f \n' \
                              '    Acc2:  %.4f, F1(weighted): %.4f, F1(macro): %.4f\n' \
                              '    Acc7:  %.4f, F1(weighted): %.4f, F1(macro): %.4f\n' \
                              '    Acc25: %.4f, F1(weighted): %.4f, F1(macro): %.4f\n' % \
                       (loss, accuracy[0], F1_weighted[0], F1_micro[0],
                              accuracy[1], F1_weighted[1], F1_micro[1],
                              accuracy[2], F1_weighted[2], F1_micro[2])
        output_to_log(args, save_content)
    else:
        #assert args.dataset_name in (args.MSA_dataset + ['FI', 'Emotion6', 'UnbiasedEmo', 'IMDB'])
        save_content = mode + ': Accuracy: %.4f, F1(weighted): %.4f, F1(macro): %.4f, total loss: %.4f' % \
                       (accuracy, F1_weighted, F1_micro, loss)
        output_to_log(args, save_content)


# mode = Train/Dev/Test/Hard
def proceed_epoch(args, model:nn.Module, dataloader, loss_func, optimizer, scheduler, mode):
    if mode == 'Train':
        #y_true, y_pred, total_loss, num_labels = dt.debug_epoch(args, model, dataloader, loss_func, optimizer)
        y_true, y_pred, loss, num_labels = train_epoch(args, model, dataloader, loss_func, optimizer, scheduler)
    else:
        assert mode in ['Dev', 'Test', 'Hard']
        y_true, y_pred, loss, num_labels = dev_epoch(args, model, dataloader, loss_func, mode)

    loss, accuracy, F1_weighted, F1_micro = evaluate_epoch(args, y_true, y_pred, loss, num_labels)
    output_epoch_result(args, mode, accuracy, F1_weighted , F1_micro, loss)
    return [accuracy, F1_weighted , F1_micro, loss]


def output_to_log(args, content):
    print(content)
    with open(args.log_name, 'a+', encoding='utf-8') as f:
        f.write(content + '\n')

def output_to_error_log(args, content):
    print(content)
    with open(args.error_log, 'a+', encoding='utf-8') as f:
        f.write(content + '\n')


def train_model(args, epoch_num, model, dataloader_list, loss_func, optimizer, scheduler):
    [train_loader, dev_loader, test_loader, hard_loader] = dataloader_list
    # dev_acc, dev_F1, test_acc, test_F1, hard_acc, hard_F1
    ap_eval_metric_name = ("dev_ap", "dev_auc", "test_ap", "test_auc", "hard_ap", "hard_auc")
    acc_eval_metric_name = ("dev_acc", "dev_F1", "test_acc", "test_F1", "hard_acc", "hard_F1")
    webemo_eval_metric_name = ("test_acc2", "test_2_F1", "test_acc7", "test_7_F1", "test_acc25", "test_25_F1",)

    if args.dataset_name in args.AA_dataset:
        dataloader_name = ("Train", "Dev", "Test", "Hard")
    elif args.dataset_name in args.TO_dataset:
        dataloader_name = ("Train", "Test")
    else:
        dataloader_name = ("Train", "Dev", "Test")

    acc_list, F1_weighted_list, F1_micro_list, ap_list, auc_list = None, None, None, None, None
    #To save precision for each class
    best_dev_info, best_test_info = None, None
    if args.dataset_name == 'Emotic-reg':
        best_F1_acc = [0] * (2 * (len(dataloader_name) - 1))
        best_epoch = [-1] * (2 * (len(dataloader_name) - 1))

        loss_list = [[] for _ in range(len(dataloader_name))]
        ap_list = [[] for _ in range(len(dataloader_name))]
        auc_list = [[] for _ in range(len(dataloader_name))]
    elif args.dataset_name == 'WEBEmo':
        best_F1_acc = [0] * (6 * (len(dataloader_name) - 1))
        best_epoch = [-1] * (6 * (len(dataloader_name) - 1))

        loss_list = [[] for _ in range(len(dataloader_name))]
        acc_list = [[[],[],[]] for _ in range(len(dataloader_name))]
        F1_weighted_list = [[[],[],[]] for _ in range(len(dataloader_name))]
    else:
        best_F1_acc = [0] * (2 * (len(dataloader_name) - 1))
        best_epoch = [-1] * (2 * (len(dataloader_name) - 1))

        loss_list = [[] for _ in range(len(dataloader_name))]
        acc_list = [[] for _ in range(len(dataloader_name))]
        F1_weighted_list = [[] for _ in range(len(dataloader_name))]
        F1_micro_list = [[] for _ in range(len(dataloader_name))]

    for epoch in range(epoch_num):
        args.cur_epoch = epoch
        epoch_id = 'Train epoch: %d/%d:' %(args.cur_epoch + 1, epoch_num)
        output_to_log(args, epoch_id)

        #metric = [acc, F1_weighted, F1_micro, total_loss]
        train_metric = proceed_epoch(args, model, train_loader, loss_func, optimizer, scheduler, 'Train')
        if args.dataset_name in args.TO_dataset:
            test_metric = proceed_epoch(args, model, test_loader, loss_func, optimizer, scheduler, 'Test')
            metric_list = [train_metric, test_metric]
        else:
            dev_metric = proceed_epoch(args, model, dev_loader, loss_func, optimizer, scheduler, 'Dev')
            test_metric = proceed_epoch(args, model, test_loader, loss_func, optimizer, scheduler, 'Test')
            metric_list = [train_metric, dev_metric, test_metric]
            if args.dataset_name in args.AA_dataset:
                hard_metric = proceed_epoch(args, model, hard_loader, loss_func, optimizer, scheduler, 'Hard')
                metric_list = [train_metric, dev_metric, test_metric, hard_metric]

        for i in range(len(metric_list)):
            if args.dataset_name == 'Emotic-reg':
                ap_list[i].append(np.mean(np.array(metric_list[i][0])))
                auc_list[i].append(metric_list[i][2])
                loss_list[i].append(metric_list[i][3])
            elif args.dataset_name == 'WEBEmo':
                for j in range(len(metric_list[i][0])):
                    acc_list[i][j].append(metric_list[i][0][j])
                    F1_weighted_list[i][j].append(metric_list[i][1][j])
                loss_list[i].append(metric_list[i][3])
            else:
                acc_list[i].append(metric_list[i][0])
                F1_weighted_list[i].append(metric_list[i][1])
                F1_micro_list[i].append(metric_list[i][2])
                loss_list[i].append(metric_list[i][3])


        best_flag = False
        epoch_dscrpt = 'New records: '
        cur_F1_acc = []
        for metric in metric_list[1:]:
            # Multiple label prediction task
            if args.dataset_name == 'Emotic-reg':
                # Record AP and AUC score
                cur_F1_acc.append(np.mean(np.array(metric[0])))
                cur_F1_acc.append(metric[2])
            # Single label prediction task
            elif args.dataset_name == 'WEBEmo':
                for i in range(len(metric[0])):
                    cur_F1_acc.append(metric[0][i])
                    cur_F1_acc.append(metric[1][i])
            #Single label prediction task
            else:
                #Record Accuracy and F1 score
                for j in range(2):
                    cur_F1_acc.append(metric[j])

        for i in range(len(cur_F1_acc)):
            if cur_F1_acc[i] > best_F1_acc[i]:
                best_flag = True
                acc_improve = cur_F1_acc[i] - best_F1_acc[i]
                best_F1_acc[i] = cur_F1_acc[i]
                best_epoch[i] = epoch
                # Multiple label prediction task
                if args.dataset_name == 'Emotic-reg':
                    epoch_dscrpt += ap_eval_metric_name[i] + '(+%.4f) ' % (acc_improve)
                    if i == 0:  # Save according to the best dev AP
                        best_dev_info = dev_metric[0]
                        if args.save_model != '0':
                            torch.save(model.state_dict(), args.save_dir + '/best_model.pth')
                    if i == 2:  # the best test AP
                        best_test_info = test_metric[0]

                # Single label prediction task
                elif args.dataset_name == 'WEBEmo':
                    epoch_dscrpt += webemo_eval_metric_name[i] + '(+%.4f) ' % (acc_improve)
                    if i == 1:  # Save according to the best dev F1
                        if args.save_model != '0':
                            torch.save(model.state_dict(), args.save_dir + '/best_model.pth')

                # Single label prediction task
                else:
                    epoch_dscrpt += acc_eval_metric_name[i] + '(+%.4f) ' %(acc_improve)
                    if i == 1:  # Save according to the best dev F1
                        #if args.save_model != '0':
                        torch.save(model.state_dict(), args.save_dir + '/best_model.pth')

        with open(args.log_name, 'a+', encoding='utf-8') as f:
            if best_flag == False:
                f.write("Model is not best up to current epoch.\n\n")
            else:
                f.write(epoch_dscrpt + '\n\n')

    if args.dataset_name == 'Emotic-reg':
        # draw metric curve for multiple label prediction task
        draw_metric_curve_emotic(epoch_num, args.metric_dir, ap_list, auc_list, loss_list, dataloader_name)
        record_best_epochs(args, best_F1_acc, best_epoch, ap_eval_metric_name, ap_list, auc_list, ("AP", "AUC"))
        record_each_label_precision(args, best_dev_info, best_test_info)
    elif args.dataset_name == 'WEBEmo':
        draw_metric_curve_webemo(epoch_num, args.metric_dir, acc_list, F1_weighted_list, loss_list, dataloader_name)
        record_best_epochs(args, best_F1_acc, best_epoch, webemo_eval_metric_name, acc_list, F1_weighted_list, ("Acc", "F1"))
    else:
        # draw metric curve for single label prediction task
        draw_metric_curve_msa_dataset(epoch_num, args.metric_dir, acc_list, F1_weighted_list, F1_micro_list, loss_list, dataloader_name)
        record_best_epochs(args, best_F1_acc, best_epoch, acc_eval_metric_name, acc_list, F1_weighted_list, ("Acc", "F1"))



def draw_metric_curve_msa_dataset(epoch_num, save_path, acc_list, F1_weighted_list, F1_micro_list, loss_list, dataloader_name):
    x = range(epoch_num)

    plt.figure()
    draw_single_metric_curve(save_path, x, acc_list, dataloader_name, "/acc.jpg")
    plt.clf()

    draw_single_metric_curve(save_path, x, F1_weighted_list, dataloader_name, "/F1_weighted.jpg")
    plt.clf()

    draw_single_metric_curve(save_path, x, F1_micro_list, dataloader_name, "/F1_micro.jpg")
    plt.clf()

    draw_single_metric_curve(save_path, x, loss_list, dataloader_name, "/loss.jpg")
    plt.cla()

def draw_metric_curve_webemo(epoch_num, save_path, acc_list, F1_weighted_list, loss_list, dataloader_name):
    x = range(epoch_num)

    plt.figure()
    draw_multiple_metric_curve(save_path, x, acc_list, dataloader_name, "/acc.jpg")
    plt.clf()

    draw_multiple_metric_curve(save_path, x, F1_weighted_list, dataloader_name, "/F1_weighted.jpg")
    plt.clf()

    draw_single_metric_curve(save_path, x, loss_list, dataloader_name, "/loss.jpg")
    plt.cla()


def draw_metric_curve_emotic(epoch_num, save_path, ap_list, auc_list, loss_list, dataloader_name):
    x = range(epoch_num)

    plt.figure()
    draw_single_metric_curve(save_path, x, ap_list, dataloader_name, "/ap.jpg")
    plt.clf()

    draw_single_metric_curve(save_path, x, auc_list, dataloader_name, "/auc.jpg")
    plt.clf()

    draw_single_metric_curve(save_path, x, loss_list, dataloader_name, "/loss.jpg")
    plt.cla()

def draw_single_metric_curve(save_path, x_list, y_list, dataloader_name, discriptor):
    y_low_bound = min([min([y_list[i][j] for j in range(len(y_list[i]))]) for i in range(len(y_list))]) * 0.95
    y_up_bound = max([max([y_list[i][j] for j in range(len(y_list[i]))]) for i in range(len(y_list))]) * 1.05
    for i in range(len(y_list)):
        plt.subplot(2, 2, i + 1)
        plt.plot(x_list, y_list[i], '.-')
        plt.ylim(y_low_bound, y_up_bound)
        plt.title(dataloader_name[i])
        #plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(save_path + discriptor)

def draw_multiple_metric_curve(save_path, x_list, y_list_list, dataloader_name, discriptor):
    y_tmp = np.array(y_list_list)
    y_tmp = y_tmp.flatten()
    y_up_bound = y_tmp.flatten().max() * 1.05
    y_low_bound = y_tmp.flatten().min() * 0.95
    for i in range(len(y_list_list)):
        assert len(y_list_list) <= 4
        plt.subplot(2, 2, i + 1)
        plt.plot(x_list, y_list_list[i][0], '.-')
        plt.plot(x_list, y_list_list[i][1], '.-')
        plt.plot(x_list, y_list_list[i][2], '.-')
        plt.ylim(y_low_bound, y_up_bound)
        plt.title(dataloader_name[i])
        # plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(save_path + discriptor)


def record_best_epochs(args, best_F1_acc, best_epoch, eval_metric_name, acc_list, F1_weighted_list, discriptor):
    save_content = []
    if args.dataset_name == 'WEBEmo':
        for i in range(len(best_F1_acc)):
            acc_list_index = int(i // 2)
            epoch = best_epoch[i]
            if i % 2 == 1:
                if epoch == best_epoch[i - 1]:
                    save_content.pop()
                    save_content.append("Best %s and %s at epoch %d: %s (%.4f), %s (%.4f)\n" \
                                        % (eval_metric_name[i - 1], eval_metric_name[i], epoch + 1,
                                           discriptor[0], acc_list[1][acc_list_index][epoch],
                                           discriptor[1], F1_weighted_list[1][acc_list_index][epoch]))
                    continue
            save_content.append("Best %s at epoch %d: %s (%.4f), %s (%.4f)\n" \
                                % (eval_metric_name[i], epoch + 1,
                                   discriptor[0], acc_list[1][acc_list_index][epoch],
                                   discriptor[1], F1_weighted_list[1][acc_list_index][epoch]))
            # save_content += "Best %s: %.4f epoch: %d\n" %(eval_metric_name[i], best_F1_acc[i], best_epoch[i] + 1)
    else:
        for i in range(len(best_F1_acc)):
            acc_list_index = int(i // 2) + 1
            epoch = best_epoch[i]
            if i % 2 == 1:
                if epoch == best_epoch[i - 1]:
                    save_content.pop()
                    save_content.append("Best %s and %s at epoch %d: %s (%.4f), %s (%.4f)\n" \
                                        % (eval_metric_name[i - 1], eval_metric_name[i], epoch + 1,
                                           discriptor[0], acc_list[acc_list_index][epoch],
                                           discriptor[1], F1_weighted_list[acc_list_index][epoch]))
                    continue
            save_content.append("Best %s at epoch %d: %s (%.4f), %s (%.4f)\n" \
                                % (eval_metric_name[i], epoch + 1,
                                   discriptor[0], acc_list[acc_list_index][epoch],
                                   discriptor[1], F1_weighted_list[acc_list_index][epoch]))
            # save_content += "Best %s: %.4f epoch: %d\n" %(eval_metric_name[i], best_F1_acc[i], best_epoch[i] + 1)

    with open(args.log_name, 'a+', encoding='utf-8') as f:
        f.write("".join(save_content))


def record_each_label_precision(args, best_dev_info, best_test_info):
    cfg = Config()
    dataset_cfg = cfg.get_dataset_config()
    Emotic_label_map = dataset_cfg['Emotic']['label_map']

    save_content = "\n"
    info_list = [best_dev_info, best_test_info]
    discriptor_list = ('Dev precision', 'Test precision')
    key_list = list(Emotic_label_map.keys())
    for j in range(len(info_list)):
        save_content += discriptor_list[j] + ':\n'
        for i in range(len(key_list)):
            key = key_list[i]
            precision = info_list[j][i]
            save_content += "%s(%.4f)\n" % (key, precision)
        save_content += '\n'

    with open(args.log_name, 'a+', encoding='utf-8') as f:
        f.write(save_content)

def zs_dev_epoch(args, model:nn.Module, dataloader, mode):
    param = ModelParam()

    with torch.no_grad():
        model.eval()
        dev_loader_tqdm = tqdm(dataloader, desc=mode + ' Iteration:')
        if args.dataset_name in ['FI', 'Emotion6', 'UnbiasedEmo', 'Emotic'] + args.MSA_dataset:
            y_true = []
            y_pred = []
            for index, data in enumerate(dev_loader_tqdm):
                if args.dataset_name == 'Emotic':
                    image, label, vad_label = read_from_batch(args, data)
                elif args.dataset_name in ['FI', 'Emotion6', 'UnbiasedEmo']:
                    image, label = read_from_batch(args, data)
                else:
                    text, image, label, ocr = read_from_batch(args, data)

                text = process_text(args.text_prompt, model, args.text_max_length)
                param.set_data_param(text, None, image, None, None, None, None)
                logits_per_text, logits_per_image = model(param)

                _, predicted_label = torch.max(logits_per_image, 1)
                y_true.extend(label.cpu())
                y_pred.extend(predicted_label.cpu())

            accuracy = accuracy_score(y_true, y_pred)
            F1_weighted = f1_score(y_true, y_pred, average='weighted')
            F1_micro = f1_score(y_true, y_pred, average='macro')

            save_content = mode + ': Accuracy: %.4f, F1(weighted): %.4f, F1(macro): %.4f' % \
                               (accuracy, F1_weighted, F1_micro)
            output_to_log(args, save_content)

        else:
            assert args.dataset_name == 'WEBEmo'
            y_true = [[], [], []]
            y_pred = [[], [], []]
            for index, data in enumerate(dev_loader_tqdm):
                image, label1, label2, label3 = read_from_batch(args, data)

                text = process_text(args.text_prompt[0], model, args.text_max_length)
                param.set_data_param(text, None, image, None, None, None, None)
                logits_per_text, logits_per_image = model(param)

                _, predicted_label = torch.max(logits_per_image, 1)
                y_true[0].extend(label1.cpu())
                y_pred[0].extend(predicted_label.cpu())

                text = process_text(args.text_prompt[1], model, args.text_max_length)
                param.set_data_param(text, None, image, None, None, None, None)
                logits_per_text, logits_per_image = model(param)

                _, predicted_label = torch.max(logits_per_image, 1)
                y_true[1].extend(label2.cpu())
                y_pred[1].extend(predicted_label.cpu())

                text = process_text(args.text_prompt[2], model, args.text_max_length)
                param.set_data_param(text, None, image, None, None, None, None)
                logits_per_text, logits_per_image = model(param)

                _, predicted_label = torch.max(logits_per_image, 1)
                y_true[2].extend(label3.cpu())
                y_pred[2].extend(predicted_label.cpu())

                ###################################
            accuracy = [accuracy_score(y_true[i], y_pred[i]) for i in range(len(y_true))]
            F1_weighted = [f1_score(y_true[i], y_pred[i], average='weighted') for i in range(len(y_true))]
            F1_micro = [f1_score(y_true[i], y_pred[i], average='macro') for i in range(len(y_true))]

            save_content = mode + ':' \
                                  '    Acc2:  %.4f, F1(weighted): %.4f, F1(macro): %.4f\n' \
                                  '    Acc7:  %.4f, F1(weighted): %.4f, F1(macro): %.4f\n' \
                                  '    Acc25: %.4f, F1(weighted): %.4f, F1(macro): %.4f\n' % \
                           (accuracy[0], F1_weighted[0], F1_micro[0],
                            accuracy[1], F1_weighted[1], F1_micro[1],
                            accuracy[2], F1_weighted[2], F1_micro[2])
            output_to_log(args, save_content)

    return [accuracy, F1_weighted , F1_micro]

def zero_shot_infer(args, model, dataloader_list):
    [train_loader, dev_loader, test_loader, hard_loader] = dataloader_list
    # dev_acc, dev_F1, test_acc, test_F1, hard_acc, hard_F1
    eval_metric_name = ("dev_acc", "dev_F1", "test_acc", "test_F1", "hard_acc", "hard_F1")

    dataloader_name = ("Train", "Dev", "Test", "Hard")

    title = 'Zero shot inference begin:'
    output_to_log(args, title)
    #train_metric = zs_dev_epoch(args, model, train_loader, 'Train')
    if args.dataset_name in args.TO_dataset:
        test_metric = zs_dev_epoch(args, model, test_loader, 'Test')
    else:
        dev_metric = zs_dev_epoch(args, model, dev_loader, 'Dev')
        test_metric = zs_dev_epoch(args, model, test_loader, 'Test')
        if args.dataset_name in args.AA_dataset:
            hard_metric = zs_dev_epoch(args, model, hard_loader, 'Hard')

    #metric = [acc, F1_weighted, F1_micro]

def process_text(text_list, model, text_max_length):
    tokenizor = model.get_text_tokenizor()
    text_list = tokenizor(text_list, padding=True, return_tensors="pt", truncation=True, max_length=text_max_length)
    return text_list