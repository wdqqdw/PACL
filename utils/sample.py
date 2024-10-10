import re
import json
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class AbbreviationReduction():

    def __init__(self):
        replacement_patterns = [
            (r'won\'t', 'will not'),
            (r'can\'t', 'can not'),
            (r'i\'m', 'i am'),
            (r'ain\'t', 'is not'),
            (r'(\w+)\'ll', '\g<1> will'),
            (r'(\w+)n\'t', '\g<1> not'),
            (r'(\w+)\'ve', '\g<1> have'),
            (r'(\w+)\'s', '\g<1> is'),
            (r'(\w+)\'re', '\g<1> are'),
            (r'(\w+)\'d', '\g<1> would')]

        self.patterns = [(re.compile(regex, re.I), repl)
                         for (regex, repl) in replacement_patterns]

    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)
        return s

class UrlProcess():
    def __init__(self):
        # remove start with #
        self.hashtag_pattern = re.compile('.?#[a-zA-Z0-9_\.]+', re.I)
        # self.hashtag_pattern = re.compile('#[a-zA-Z0-9]+')
        # remove start with @
        self.at_pattern = re.compile('.?@[a-zA-Z0-9_\.]+', re.I)
        # self.at_pattern = re.compile('@[a-zA-Z0-9]+')
        # remove url
        self.http_pattern = re.compile("(http|ftp|https)://[a-zA-Z0-9\./]+|www\.(\w+\.)+\S*/", re.I)
        # self.http_pattern = re.compile(
        #     "((http|ftp|https)://)(([a-zA-Z0-9\._-]+\.[a-zA-Z]{2,6})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(:[0-9]{1,4})*(/[a-zA-Z0-9\&%_\./-~-]*)?")

    def replace(self, text):
        text = re.sub(self.hashtag_pattern, '', text)
        text = re.sub(self.at_pattern, '', text)
        text = re.sub(self.http_pattern, '', text)
        return text

class TextProcess_3():
    def __init__(self):
        self.abbreviation_reduction = AbbreviationReduction()
        #self.repeat_replacer = RepeatReplacer()
        self.url_replacer = UrlProcess()

    def process(self, text):
        text = self.abbreviation_reduction.replace(text)
        #text = self.repeat_replacer.replace(text)
        text = self.url_replacer.replace(text)
        return text

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0',
                        help='which GPU to run')
    parser.add_argument('-dataset_name', type=str, default='MVSA-multiple',
                        help='support MVSA-single/MVSA-multiple/HFM/TumEmo/FI/Emotion6/UnbiasedEmo/ CC3M/Valid_split ')
    parser.add_argument('-sample_id', type=str, default='image',
                        help='suffixes to denote defferent partition')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    MSA_dataset = ['MVSA-single', 'MVSA-multiple', 'TumEmo', 'HFM']
    pretrain_dataset = ['CC3M/Valid_split']
    if args.dataset_name in MSA_dataset:
        source_dir = '/change_to_your_path/MSA_dataset/'
    elif args.dataset_name in pretrain_dataset:
        source_dir = '/change_to_your_path/pretrain_dataset/'
    else:
        source_dir = '/change_to_your_path/VSA_dataset/'
    sample_dir = '/10-fold-1/'
    visible_train_path = source_dir + args.dataset_name + sample_dir + 'train_' + args.sample_id + '.json'
    blind_train_path = source_dir + args.dataset_name + sample_dir + 'train_' + args.sample_id + '_x.json'
    test_path = source_dir + args.dataset_name + sample_dir + 'test_' + args.sample_id + '.json'
    dev_path = source_dir + args.dataset_name + sample_dir + 'dev_' + args.sample_id + '.json'
    visible_hard_path = source_dir + args.dataset_name + sample_dir + 'hard_' + args.sample_id + '.json'
    blind_hard_path = source_dir + args.dataset_name + sample_dir + 'hard.json'

    set_length = {"MVSA-single": [4869, 3611, 450, 450, 358], "MVSA-multiple": [17366, 13624, 1700, 1700, 342],
                  "HFM": [24635, 19816, 2410, 2409], "TumEmo": [186026, 148822, 18602, 18602],
                  "FI": [23185, 18549, 2318, 2318], "Emotion6": [8349, 6679, 835, 835],
                  "UnbiasedEmo": [3045, 2437, 304, 304], "CC3M/Valid_split": [11832, 9466, 1183, 1183]}
    label_map = {"MVSA-single": {"Positive": 0, "Neutral": 1, "Negative": 2},
                 "MVSA-multiple": {"Positive": 0, "Neutral": 1, "Negative": 2},
                  "HFM": {"Positive": 0, "Negative": 1},
                 "TumEmo": {"Love": 0, "Happy": 1, "Calm": 2, "Bored": 3, "Sad": 4, "Angry": 5, "Fear": 6},
                 "FI": {'amusement': 1, 'awe': 2, 'anger': 6, 'contentment': 3,
                        'disgust': 5, 'excitement': 0, 'fear': 7, 'sadness': 4},
                 "Emotion6":{'anger': 4, 'fear': 5, 'joy': 1, 'love': 0, 'sadness': 3, 'surprise': 2},
                 "UnbiasedEmo":{'anger': 4, 'fear': 5, 'joy': 1, 'love': 0, 'sadness': 3, 'surprise': 2},}

    label_path = source_dir + args.dataset_name + '/labelProcessed.json'
    ocr_path = source_dir + args.dataset_name + '/' + args.dataset_name + '_ocr.json'
    text_path = source_dir + args.dataset_name + '/dataset_image'
    HFM_text_path = source_dir + args.dataset_name + '/text.json'

    with open(label_path, 'r', encoding='utf-8') as label_read:
        label_content = json.load(label_read)

    if args.dataset_name in MSA_dataset:
        with open(ocr_path, 'r', encoding='utf-8') as ocr_read:
            ocr_content = json.load(ocr_read)

    tp3 = TextProcess_3()

    if args.dataset_name in ["MVSA-single", "MVSA-multiple"]:

        print("Sample Dataset: ", args.dataset_name, " Following files will be created:")
        print(visible_train_path)
        print(blind_train_path)
        print(test_path)
        print(dev_path)
        print(visible_hard_path)
        print(blind_hard_path)

        while True:
            vld = input("Enter yes/no to continue/exit:")
            if vld == 'Yes' or vld == 'yes' or vld == 'Y' or vld == 'y':
                break
            elif vld == 'No' or vld == 'no' or vld == 'N' or vld == 'n':
                exit()

        path_list = [visible_train_path, blind_train_path, test_path, dev_path, visible_hard_path, blind_hard_path]
        content_list = [[], [], [], [], [], []]

        hard_set = [[] for i in range(len(label_map[args.dataset_name]))]
        easy_set = [[] for i in range(len(label_map[args.dataset_name]))]

        sd_count = 0
        for i in label_content:
            senti_index = label_map[args.dataset_name][i["image_text"]]
            if i["text"] == "Positive" and i["image"] == "Negative":
                hard_set[senti_index].append(i["id"])
            elif i["text"] == "Negative" and i["image"] == "Positive":
                hard_set[senti_index].append(i["id"])
            else:
                if i["text"] != i["image"]:
                    sd_count += 1
                easy_set[senti_index].append(i["id"])


        easy_set_sttts = {}
        hard_set_sttts = {}
        len_easy_set = 0
        len_hard_set = 0
        for k in label_map[args.dataset_name].keys():
            len_easy_set += len(easy_set[label_map[args.dataset_name][k]])
            len_hard_set += len(hard_set[label_map[args.dataset_name][k]])
            easy_set_sttts[k] = len(easy_set[label_map[args.dataset_name][k]])
            hard_set_sttts[k] = len(hard_set[label_map[args.dataset_name][k]])
        print("Easy set statistics:", easy_set_sttts)
        print("Hard set statistics:", hard_set_sttts)

        assert len_hard_set == set_length[args.dataset_name][4]
        assert len_easy_set + len_hard_set == set_length[args.dataset_name][0]

        easy_sample_partition = {'train':set_length[args.dataset_name][1],
                                'dev':set_length[args.dataset_name][2],
                                'test':set_length[args.dataset_name][3]}
        easy_sample_partition_dict = {'train': {}, 'dev': {}, 'test': {}}

        hard_sample_partition = {'vsb':set_length[args.dataset_name][4]//2,
                                'bld':set_length[args.dataset_name][4] - set_length[args.dataset_name][4]//2}
        hard_sample_partition_dict = {'vsb': {}, 'bld': {}}

        #partition according to usage
        for k in easy_sample_partition.keys():
            num_sample_usage_k = easy_sample_partition[k]
            partition_len = {}
            current_sample_usage_k_int = 0
            current_sample_usage_k = 0
            for i in easy_set_sttts.keys():
                num_partition = num_sample_usage_k * easy_set_sttts[i] / len_easy_set
                current_sample_usage_k += num_partition
                #Last catagory
                if int(current_sample_usage_k) == num_sample_usage_k:
                    partition_len[i] = num_sample_usage_k - current_sample_usage_k_int
                else:
                    current_sample_usage_k_int += int(num_partition)
                    partition_len[i] = int(num_partition)
            easy_sample_partition_dict[k] = partition_len

        for k in easy_set_sttts.keys():
            num_label_k = 0
            for j in easy_sample_partition.keys():
                num_label_k += easy_sample_partition_dict[j][k]
            num_label_k_lack = easy_set_sttts[k] - num_label_k
            easy_sample_partition_dict['train'][k] += num_label_k_lack

        print(easy_sample_partition)
        print(easy_sample_partition_dict)

        #partition according to usage
        for k in hard_sample_partition.keys():
            num_sample_usage_k = hard_sample_partition[k]
            partition_len = {}
            current_sample_usage_k_int = 0
            current_sample_usage_k = 0
            for i in hard_set_sttts.keys():
                num_partition = num_sample_usage_k * hard_set_sttts[i] / len_hard_set
                current_sample_usage_k += num_partition
                #Last catagory
                if int(current_sample_usage_k) == num_sample_usage_k:
                    partition_len[i] = num_sample_usage_k - current_sample_usage_k_int
                else:
                    current_sample_usage_k_int += int(num_partition)
                    partition_len[i] = int(num_partition)
            hard_sample_partition_dict[k] = partition_len

        for k in hard_set_sttts.keys():
            num_label_k = 0
            for j in hard_sample_partition.keys():
                num_label_k += hard_sample_partition_dict[j][k]
            num_label_k_lack = hard_set_sttts[k] - num_label_k
            hard_sample_partition_dict['vsb'][k] += num_label_k_lack

        print(hard_sample_partition)
        print(hard_sample_partition_dict)

        train_vsb, train_bld, test, val, hard_left, hard_full = [], [], [], [], [], []
        for label in easy_set_sttts.keys():
            senti_index = label_map[args.dataset_name][label]
            # do not contain hard sample
            sub_train_val, sub_test = train_test_split(easy_set[senti_index], test_size=easy_sample_partition_dict['test'][label])
            # train_bld for blind set
            sub_train_bld, sub_val = train_test_split(sub_train_val, test_size=easy_sample_partition_dict['dev'][label])
            # train for visible set
            sub_train_vsb, _ = train_test_split(sub_train_bld, test_size=hard_sample_partition_dict['vsb'][label])
            # hard_left for test, hard_right for train
            sub_hard_left, sub_hard_right = train_test_split(hard_set[senti_index], test_size=hard_sample_partition_dict['vsb'][label])

            sub_train_vsb.extend(sub_hard_right)

            train_vsb.extend(sub_train_vsb)
            train_bld.extend(sub_train_bld)
            test.extend(sub_test)
            val.extend(sub_val)
            hard_left.extend(sub_hard_left)
            hard_full.extend(hard_set[senti_index])

        assert len(train_bld) == set_length[args.dataset_name][1]
        assert len(train_vsb) == set_length[args.dataset_name][1]
        assert len(val) == set_length[args.dataset_name][2]
        assert len(test) == set_length[args.dataset_name][3]

        id_list = [train_vsb, train_bld, test, val, hard_left, hard_full]
        hard_scene_text = 0
        for i in tqdm(range(len(label_content))):
            sample_info = {"id": None, "text": None, "emotion_label": -1, "ocr": [], "text_with_ocr": None}
            label_info = label_content[i]
            ocr_info = ocr_content[i]
            assert label_info["id"] == ocr_info["id"]

            sample_info["id"] = label_info["id"]

            text_index_path = text_path + '/' + sample_info["id"] + '.txt'
            with open(text_index_path, 'r', encoding='unicode_escape') as text:
                sample_info["text"] = text.read()[:-1]

            sample_info["emotion_label"] = label_map[args.dataset_name][label_info["image_text"]]
            #sample_info["emotion_label"] = label_map[args.dataset_name][label_info["image"]]
            sample_info["ocr"] = ocr_info["ocr"]
            if len(sample_info["ocr"]) == 0:
                sample_info["text_with_ocr"] = sample_info["text"] + " <ocr> "
            else:
                if sample_info["id"] in id_list[5]:
                    hard_scene_text += 1
                scene_text = " ".join(sample_info["ocr"])
                sample_info["text_with_ocr"] = sample_info["text"] + " <ocr> " + scene_text

            #path_list = [visible_train_path, blind_train_path, test_path, dev_path, visible_hard_path, blind_hard_path]
            for j in range(len(id_list)):
                if sample_info["id"] in id_list[j]:
                    content_list[j].append(sample_info)

        for index in range(len(content_list)):
            json_data = json.dumps(content_list[index], indent=2)
            with open(path_list[index], 'w') as f:
                f.write(json_data)
        #print(hard_scene_text)
        print("New dasetset partition have been created successfully")


    if args.dataset_name in ["HFM", "TumEmo"]:

        print("Sample Dataset: ", args.dataset_name, " Following files will be created:")
        print(visible_train_path)
        print(test_path)
        print(dev_path)

        while True:
            vld = input("Enter yes/no to continue/exit:")
            if vld == 'Yes' or vld == 'yes' or vld == 'Y' or vld == 'y':
                break
            elif vld == 'No' or vld == 'no' or vld == 'N' or vld == 'n':
                exit()

        path_list = [visible_train_path, test_path, dev_path]
        content_list = [[], [], []]

        easy_set = [[] for i in range(len(label_map[args.dataset_name]))]

        for i in label_content:
            senti_index = label_map[args.dataset_name][i["image_text"]]
            easy_set[senti_index].append(i["id"])


        easy_set_sttts = {}
        len_easy_set = 0
        for k in label_map[args.dataset_name].keys():
            len_easy_set += len(easy_set[label_map[args.dataset_name][k]])
            easy_set_sttts[k] = len(easy_set[label_map[args.dataset_name][k]])
        print("Easy set statistics:", easy_set_sttts)

        assert len_easy_set == set_length[args.dataset_name][0]

        easy_sample_partition = {'train':set_length[args.dataset_name][1],
                                'dev':set_length[args.dataset_name][2],
                                'test':set_length[args.dataset_name][3]}
        easy_sample_partition_dict = {'train': {}, 'dev': {}, 'test': {}}

        #partition according to usage
        for k in easy_sample_partition.keys():
            num_sample_usage_k = easy_sample_partition[k]
            partition_len = {}
            current_sample_usage_k_int = 0
            current_sample_usage_k = 0
            for i in easy_set_sttts.keys():
                num_partition = num_sample_usage_k * easy_set_sttts[i] / len_easy_set
                current_sample_usage_k += num_partition
                #Last catagory
                if int(current_sample_usage_k) == num_sample_usage_k:
                    partition_len[i] = num_sample_usage_k - current_sample_usage_k_int
                else:
                    current_sample_usage_k_int += int(num_partition)
                    partition_len[i] = int(num_partition)
            easy_sample_partition_dict[k] = partition_len

        for k in easy_set_sttts.keys():
            num_label_k = 0
            for j in easy_sample_partition.keys():
                num_label_k += easy_sample_partition_dict[j][k]
            num_label_k_lack = easy_set_sttts[k] - num_label_k
            easy_sample_partition_dict['train'][k] += num_label_k_lack

        print(easy_sample_partition)
        print(easy_sample_partition_dict)


        train, test, val = [], [], []
        for label in easy_set_sttts.keys():
            senti_index = label_map[args.dataset_name][label]
            sub_train_val, sub_test = train_test_split(easy_set[senti_index], test_size=easy_sample_partition_dict['test'][label])
            sub_train, sub_val = train_test_split(sub_train_val, test_size=easy_sample_partition_dict['dev'][label])

            train.extend(sub_train)
            test.extend(sub_test)
            val.extend(sub_val)

        assert len(train) == set_length[args.dataset_name][1]
        assert len(val) == set_length[args.dataset_name][2]
        assert len(test) == set_length[args.dataset_name][3]

        id_list = [train, test, val]

        if args.dataset_name == "HFM":
            with open(HFM_text_path, 'r', encoding='utf-8') as text_read:
                text_content = json.load(text_read)
        else:
            text_content = None

        for i in tqdm(range(len(label_content))):
            sample_info = {"id": None, "text": None, "emotion_label": -1, "ocr": [], "text_with_ocr": None}
            label_info = label_content[i]
            ocr_info = ocr_content[i]
            #assert label_info["id"] == ocr_info["id"]

            sample_info["id"] = label_info["id"]

            if args.dataset_name == "HFM":
                text_info = text_content[i]
                assert text_info["id"] == sample_info["id"]
                sample_info["text"] = tp3.process(text_info["text"])
            else:
                text_index_path = text_path + '/' + sample_info["id"] + '.txt'
                with open(text_index_path, 'r', encoding='ISO-8859-1') as text:
                    sample_info["text"] = tp3.process(text.read()[:-1])

            sample_info["emotion_label"] = label_map[args.dataset_name][label_info["image_text"]]
            # sample_info["ocr"] = ocr_info["ocr"]
            sample_info["ocr"] = []
            if len(sample_info["ocr"]) == 0:
                sample_info["text_with_ocr"] = sample_info["text"] + " <ocr> "
            else:
                scene_text = " ".join(sample_info["ocr"])
                sample_info["text_with_ocr"] = sample_info["text"] + " <ocr> " + scene_text

            #path_list = [visible_train_path, blind_train_path, test_path, dev_path, visible_hard_path, blind_hard_path]
            for j in range(len(id_list)):
                if sample_info["id"] in id_list[j]:
                    content_list[j].append(sample_info)

        for index in range(len(content_list)):
            json_data = json.dumps(content_list[index], indent=2)
            with open(path_list[index], 'w') as f:
                f.write(json_data)

        print("New dasetset partition have been created successfully")

    if args.dataset_name in ["FI", "Emotion6", "UnbiasedEmo"]:

        print("Sample Dataset: ", args.dataset_name, " Following files will be created:")
        print(visible_train_path)
        print(test_path)
        print(dev_path)

        while True:
            vld = input("Enter yes/no to continue/exit:")
            if vld == 'Yes' or vld == 'yes' or vld == 'Y' or vld == 'y':
                break
            elif vld == 'No' or vld == 'no' or vld == 'N' or vld == 'n':
                exit()

        path_list = [visible_train_path, test_path, dev_path]
        content_list = [[], [], []]

        easy_set = [[] for i in range(len(label_map[args.dataset_name]))]

        for i in label_content:
            senti_index = label_map[args.dataset_name][i["label"]]
            easy_set[senti_index].append(i["id"])


        easy_set_sttts = {}
        len_easy_set = 0
        for k in label_map[args.dataset_name].keys():
            len_easy_set += len(easy_set[label_map[args.dataset_name][k]])
            easy_set_sttts[k] = len(easy_set[label_map[args.dataset_name][k]])
        print("Easy set statistics:", easy_set_sttts)

        assert len_easy_set == set_length[args.dataset_name][0]

        easy_sample_partition = {'train':set_length[args.dataset_name][1],
                                'dev':set_length[args.dataset_name][2],
                                'test':set_length[args.dataset_name][3]}
        easy_sample_partition_dict = {'train': {}, 'dev': {}, 'test': {}}

        #partition according to usage
        for k in easy_sample_partition.keys():
            num_sample_usage_k = easy_sample_partition[k]
            partition_len = {}
            current_sample_usage_k_int = 0
            current_sample_usage_k = 0
            for i in easy_set_sttts.keys():
                num_partition = num_sample_usage_k * easy_set_sttts[i] / len_easy_set
                current_sample_usage_k += num_partition
                #Last catagory
                if int(current_sample_usage_k) == num_sample_usage_k:
                    partition_len[i] = num_sample_usage_k - current_sample_usage_k_int
                else:
                    current_sample_usage_k_int += int(num_partition)
                    partition_len[i] = int(num_partition)
            easy_sample_partition_dict[k] = partition_len

        for k in easy_set_sttts.keys():
            num_label_k = 0
            for j in easy_sample_partition.keys():
                num_label_k += easy_sample_partition_dict[j][k]
            num_label_k_lack = easy_set_sttts[k] - num_label_k
            easy_sample_partition_dict['train'][k] += num_label_k_lack

        print(easy_sample_partition)
        print(easy_sample_partition_dict)


        train, test, val = [], [], []
        for label in easy_set_sttts.keys():
            senti_index = label_map[args.dataset_name][label]
            sub_train_val, sub_test = train_test_split(easy_set[senti_index], test_size=easy_sample_partition_dict['test'][label])
            sub_train, sub_val = train_test_split(sub_train_val, test_size=easy_sample_partition_dict['dev'][label])

            train.extend(sub_train)
            test.extend(sub_test)
            val.extend(sub_val)

        assert len(train) == set_length[args.dataset_name][1]
        assert len(val) == set_length[args.dataset_name][2]
        assert len(test) == set_length[args.dataset_name][3]

        id_list = [train, test, val]

        for i in tqdm(range(len(label_content))):
            sample_info = {"id": None, "label": None}
            label_info = label_content[i]
            sample_info["id"] = label_info["id"]
            sample_info["label"] = label_map[args.dataset_name][label_info["label"]]

            #path_list = [visible_train_path, blind_train_path, test_path, dev_path, visible_hard_path, blind_hard_path]
            for j in range(len(id_list)):
                if sample_info["id"] in id_list[j]:
                    content_list[j].append(sample_info)

        for index in range(len(content_list)):
            json_data = json.dumps(content_list[index], indent=2)
            with open(path_list[index], 'w') as f:
                f.write(json_data)

        print("New dasetset partition have been created successfully")

    if args.dataset_name in pretrain_dataset:

        print("Sample Dataset: ", args.dataset_name, " Following files will be created:")
        print(visible_train_path)
        print(test_path)
        print(dev_path)

        while True:
            vld = input("Enter yes/no to continue/exit:")
            if vld == 'Yes' or vld == 'yes' or vld == 'Y' or vld == 'y':
                break
            elif vld == 'No' or vld == 'no' or vld == 'N' or vld == 'n':
                exit()

        path_list = [visible_train_path, test_path, dev_path]
        content_list = [[], [], []]

        id_set = []

        for i in label_content:
            id_set.append(i["id"])

        assert len(id_set) == set_length[args.dataset_name][0]

        sample_partition = {'train':set_length[args.dataset_name][1],
                                'dev':set_length[args.dataset_name][2],
                                'test':set_length[args.dataset_name][3]}
        sample_partition_dict = {'train': {}, 'dev': {}, 'test': {}}


        train_val, test = train_test_split(id_set, test_size=sample_partition['test'])
        train, val = train_test_split(train_val, test_size=sample_partition['dev'])


        assert len(train) == set_length[args.dataset_name][1]
        assert len(val) == set_length[args.dataset_name][2]
        assert len(test) == set_length[args.dataset_name][3]

        id_list = [train, test, val]
        for i in tqdm(range(len(label_content))):
            sample_info = label_content[i]

            #path_list = [visible_train_path, blind_train_path, test_path, dev_path, visible_hard_path, blind_hard_path]
            for j in range(len(id_list)):
                if sample_info["id"] in id_list[j]:
                    content_list[j].append(sample_info)

        for index in range(len(content_list)):
            json_data = json.dumps(content_list[index], indent=2)
            with open(path_list[index], 'w') as f:
                f.write(json_data)

        print("New dasetset partition have been created successfully")

