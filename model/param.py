class ModelParam:
    def __init__(self,
                 texts=None,
                 bert_attention_mask=None,
                 images=None,
                 text_image_attention_mask=None,
                 label=None,
                 ocr=None,
                 ocr_attention_mask=None,
                 use_fact_mask=False,
                 use_senti_mask=False,
                 fact_cluster=None,
                 senti_cluster=None):
        self.for_pretrain = False
        self.texts = texts
        self.images = images
        self.text_image_attention_mask = text_image_attention_mask
        self.bert_attention_mask = bert_attention_mask
        self.label = label
        self.ocr = ocr
        self.ocr_attention_mask = ocr_attention_mask
        self.match_label = None
        self.consistency_label = None
        self.use_fact_mask = use_fact_mask
        self.use_senti_mask = use_senti_mask
        self.fact_cluster = fact_cluster
        self.senti_cluster = senti_cluster

    def set_data_param(self,
                       texts=None,
                       bert_attention_mask=None,
                       images=None,
                       text_image_attention_mask=None,
                       label=None,
                       ocr=None,
                       ocr_attention_mask=None,
                       use_fact_mask=False,
                       use_senti_mask=False,
                       fact_cluster=None,
                       senti_cluster=None):
        self.texts = texts
        self.images = images
        self.text_image_attention_mask = text_image_attention_mask
        self.bert_attention_mask = bert_attention_mask
        self.label = label
        self.ocr = ocr
        self.ocr_attention_mask = ocr_attention_mask
        self.use_fact_mask = use_fact_mask
        self.use_senti_mask = use_senti_mask
        self.fact_cluster = fact_cluster
        self.senti_cluster = senti_cluster

    def add_pretrain_param(self, match_label, consistency_label):
        self.for_pretrain = True
        self.match_label = match_label
        self.consistency_label = consistency_label