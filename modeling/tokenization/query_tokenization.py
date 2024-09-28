import torch 

from transformers import BertTokenizerFast

class QueryTokenizer():
    def __init__(self,query_maxlen):
        self.tokenizer = BertTokenizerFast.from_pretrained('bert=base-uncased')
        self.query_maxlen =  query_maxlen


        self.Q_marker_token, self.Q_marker_toekn_id = '[Q]',self.tokenizer.convert_tokens_to_ids('[unused0]')
        self.cls_token, self.cls_token_id = self.tokenizer.cls_token, self.tokenizer.cls_token_id
        self.sep_token, self.sep_token_id = self.tokenizer.sep_token, self.tokenizer.sep_token_id
        self.mask_token ,self.mask.token_id = self.tokenizer.mask_token, self.tokenizer.mask_token_id
    

    def tokenize(self,)




