#!/usr/bin/env python
# coding: utf-8

# These class functions are the adversarial attack systems for NER; if entities == True an entity attack is performed, if entities == False an entity context attack is performed. It has options for performing a Random Attack (default is set to False). 

# In[1]:


import argparse
import glob
import logging
import os
import random
import criteria
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import tensorflow_hub as hub
import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score
# from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from utils_ner import convert_examples_to_features, get_labels, read_examples_from_file

from transformers import AdamW, WarmupLinearSchedule
from transformers import WEIGHTS_NAME, BertConfig, BertForTokenClassification, BertTokenizer
# from transformers import RobertaConfig, RobertaForTokenClassification, RobertaTokenizer
import tensorflow as tf


# In[2]:


import pickle
import pandas as pd
import re

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin1')


# In[3]:


# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5" 


# In[4]:


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


# In[5]:


class AdversarialBERT(): 
    def __init__(self):
        ##can uncomment to use a cache version of the USE
#         nw_cache_path = '/data/dirksonar/NER_data/42480c3c7f42bf87d36d4c58fc4374b08dae2909/'
        self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.stop_words_set = self.get_stopwords()
        self.set_seed(1, 0)
        
    def initialize_essential(self, entities):
        if entities == True: 
            pass
            
        if entities == False: 
            ## get cos sim matrix
            path = '/data/dirksonar/TextFooler/TextFooler-master/TextFooler-master/cos_sim_counter_fitting.npy'
            self.cos_sim = np.load (path)
            
            self.idx2word = {}
            self.word2idx = {}

            pathcf = '/data/dirksonar/TextFooler/TextFooler-master/TextFooler-master/counter-fitted-vectors.txt'

            print("Building vocab...")
            with open(pathcf, 'r') as ifile:
                for line in ifile:
                    word = line.split()[0]
                    if word not in self.idx2word:
                        self.idx2word[len(self.idx2word)] = word
                        self.word2idx[word] = len(self.idx2word) - 1
    
    def load_obj(self, name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f, encoding='latin1')
    
    def save_obj(self, obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
    
    def initialize_model(self,modelpath, labelfile):
#         labellst2 = [i for j in labellst for i in j]
#         self.labels = list(set(labellst2)) 
#         print(self.labels)
        self.labels = get_labels(labelfile)
#         self.labels = ['I-location', 'I-group', 'O', 'B-creative-work', 'I-product', 'B-corporation', 'I-corporation', 'B-product', 'I-creative-work', 'B-location', 'B-group', 'I-person', 'B-person']
    
        self.num_labels = len(self.labels)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.max_seq_length = 128

        self.tokenizer = BertTokenizer.from_pretrained(modelpath, do_lower_case = False)
        self.model = BertForTokenClassification.from_pretrained(modelpath)
    
    
    def set_seed(self,num, n_gpu):
#         random.seed(num)
        np.random.seed(num)
        torch.manual_seed(num)
        if n_gpu > 0:
                torch.cuda.manual_seed_all(num)
                
    def cosine_similarity(self, v1, v2):
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        if (not mag1) or (not mag2):
            return 0
        return np.dot(v1, v2) / (mag1 * mag2)

    def test_similarity(self,text1, text2):
        vecs = self.embed([text1, text2])['outputs']
        v1 = vecs[0]
        v2 = vecs[1]
        return self.cosine_similarity(v1, v2)
    
    def get_stopwords(self):
        '''
        :return: a set of 266 stop words from nltk. eg. {'someone', 'anyhow', 'almost', 'none', 'mostly', 'around', 'being', 'fifteen', 'moreover', 'whoever', 'further', 'not', 'side', 'keep', 'does', 'regarding', 'until', 'across', 'during', 'nothing', 'of', 'we', 'eleven', 'say', 'between', 'upon', 'whole', 'in', 'nowhere', 'show', 'forty', 'hers', 'may', 'who', 'onto', 'amount', 'you', 'yours', 'his', 'than', 'it', 'last', 'up', 'ca', 'should', 'hereafter', 'others', 'would', 'an', 'all', 'if', 'otherwise', 'somehow', 'due', 'my', 'as', 'since', 'they', 'therein', 'together', 'hereupon', 'go', 'throughout', 'well', 'first', 'thence', 'yet', 'were', 'neither', 'too', 'whether', 'call', 'a', 'without', 'anyway', 'me', 'made', 'the', 'whom', 'but', 'and', 'nor', 'although', 'nine', 'whose', 'becomes', 'everywhere', 'front', 'thereby', 'both', 'will', 'move', 'every', 'whence', 'used', 'therefore', 'anyone', 'into', 'meanwhile', 'perhaps', 'became', 'same', 'something', 'very', 'where', 'besides', 'own', 'whereby', 'whither', 'quite', 'wherever', 'why', 'latter', 'down', 'she', 'sometimes', 'about', 'sometime', 'eight', 'ever', 'towards', 'however', 'noone', 'three', 'top', 'can', 'or', 'did', 'seemed', 'that', 'because', 'please', 'whereafter', 'mine', 'one', 'us', 'within', 'themselves', 'only', 'must', 'whereas', 'namely', 'really', 'yourselves', 'against', 'thus', 'thru', 'over', 'some', 'four', 'her', 'just', 'two', 'whenever', 'seeming', 'five', 'him', 'using', 'while', 'already', 'alone', 'been', 'done', 'is', 'our', 'rather', 'afterwards', 'for', 'back', 'third', 'himself', 'put', 'there', 'under', 'hereby', 'among', 'anywhere', 'at', 'twelve', 'was', 'more', 'doing', 'become', 'name', 'see', 'cannot', 'once', 'thereafter', 'ours', 'part', 'below', 'various', 'next', 'herein', 'also', 'above', 'beside', 'another', 'had', 'has', 'to', 'could', 'least', 'though', 'your', 'ten', 'many', 'other', 'from', 'get', 'which', 'with', 'latterly', 'now', 'never', 'most', 'so', 'yourself', 'amongst', 'whatever', 'whereupon', 'their', 'serious', 'make', 'seem', 'often', 'on', 'seems', 'any', 'hence', 'herself', 'myself', 'be', 'either', 'somewhere', 'before', 'twenty', 'here', 'beyond', 'this', 'else', 'nevertheless', 'its', 'he', 'except', 'when', 'again', 'thereupon', 'after', 'through', 'ourselves', 'along', 'former', 'give', 'enough', 'them', 'behind', 'itself', 'wherein', 'always', 'such', 'several', 'these', 'everyone', 'toward', 'have', 'nobody', 'elsewhere', 'empty', 'few', 'six', 'formerly', 'do', 'no', 'then', 'unless', 'what', 'how', 'even', 'i', 'indeed', 'still', 'might', 'off', 'those', 'via', 'fifty', 'each', 'out', 'less', 're', 'take', 'by', 'hundred', 'much', 'anything', 'becoming', 'am', 'everything', 'per', 'full', 'sixty', 'are', 'bottom', 'beforehand'}
        '''
        stop_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as', 'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'both',  'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn', "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere', 'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except',  'first', 'for', 'former', 'formerly', 'from', 'hadn', "hadn't",  'hasn', "hasn't",  'haven', "haven't", 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn', "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly',  'must', 'mustn', "mustn't", 'my', 'myself', 'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per', 'please','s', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow', 'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they','this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too','toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used',  've', 'was', 'wasn', "wasn't", 'we',  'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won', "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
        stop_words = set(stop_words)
        return stop_words
    
    def pos_filter(self, ori_pos, new_pos_list):
        same = [True if ori_pos == new_pos
                else False
                for new_pos in new_pos_list]
        return same

    def pick_most_similar_words_batch(self, src_words, sim_mat, idx2word, ret_count=10, threshold=0.5):
        """
        embeddings is a matrix with (d, vocab_size)
        """
        sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
        sim_words, sim_values = [], []
        for idx, src_word in enumerate(src_words):
            sim_value = sim_mat[src_word][sim_order[idx]]
            mask = sim_value >= threshold
            sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
            sim_word = [idx2word[id] for id in sim_word]
            sim_words.append(sim_word)
            sim_values.append(sim_value)
        return sim_words, sim_values

    def prepare_data_for_eval(self, examples): 
        features = convert_examples_to_features(examples, self.labels, self.max_seq_length, self.tokenizer,
                                                    # xlnet has a cls token at the end
                                                    cls_token=self.tokenizer.cls_token,
                                                    cls_token_segment_id=0,
                                                    sep_token=self.tokenizer.sep_token,
                                                    pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                                                    pad_token_segment_id= 0,
                                                    pad_token_label_id=self.pad_token_label_id
                                                    )
        
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset

    def read_examples_from_list(self, data, mode): ## this does not incorporate labels but just adds 'O's
        guid_index = 0
        examples = []
        for s in data: #data is a list
            words = []
            labels = []
            guid_index += 1
            for num, w in enumerate(s): 
                words.append(w)
                labels.append("O")

#             print(len(words))
#             print(words)

            examples.append(InputExample(guid="%s-%d".format(mode, guid_index),
                                            words=words,
                                             labels=labels))
        return examples


    def read_examples_from_lists_wlables(self, data, labels, mode): ## this does not incorporate labels but just adds 'O's
        guid_index = 0
        examples = []
        for s, l in zip(data, labels): #data is a list
            words = []
            labels = []
            guid_index += 1
            for num, w in enumerate(s): 
                words.append(w)
                labels.append(l[num])

            examples.append(InputExample(guid="%s-%d".format(mode, guid_index),
                                            words=words,
                                             labels=labels))
        return examples

    def evaluate(self, eval_dataset, model, tokenizer, labels, pad_token_label_id, mode = 'test', prefix=""):
        eval_sampler = SequentialSampler(eval_dataset) 
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=32)

        # Eval!
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
    #         batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                          "attention_mask": batch[1],
                          "token_type_ids": batch[2],
                          # XLM and RoBERTa don"t use segment_ids
                          "labels": batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

    #             if args.n_gpu > 1:
    #                 tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

                eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        probs = preds
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(labels)}
        
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]
        old_preds_list = preds_list

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        results = {
            "loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list)
        }

    #     logger.info("***** Eval results %s *****", prefix)
    #     for key in sorted(results.keys()):
    #         logger.info("  %s = %s", key, str(results[key]))

        return results, preds_list, probs, preds, old_preds_list, out_label_list, label_map
    

    def get_entities (self, pred_tags): 
        ent_ids = []
        ent_tags = []
        for num, i in enumerate (pred_tags):
            temp = []
            temp2 = []
            if i.startswith('B-'):
                temp.append(num)
                temp2.append(i)
                for x in range(1, 20): 
                    try: 

                        if pred_tags[num+x].startswith ('I-'): 
                            temp.append(num+x)
                            temp2.append(pred_tags[num+x])
                        else: 
                            break 
                    except IndexError: 
                        break
                ent_ids.append(temp)
                ent_tags.append(temp2)
            else: 
                pass
        return ent_ids, ent_tags
    
    def calcwordscore(self, orig_label, orig_prob, leave_1_probs, nw_labels, ix2): 
        w =  (orig_prob.max() - leave_1_probs[ix2][orig_prob.argmax()] + (int(nw_labels[ix2] != orig_label) * (
                        leave_1_probs[ix2].max() - orig_prob[leave_1_probs[ix2].argmax()])))
#         if nw_labels[ix2] != orig_label: 
#             print(nw_labels[ix2])
        return w

    def calcwordscore_Itag(self, orig_label, orig_prob, leave_1_probs, nw_labels, ix2):
        w =  (orig_prob.max() - leave_1_probs[ix2][orig_prob.argmax()] + (int(nw_labels[ix2][2:] != orig_label[2:]) * (
                        leave_1_probs[ix2].max() - orig_prob[leave_1_probs[ix2].argmax()])))
        return w
    
    def calculate_importance_scores (self, ent, preds_listnw, probsnw, leave_1_texts, sent_allowed, allowed_ids): 
        mean_import_scores = []
        all_import_scores = []

        for ix in ent: ##for each word in entity we calculate importance of words. ix is hte index of hte word in the sentence
            orig_label = preds_listnw[0][ix]
            orig_prob = probsnw[0][ix+1]
            leave_1_probs= []
            nw_labels =[]
            for i in range(1, len(leave_1_texts)): 
                [leave_1_probs.append(probsnw[i][ix+1])]##new probabilities of this word ##need to add 1 to compensate for padded token
                try: 
                    [nw_labels.append (preds_listnw[i][ix])] ##new lables of this word
                except IndexError:
                    return 0
            import_scores = []
            nw_leave_1_texts = leave_1_texts[1:]
            for ix2, word in enumerate(nw_leave_1_texts):   
                if orig_label.startswith('I'): 
                    wordscore = self.calcwordscore_Itag(orig_label, orig_prob, leave_1_probs, nw_labels, ix2)
                else: 
                    wordscore = self.calcwordscore(orig_label, orig_prob, leave_1_probs, nw_labels, ix2)
                ## the last part says what the probability of the new label (if there is one) without hte word minus that same prob WITH the word
                import_scores.append(wordscore)
            all_import_scores.append(import_scores) ##list of lists
        sum_import_scores = np.sum(all_import_scores, axis =0)
        import_score_threshold=-1
        words_perturb = []
        for idx, score in sorted(enumerate(sum_import_scores), key=lambda x: x[1], reverse=True):

            if score > import_score_threshold and sent_allowed[idx] not in self.stop_words_set:
                words_perturb.append((allowed_ids[idx], sent_allowed[idx]))

#         print(words_perturb)
        return words_perturb

    def random_ranking(self, ent, leave_1_texts, sent_allowed, allowed_ids): 
        words_perturb = []
        nw_leave_1_texts = leave_1_texts[1:]
        for idx, word in enumerate(nw_leave_1_texts): 
            if sent_allowed[idx] not in self.stop_words_set:
                words_perturb.append((allowed_ids[idx], sent_allowed[idx]))
        random.seed(1)
        random.shuffle(words_perturb)
#         print(words_perturb)
        return words_perturb  
    
    def get_adversarial_examples_per_sent(self, sent, origlbl, predlbl, ent_tags, ent_ids, random_attack = False, sim_synonyms=0.5, sim_score_threshold=0.8, import_score_threshold = -1, sim_predictor = None, synonym_num=50,batch_size=32):   
        if sim_predictor == None: 
            sim_predictor == self.embed
        
        out_texts = []
#         ent_ids, ent_tags = get_entities(origlbl)
        taboo_ids = [i for j in ent_ids for i in j]
        len_text = len(sent)
        sent_allowed = [i for num, i in enumerate(sent) if num not in taboo_ids]
        allowed_ids = [num for num, i in enumerate(sent) if num not in taboo_ids]
#         print(sent_allowed)
        pos_ls = criteria.get_pos(sent)

        for entnum, ent in enumerate(ent_ids): ##for each entity
            success= 0
#             print(ent)
#             print(sent)
#             print(origlbl)
            
            
            leave_1_texts = []
            leave_1_texts.append(sent)
            for idword, word in enumerate(sent): 
                if idword not in taboo_ids:
    #                 print(text_ls)
                    a_text = sent[:idword] + ['<oov>'] + sent[min(idword+1, len_text):] #until the end of sentence or id + 1
                    leave_1_texts.append(a_text)

            ##need to prepare for eval 
            examples = self.read_examples_from_list(leave_1_texts, 'test')
            dataset = self.prepare_data_for_eval(examples)

            resultsnw, preds_listnw, probsnw, predsnw, old_preds_listnw, true_labelsnw, label_map2= self.evaluate(dataset, self.model, self.tokenizer, self.labels, self.pad_token_label_id)
            #this gives predictions for all words but we need to pick out hte relevant ones
#             print(preds_listnw[0])
#             print(preds_listnw[0] == predlbl)
            mean_import_scores = []
            all_import_scores = []
            
            if random_attack == False:
                words_perturb = self.calculate_importance_scores (ent, preds_listnw, probsnw, leave_1_texts, sent_allowed, allowed_ids)
            else: 
                words_perturb = self.random_ranking(ent, leave_1_texts, sent_allowed, allowed_ids)
#             print(words_perturb)
            
            if words_perturb == 0: ##there are none
                out = 'This sentence was a problem'
                out_texts.append(out)
            else:
                # find synonyms
                words_perturb_idx = [self.word2idx[word] for idx, word in words_perturb if word in self.word2idx]
                synonym_words, _ = self.pick_most_similar_words_batch(words_perturb_idx, self.cos_sim, self.idx2word, synonym_num, sim_synonyms)
                synonyms_all = []
                for idx, word in words_perturb:
                    if word in self.word2idx:
                        synonyms = synonym_words.pop(0)
                        if synonyms:
                            synonyms_all.append((idx, synonyms))

                # start replacing and attacking until label changes
                text_prime = sent[:]
                text_cache = text_prime[:]
                num_changed = 0
                unchanged_ent = ent
    #             print(len(synonyms_all))
    
                if len(synonyms_all) == 0: 
                    out2 = 'No synonyms'
                    out_texts.append(out2)
                    break
                    
                for idx, synonyms in synonyms_all:
    #                 print(len(synonyms))
    #                 finalsim = self.test_similarity(' '.join(text_cache),' '.join(text_prime))
    #                 print(finalsim)
                    new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
    #                 print(len(new_texts))
                    new_texts.append(sent)
                    examples = self.read_examples_from_list(new_texts, 'test')# 
    #                 print(new_texts)
                    dataset = self.prepare_data_for_eval(examples)

                    results2, preds_list2, probs2, preds2, old_preds_list2, true_labels2, label_map2nw= self.evaluate(dataset,self.model, self.tokenizer, self.labels, self.pad_token_label_id)           

                    # compute semantic similarity
                    semantic_sims = []

                    for i in new_texts: 
    #                     print(i)
    #                     print(text_cache)
                        sim = self.test_similarity(' '.join(text_cache),' '.join(i))
    #                     print(sim)
                        semantic_sims.append(sim)
    #                 print(semantic_sims)

                    if len(ent) == 1: # it is a single word entity
    #                     print('Branch of the single word')
                        rel_probs=[]
                        new_probs_mask=[]
                        ix = ent[0]
                        correct = probs2[-1][ix+1].argmax()
                        for i in range(0, (len(new_texts)-1)): 
                            [rel_probs.append(probs2[i][ix+1])]
                        new_texts2 = new_texts[:-1]
                        for ix2, t in enumerate(new_texts2):
                            r = rel_probs[ix2]
                            new_probs_mask.append(correct != r.argmax())    
                        new_probs_mask2 = np.array(new_probs_mask)

                        semsims = semantic_sims[:-1]
                        semantic_sims2 = np.array(semsims)

                        synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                                                           if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts2]

                        pos_ls = criteria.get_pos(sent)

                        new_probs_mask2 *= (semantic_sims2 >= sim_score_threshold)
    #                     print(new_probs_mask2)
                        pos_mask = np.array(self.pos_filter(pos_ls[idx], synonyms_pos_ls))
                        new_probs_mask2 *= pos_mask

                        if np.sum(new_probs_mask2) > 0: ##there is an instance that changes the label
                            z = (new_probs_mask2 * semantic_sims2).argmax()
                            if semantic_sims2[z] < sim_score_threshold:
    #                             print('This is below the threshold')
                                break
                            text_prime[idx] = synonyms[z]
                            num_changed += 1
    #                         print(text_prime)
    #                         print('We will stop')
                            success = 1
                            success_p = success/1
                            break
                        else: ##no label change at all - But if not, then we select the word with the least confidence score of label y as the best replacement word for wi
                            nw_rel_probs = [i[correct] for i in rel_probs]

                            nw_rel_probs2 = nw_rel_probs + (semantic_sims2 < sim_score_threshold) + (1 - pos_mask)

                            new_label_prob_min = nw_rel_probs2.min()
                            new_label_prob_argmin = nw_rel_probs2.argmin()
                            if semantic_sims2[new_label_prob_argmin] < sim_score_threshold:
    #                             print('This is below the threshold')
                                break

                            orig_prob = probs2[-1][ix+1].max()
                            if new_label_prob_min < orig_prob:
                                text_prime[idx] = synonyms[new_label_prob_argmin]
    #                             print(text_prime)
    #                             print(synonyms[new_label_prob_argmin])
                                num_changed += 1
    #                             print('We will go on')
    #                         else:
    #                             print('Probs were not high enough')


                    else: #it is a multiword entity   -- unchanged ent is the part of hte entity that has not been changed.                 
    #                     print('Branch of the multiword')
                        correctlbl = []
                        correctlblname = []
    #                     print('the unchanged ents are now:')
    #                     print(unchanged_ent)
                        for ix in unchanged_ent: 
                            correctlbl.append(probs2[-1][ix+1].argmax())
                            correctlblname.append(preds_list2[-1][ix])
    #                     print(correctlblname)

                        new_probs_mask_temp=[]
                        all_rel_probs = []
                        for number, ix in enumerate(unchanged_ent):
                            rel_probs=[]
                            new_probs_mask=[]
                            for i in range(0, (len(new_texts)-1)): 
                                [rel_probs.append(probs2[i][ix+1])]
                            all_rel_probs.append(rel_probs)
                            correct = probs2[-1][ix+1].argmax()
                            new_texts2 = new_texts[:-1]
                            c = correctlblname[number]
                            if c.startswith('I'): 
    #                             print(c)
                                for ix2, t in enumerate(new_texts2): 
                                    pl = preds_list2[ix2][ix] ##predicted label
                                    if pl[2:] == c[2:]: ##if the predicted label is either the B or I version of the originla label = no label change 
                                        new_probs_mask.append(False)
                                    else:
                                        new_probs_mask.append(True)                            
                            else: 
                                for ix2, t in enumerate(new_texts2): 
                                    r = rel_probs[ix2]
                                    new_probs_mask.append(correct != r.argmax()) ## true is label has been changed.

                            new_probs_mask_temp.append(np.array(new_probs_mask))

                        m = np.matrix(new_probs_mask_temp)


                        unchanged_lbls = [] ## contains hte indexes of hte words changed in the entity- not the ix in the sentence
    #                     print(len(new_texts))

                        if len(new_texts) > 0: ## there are some synonyms 
    #                         print(len(new_texts))
    #                         print(m)
                            for i in range(0, (len(new_texts)-1)):
    #                             print(i)
                                try: 
                                    z = np.argwhere(m[:,i] == 0)[:,0]
                                    unchanged_lbls.append(z.flatten().tolist()) 
                                except IndexError: 
                                    print(m)
                                    z = []
                                    unchanged_lbls.append(z)

    #                         print('the unchanged labels for each synonym are:') 
    #                         print(unchanged_lbls)

                        new_probs_mask2 = np.sum(new_probs_mask_temp,axis = 0) ##true if any of the entity words have changed label with higher number for more label changes. 

                        semsims = semantic_sims[:-1]
                        semantic_sims2 = np.array(semsims)
                        synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                                                           if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts2]
                        pos_ls = criteria.get_pos(sent)
                        pos_mask = np.array(self.pos_filter(pos_ls[idx], synonyms_pos_ls))

                        new_probs_mask2 *= (semantic_sims2 >= sim_score_threshold)
                        new_probs_mask2 *= pos_mask

                        ## are there any that change all labels? MAKE A MASK
                        entmax = len(unchanged_ent)
    #                     w = np.argwhere(new_probs_mask2 == entmax)
    #                     lstw = w.flatten().tolist()
    #                     print('entmax: ') 
    #                     print(entmax)
                        new_probs_mask_all = [1 if x == entmax else 0 for x in new_probs_mask2]
    #                     print(new_probs_mask_all)
    #                     print(new_probs_mask2)
    #                     print(np.sum(new_probs_mask2))

                        if np.sum(new_probs_mask_all) > 0: ## there are synonyms that can change all the labels! BEST SCENARIO
                            z= (new_probs_mask_all * semantic_sims2).argmax()
                            if semantic_sims2[z] < sim_score_threshold:
    #                             print('This is below the threshold')
                                break
                            text_prime[idx] = synonyms[z]
                            num_changed += 1
    #                         print(text_prime)                          
    #                         print('We will stop')
                            success = 1
                            success_p = success/1
                            break

                        elif np.sum(new_probs_mask2) > 0: ##there is an instance that changes the label
                            winners = np.argwhere(new_probs_mask2 == np.amax(new_probs_mask2))
    #                         print('some have changed')
                            lstwinners = winners.flatten().tolist() ##the ix of ones that make the highest number of words change but also conform to filters

                            winner_probs = []

                            for a in lstwinners:
                                u = unchanged_lbls[a]

                                ##retrieve correct prob of unchanged labels to compare
                                rp = [i for num, i in enumerate(all_rel_probs) if num in u] ##ones for right entities

                                rp2 = [i[a] for i in rp] #get the ones for this synonym

                                cor = [i for num, i in enumerate(correctlbl) if num in u] ##ix of correct labels for relevant entities

                                rp_out = []
                                for a,b in zip(cor, rp2):
                                    rp_out.append(np.array(b[a]))

                                rp_out2 = np.array(rp_out)
                                rp_sum = np.sum(rp_out2,axis =0) #summed confidence of all the relevant words in the entity
                                winner_probs.append(rp_sum)

                            ##choose the lowest winner prob - lowest for other entities
                            winner_probs2 = np.array(winner_probs)

                            new_label_prob_min = winner_probs2.min()
                            new_label_prob_argmin = winner_probs2.argmin()

                            winning_ix = lstwinners[new_label_prob_argmin]

                            ##change the unchanged ent for next iteration
                            u = unchanged_lbls[winning_ix]
                            nw_unchanged_ent = [i for num,i in enumerate(unchanged_ent) if num in u] ##actual indexes in sentences
                            unchanged_ent = nw_unchanged_ent
    #                         print(unchanged_ent)

                            if semantic_sims2[winning_ix] < sim_score_threshold:
                                print('This is below the threshold')
                                break

    #                         for ix in unchanged_ent: 
    #                             o = [probs2[-1][ix+1].max()]
    #                             o2 = np.sum(np.array(o)) 

    #                         if new_label_prob_min < o2:
                            text_prime[idx] = synonyms[winning_ix]
    #                             print(text_prime)
                            num_changed += 1
    #                             print('We have changed some but will continue')
                            success += 1
                            success_p = success/len(ent)
                            print(success_p)
                            text_cache2 = text_prime
                            if success_p == 1:
                                break


                        else: ##no label change at all - But if not, then we select the word with the least confidence score of label y as the best replacement word for wi
    #                         print('no label change')
                            rp_out = []
                            for a,b in zip(correctlbl, all_rel_probs): #correct is the correct labels and all rel probs is the probabilities for hte wrods in the entities
                                rp = [x[a] for x in b] 
                                rp_out.append(np.array(rp))
                            rp_sum = np.sum(rp_out,axis =0) #summed confidence of all the words in the entity
                            nw_rel_probs2 = rp_sum + (semantic_sims2 < sim_score_threshold) + (1 - pos_mask)

                            new_label_prob_min = nw_rel_probs2.min()
                            new_label_prob_argmin = nw_rel_probs2.argmin()
    #                         print(semantic_sims2[new_label_prob_argmin])
    #                         print(semantic_sims2.max())

                            if semantic_sims2[new_label_prob_argmin] < sim_score_threshold:
    #                             print('This is below the threshold')
                                break
                            ##get orig prob 
                            for ix in unchanged_ent: 
                                o = [probs2[-1][ix+1].max()]
                                o2 = np.sum(np.array(o)) 

                            if new_label_prob_min < o2:
                                text_prime[idx] = synonyms[new_label_prob_argmin]
                                num_changed += 1


                ##calculate semantic sim

                try: 
                    if 0<success_p<1: 
    #                     print('We are reverting!')
    #                     print(text_prime)
                        text_prime= text_cache2
    #                     print(text_cache2)
                except UnboundLocalError: 
                    pass

                finalsim = self.test_similarity(' '.join(text_cache),' '.join(text_prime))

                max_possible = len(synonyms_all)
                if success == 0: 
                    success_p = 0
                out_texts.append(tuple([text_prime, num_changed, max_possible, success_p, finalsim]))

        return out_texts
    
    def retrieve_entities(self,devdata, traindata,testdata): 
        ##first make a list of entities possible
        alldata = pd.concat([devdata, traindata,testdata])
        words = list(alldata['words'])
        tags = list(alldata['ner'])

        flattags = [i for j in tags for i in j]
        flattags2 = [i[2:] for i in flattags]
        tagset = set(flattags2)
        taglist = [i for i in list(tagset) if i != '' and i != 'MISC']
        print(taglist)
        allent = []
        ent_ids = []
        ent_tags = []
        for i in tags: 
            e1, e2 = self.get_entities(i)
            ent_ids.append(e1)
            ent_tags.append(e2)

        for i in taglist: ##for each entity 
            temp = []
    #         print(i)
            for a,b,c,d in zip(tags, words, ent_ids, ent_tags):
                for num, t in enumerate(d): 
                    if t[0][2:] == i: ##is it hte correct entity
                        l = []
                        ix = c[num] ##get the ids
                        [l.append(b[j]) for j in ix]
                        temp.append(l)
            allent.append(temp)

        return taglist, allent
    
#     def test_get_adversarial_sent_entities (self, sent, origlbl, predlbl, ent_tags, ent_ids, taglst, allent, num_sample=50, sim_score_threshold=0.8, sim_predictor=None, batch_size=32): 
#         if sim_predictor == None: 
#             sim_predictor == self.embed
#         out_texts = []
# #         ent_idsold, ent_tagsold = get_entities(origlbl)
# #         random.seed(1)
#         len_text= len(sent)
# #         print(ent_ids)
#         for entnum, ent in enumerate(ent_ids):
# #             print(ent)
#             e = ent_tags[entnum]
#             e1 = e[0][2:]
# #             print(e1)
#             try: 
#                 corix = taglst.index(e1)
#                 correct_entlst = allent[corix]
            
#                 bcor = 'B-' + e1
#                 icor = 'I-' + e1
# #                 print(bcor)
    
#                 start = ent[0]
#                 end = ent[-1]
#                 idx = start

#                 sam = random.sample(correct_entlst, num_sample)
#                 print(sam)
#             except ValueError: 
#                 pass
    
    def get_adversarial_sent_entities (self, sent, origlbl, predlbl, ent_tags, ent_ids, taglst, allent, num_sample=50, sim_score_threshold=0.8, sim_predictor=None, batch_size=32): 
        if sim_predictor == None: 
            sim_predictor == self.embed
        out_texts = []
#         ent_idsold, ent_tagsold = get_entities(origlbl)
#         random.seed(1)
        len_text= len(sent)
#         print(ent_ids)
        for entnum, ent in enumerate(ent_ids):
#             print(ent)
            e = ent_tags[entnum]
            e1 = e[0][2:]
#             print(e1)
            try: 
                corix = taglst.index(e1)
                correct_entlst = allent[corix]
            
                bcor = 'B-' + e1
                icor = 'I-' + e1
#                 print(bcor)
    
                start = ent[0]
                end = ent[-1]
                idx = start

                sam = random.sample(correct_entlst, num_sample)
                
                comparewith = []
                entidx = []
                for syn in sam:
                    cw = [bcor]
                    if len(syn) > 1:
                        [cw.append(icor) for i in range(1, len(syn))]
                    comparewith.append(cw)
                    ei = [ent[0]]
                    if len(syn) > 1: 
                        [ei.append(ent[0] + i) for i in range(1, len(syn))]
                    entidx.append(ei)
#                 print(sam)
                
                text_prime = sent[:]
                text_cache = text_prime[:]
    #             print(sam)

                new_texts = [text_prime[:start] + syn + text_prime[end+1:] for syn in sam]
#                 print(new_texts)
                
                examples = self.read_examples_from_list(new_texts, 'test')# 
#                 print(new_texts)
                dataset = self.prepare_data_for_eval(examples)

                results2, preds_list2, probs2, preds2, old_preds_list2, true_labels2, label_map2nw= self.evaluate(dataset,self.model, self.tokenizer, self.labels, self.pad_token_label_id)           
                    
                # compute semantic similarity
                semantic_sims = []
                for i in new_texts: 
                    sim = self.test_similarity(' '.join(text_cache),' '.join(i))
                    semantic_sims.append(sim)
                new_probs_mask = []   
                ##make a mask of if anyone managed
                              
                for a, b, c  in zip(comparewith, entidx, preds_list2):
                    temp = []
                    try: 
                        z = [c[i] for i in b] # the prediction
                    except IndexError: 
                        print(comparewith)
                        print(entidx)
                        print(preds_list2)
                        break
                        
                    for num, j in enumerate(z): ## j is hte predicted label and a is the GOOD label
                        if j == a[num]: ##if the label did not change 
                            temp.append(0) 
                        if a[num] == icor and j == bcor: ##if the old lable was I and the new one is B
                            temp.append(0)
                        else: 
                            temp.append(1) ## this word did change

                        t = np.sum(temp)/len(temp)
                    new_probs_mask.append(t) ##relative amount of entity that was changed.

                new_probs_mask2 = np.array(new_probs_mask)
                semantic_sims2 = np.array(semantic_sims)
                #mask if similarity is too low 
                new_probs_mask2 *= (semantic_sims2 >= sim_score_threshold)

                ##did anyone manage? if yes output this.
                if np.sum(new_probs_mask2) > 0: ##there is an instance that changes at least a part of a label
                    success = np.amax(new_probs_mask2)

                    new_probs_mask3 = [1 if x == success else 0 for x in new_probs_mask2]

                    chosen = sam[(new_probs_mask3 * semantic_sims2).argmax()] 
#                     print(chosen)
                    text_done = text_prime[:start] + chosen + text_prime[end+1:]
                    success = 1
#                     print(text_done)
#                     print(text_cache)
                    finalsim = self.test_similarity(' '.join(text_cache),' '.join(text_done))
                    out_texts.append(tuple([text_done, success, finalsim]))

                else: 
                    success = 0  ##if not, output this.
                    chosen = random.choice(sam)
                    text_done = text_prime[:start] + chosen + text_prime[end+1:]
                    finalsim = self.test_similarity(' '.join(text_cache),' '.join(text_done))
                    out_texts.append(tuple([text_done, success, finalsim]))
            except ValueError:#happens when a MISC value occurs
#                 print('the issue is this')
#                 out_texts.append('There is a ValueError here')
                pass
        return out_texts   
    
    def first_prediction(self, sents, true_labels): 
        examples = self.read_examples_from_lists_wlables(sents, true_labels, 'test')
        dataset = self.prepare_data_for_eval(examples)
        results, preds_list, probs, preds, old_preds_list, true_labelsx, label_map= self.evaluate(dataset, self.model, self.tokenizer, self.labels, self.pad_token_label_id) 
        return results, preds_list, probs, preds, old_preds_list, true_labelsx, label_map
    
    
    def identify_unfit_sents (self, true_labels, sents): 
        only_ent = []
        no_ent = []
        pat = '[A-Za-z]+'
        for num, origlbl in enumerate(true_labels): 
            ent_ids, ent_tags = self.get_entities(origlbl)
            taboo_ids = [i for j in ent_ids for i in j]
            sent = sents[num]
            len_text = len(sent)
            nw_txt = [i for num,i in enumerate(sent) if num not in taboo_ids]
            nw_txt2 = [i.lower() for i in nw_txt]
            filt_txt= [i for i in nw_txt2 if i not in self.stop_words_set]
            filt_txt2 = [i for i in filt_txt if re.match(pat, i)]
            if len(taboo_ids) == 0: 
                no_ent.append(1)
                only_ent.append(0)
            else: 
                no_ent.append(0)
                if len(taboo_ids) == len_text: 
                    only_ent.append(1)
                elif len(filt_txt2) == 0:
                    only_ent.append(1)
                else: 
                    only_ent.append(0)

        return only_ent, no_ent
    
    def filter_sentences_correctness (self, true_labels, pred_labels): 
        correct_ent_ids = []
        correct_ent_tags = []
        only_wrong = []

        for a,b in zip(true_labels, pred_labels): 
            a2ids, a2tags = self.get_entities(a) ##correct entities
            c = 0 ##counter for how many correct
            nw_entids= []
            nw_enttags= []
            for ent, enttag in zip(a2ids, a2tags): ##per correct entity check 
                c2 = 0 
                start = ent[0]
                end = ent[-1]
                if end == start: 
                    b2 = b[start:(start+1)]
                else: 
                    b2 = b[start:(end+1)]
                if b2 == enttag: ##definitely correct if the same
                    c =+1 
                    c2 =+ 1
                else: 
                    btag = enttag[0]
                    s = set(b2)
                    if btag in s: #if the B-tag is correct then it is not completely wrong / missing
                        c =+ 1
                        c2 =+ 1
                    else: 
                        pass
                if c2 > 0: 
                    nw_entids.append(ent)
                    nw_enttags.append(enttag)

            if c == 0:  ##no correct at all
                only_wrong.append(1) ##there is no correct entity 
#                 correct_ent_ids.append(None)
            else: 
                only_wrong.append(0) ##there is at least one correct entity
            correct_ent_ids.append(nw_entids)
            correct_ent_tags.append(nw_enttags)

        return only_wrong, correct_ent_ids, correct_ent_tags
    
    def filter_unfit(self,true_labels, sents, ids, only_ent, no_ent): 
        remove = []
        ##remove unfit
        for a,b in zip(only_ent, no_ent): 
            if a==1 or b==1: 
                remove.append(1)
            else:
                remove.append(0)
        
        nw_sents = [i for num, i in enumerate(sents) if remove[num] == 0]
        nw_true_labels = [i for num, i in enumerate(true_labels) if remove[num] == 0]
        nw_ids = [i for num, i in enumerate(ids) if remove[num] == 0]
        
        return nw_sents, nw_true_labels, nw_ids
    
    def filter_unfit_r2 (self, true_labels, sents, ids, correct_ent_ids, correct_ent_tags, preds_list, only_wrong): 
        nw_sents = [i for num, i in enumerate(sents) if only_wrong[num] == 0]
        nw_true_labels = [i for num, i in enumerate(true_labels) if only_wrong[num] == 0]
        nw_ids = [i for num, i in enumerate(ids) if only_wrong[num] == 0]
        nw_cor_ent_ids = [i for num, i in enumerate(correct_ent_ids) if only_wrong[num] == 0]
        nw_cor_ent_tags = [i for num, i in enumerate(correct_ent_tags) if only_wrong[num] == 0 ]
        nw_preds_list = [i for num, i in enumerate(preds_list) if only_wrong[num] == 0 ] 
        
        return nw_true_labels, nw_sents, nw_ids, nw_cor_ent_ids, nw_cor_ent_tags, nw_preds_list
        
    
    def main(self, data, modelpath, savepath, savepath2, labelfile='labels.txt', devdata= None, traindata= None, alltestdata=None, random_attack = False, make_first_prediction= False, entities = False, sim_synonyms=0.5, sim_score_threshold= 0.8): 
#         print(sim_score_threshold)
        self.initialize_essential(entities)
        already_fully_wrong = [] #no need to break
        not_fit_for_adversarial = [] # no words besides entities and stopwords 

        self.initialize_model(modelpath, labelfile)
        
        ##get sents and labels -- input data needs to be a aggregated df (grouped by id)
        sents = list(data['words'])
        true_labels = list(data['ner'])
        try: 
            ids = list(data['id'])
        except KeyError: 
            ids = list(data.index.values)
        
        ##first filter for if sentences either miss entities or miss changeable words - there is nothing to change
        if entities == False: 
            only_ent, no_ent = self.identify_unfit_sents (true_labels, sents)

            only_ent_ids = [i for num, i in enumerate(ids) if only_ent[num] == 1]
            no_ent_ids = [i for num, i in enumerate(ids) if no_ent[num] == 1]

            nw_sents, nw_true_labels, nw_ids = self.filter_unfit (true_labels, sents, ids, only_ent, no_ent)
    #         print(nw_sents[0])
        else:
            nw_sents = sents
            nw_true_labels = true_labels
            nw_ids = ids
            only_ent_ids =[]
            no_ent_ids = []
            
        #make or cache first_prediction
        if make_first_prediction== True: 
            print(self.labels)
            results, preds_list, probs, preds, old_preds_list, true_labelsx, label_map = self.first_prediction(nw_sents, nw_true_labels)
            self.save_obj(preds_list,'first_preds_list')
            self.save_obj(probs,'first_probs')
        else: 
            preds_list = self.load_obj('first_preds_list')
            probs = self.load_obj('first_probs')
            
        if entities == False:
            sp1 = savepath2 + 'first_results_context'
            save_obj(results, sp1)
            sp2 = savepath2 + 'first_predictions_context'
            save_obj(preds_list, sp2)
            sp3 = savepath2 + 'first_probs_context'
            save_obj(probs, sp3)
            sp4 = savepath2 + 'only_ents'
            save_obj(only_ent_ids, sp4)
        
        if entities == True:
            sp1 = savepath2 + 'first_results_entities'
            save_obj(results, sp1)
            sp2 = savepath2 + 'first_predictions_entities'
            save_obj(preds_list, sp2)
            sp3 = savepath2 + 'first_probs_entities'
            save_obj(probs, sp3)
        
        ##second filter -do not need to change if they are already wrong (if some of entities are wrong only those are changed)
        
        only_wrong, correct_ent_ids, correct_ent_tags = self.filter_sentences_correctness (nw_true_labels, preds_list)
         
        only_wrong_ids = [i for num, i in enumerate(nw_ids) if only_wrong[num] == 1]
        
        nwer_true_labels, nwer_sents, nwer_ids, correct_ent_ids, correct_ent_tags, nw_preds_list = self.filter_unfit_r2 (nw_true_labels, nw_sents, nw_ids, correct_ent_ids, correct_ent_tags, preds_list, only_wrong)
        
#         print(len(nw_preds_list))

        if entities == True: 
            sp1 = savepath2 + 'only_wrong_ids'
            save_obj(only_wrong_ids, sp1)
        
        out_texts = []
        
        if entities == False: 
            if random_attack == False:
                for num, sent in enumerate(nwer_sents):
                        
                        print(sent)
                        ##initialize
                        print(num +1)
                        print(len(nwer_sents))
                        predlbl = nw_preds_list[num]
                        origlbl = nwer_true_labels[num]
                        ent_tags = correct_ent_tags[num]
                        ent_ids = correct_ent_ids[num]

                        ##run
                        out_text = self.get_adversarial_examples_per_sent(sent, origlbl, predlbl, ent_tags, ent_ids, sim_synonyms = 0.5, sim_score_threshold = 0.8) 
#                         print(out_text)
                        out_texts.append(out_text)

                        save_obj(out_texts,savepath)
            else:
                for num, sent in enumerate(nwer_sents):
                     ##initialize
                    print(num + 1)
                    print(len(nwer_sents))
                    predlbl = nw_preds_list[num]
                    origlbl = nwer_true_labels[num]
                    ent_tags = correct_ent_tags[num]
                    ent_ids = correct_ent_ids[num]
                    print(ent_tags)
                    
                   
                    ##run 
                    out_text = self.get_adversarial_examples_per_sent(sent, origlbl, predlbl, ent_tags, ent_ids, random_attack = True, sim_synonyms = sim_synonyms, sim_score_threshold = sim_score_threshold) 
                    out_texts.append(out_text)        
                    save_obj(out_texts,savepath)
                                       
            
        if entities == True:
            taglst, allent = self.retrieve_entities(devdata, traindata, alltestdata)
            for num, sent in enumerate(nwer_sents):
                 ##initialize
                print(num + 1)
#                 if num > 10: 
#                     break
                print(sent)
                predlbl = nw_preds_list[num]
                origlbl = nwer_true_labels[num]
                ent_tags = correct_ent_tags[num]
                ent_ids = correct_ent_ids[num]
                
                ##run
                out_text = self.get_adversarial_sent_entities (sent, origlbl, predlbl, ent_tags, ent_ids, taglst, allent, sim_score_threshold=sim_score_threshold) 
                out_texts.append(out_text)
                save_obj(out_texts,savepath)
        if entities == None: 
#             for num, sent in enumerate(nwer_sents):
#             out_texts = []
            return only_ent_ids, no_ent_ids, only_wrong_ids, nwer_sents, results

        return only_ent_ids, no_ent_ids, only_wrong_ids, out_texts


# # EXAMPLE USAGE 

# In[11]:


##loading data 

path = '/data/WNUT_AdversarialSample.tsv'
test = pd.read_csv(path, sep = '\t')

devdata= pd.read_csv('/data/WNUT_devdata.tsv', sep = '\t')
traindata = pd.read_csv('/data/WNUT_traindata.tsv', sep = '\t')
testdata = pd.read_csv('/data/WNUT_testdata.tsv', sep = '\t')


# In[8]:


# EXAMPLE OF ENTITY ATTACK -- entities argument set to True

modelpath = '/NER_data/WNUT/bert_e4_lr5/'

##The first save path is for the adversarial sentences 
savepath = '/NER_data/WNUT/BERT1/output_bert1_entity'

##The second save path is other output
savepath2 = '/NER_data/WNUT/BERT1/'

only_ent_ids, no_ent_ids, only_wrong_ids, out_texts = AdversarialBERT().main(test, modelpath,savepath, savepath2, 'labels.txt', devdata2, traindata2, testdata2, random_attack =False, make_first_prediction= True, entities = True)

##only ent ids are those sentences with only entities and no other words -- not relevant for entity attack 
## no ent ids are those sentences without any entities - EXCLUDED FROM ENTITY ATTACK
## only wrong ids are those sentences with only wrong predictions of entities - EXCLUDED FROM ENTITY ATTACK 

## out texts are the provided adversarial sentences.


# In[ ]:


# EXAMPLE CONTEXT ATTACK -- entities argument set to False

modelpath = '/NER_data/WNUT/bert_e4_lr5/'

##The first save path is for the adversarial sentences 
savepath = '/NER_data/WNUT/BERT1/output_bert1_context'

##The second save path is other output
savepath2 = '/NER_data/WNUT/BERT1/'

only_ent_ids, no_ent_ids, only_wrong_ids, out_texts = AdversarialBERT().main(test, modelpath,savepath, savepath2, 'labels.txt', devdata, traindata, testdata, random_attack =False, make_first_prediction= True, entities = False)

##only ent ids are those sentences with only entities and no other words -- EXCLUDED FROM CONTEXT ATTACK
## no ent ids are those sentences without any entities - EXCLUDED FROM CONTEXT ATTACK
## only wrong ids are those sentences with only wrong predictions of entities - EXCLUDED FROM CONTEXT ATTACK 

## out texts are the adversarial sentences.

