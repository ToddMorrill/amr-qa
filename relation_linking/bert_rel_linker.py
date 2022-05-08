from rel_linker_module import Rel_linker
from utils import precision_recall_f1, f1_score

import pickle
import json
import torch
from transformers import BertTokenizer, BertModel

from IPython import embed

class BertRelLinker(Rel_linker):

    def __init__(self, config=None):
        super().__init__(config)

        # print("Initializing Relation Mapping from pickle mapping file ....")

        # with open(config['probbank_mapping_path'], 'rb') as f:
        #     db = pickle.load(f)
        # self.relation_scores = db['relation_scores']
        # self.rel_arg_scores = db['rel_arg_scores']
        # self.binary_relation_scores = db['binary_relation_scores']
        # print("Relation Mappings: {} rel arg, {} binary, {} predicates".format(len(self.rel_arg_scores),
        #                                                                        len(self.binary_relation_scores),
        #                                                                        len(self.relation_scores)))
        self.relation_scores = config['relation_scores']

        self.bert_tokenizer = config['bert_tokenizer_class'].from_pretrained(config['bert_model_type'], 
                                                                          do_lower_case=config['do_lower_case'])
        self.bert_model = config['bert_model'].from_pretrained(config['bert_model_type'])
    
    def get_relation_candidates(self, params=None):

        relation_scores = {}
        question = params['question']

        #### Initialize the relation_scores to be the mapping values ####
        for item in self.relation_scores[params['edge_component']]:
            if params['threshold'] == 1.0:   
                if item['score'] >= params['threshold']:
                    relation_scores[item['rel']] = item['score']
            else:
                if item['score'] > params['threshold']:
                    relation_scores[item['rel']] = item['score']

        #### Encoding question and relations using Bert and update relation_scores based on dot product similarity####
        self.bert_model.eval()
        with torch.no_grad():
            question_input_ids = torch.tensor(self.bert_tokenizer.encode(question, add_special_tokens=True)).unsqueeze(0)
            question_segment_ids = torch.tensor([1] * len(question_input_ids)).unsqueeze(0)
            question_embeddings = self.bert_model(question_input_ids, question_segment_ids).pooler_output  # [CLS] embedding

            for rel in relation_scores.keys():
                dp_tokens = rel.split(':')[1]

                # split tokens based on captalization
                if params['do_cap']:
                    cap_items = [(idx, c) for idx, c in enumerate(dp_tokens) if c.isupper()]
                    if len(cap_items) != 0:
                        cap_idx, _ = cap_items[0]
                        dp_tokens = ' '.join([dp_tokens[:cap_idx], dp_tokens[cap_idx:].lower()])

                rel_input_ids = torch.tensor(self.bert_tokenizer.encode(dp_tokens, add_special_tokens=True)).unsqueeze(0)
                rel_segment_ids = torch.tensor([1] * len(rel_input_ids)).unsqueeze(0)
                rel_embeddings = self.bert_model(rel_input_ids, rel_segment_ids).pooler_output
                dot_sim_score = torch.mm(question_embeddings, rel_embeddings.T).view(-1).item()  # similarity score between question and rel
                relation_scores[rel] += dot_sim_score
            
        #### Order the updated relation_scores  ####
        relation_scores = dict(sorted(relation_scores.items(), key=lambda x: x[1], reverse=True))
        # print(relation_scores)

        return list(relation_scores.keys())[:params['top-K']]

if __name__ == "__main__":
    config = {
        'probbank_mapping_path': './data/probbank-dbpedia.pkl',
        'bert_model_type': 'bert-base-uncased',
        'do_lower_case': True,
        'bert_tokenizer_class': BertTokenizer,
        'bert_model': BertModel
    }

    # hyperparameters
    top_k = 1
    threshold = 0.9
    do_cap = False

    # bert rel linker
    rel_linker = BertRelLinker(config)

    ##### Evaluate the relation linker refering to SLING evalution scripts #####
    with open('./data/dev_qald9_rel.json') as jfile:
        data = json.load(jfile)
    
    p_tot, r_tot, f1_tot = 0.0, 0.0, 0.0
    q_count = 0

    for d, d_item in data.items():
        y_pred, y_true = [], []
        if len(d_item) == 0: 
            continue
        question = d_item['question']
        gold_relations = d_item['relations']

        if d_item['edge_component'] == '' and len(gold_relations) == 0:
            predicted_relations = []
            p, r, f1 = 1, 1, 1
        else:
            top_k = len(gold_relations)
            params = {
                'edge_component': d_item['edge_component'],
                'top-K': top_k,
                'threshold': threshold,
                'do_cap': do_cap
            }
            predicted_relations = rel_linker.get_relation_candidates(question, params=params)
            p, r, f1 = precision_recall_f1(predicted_relations, gold_relations)

        print("QID: {}".format(d))
        print("edge comp: {}".format(d_item['edge_component']))
        print("gold: {}".format(", ".join(gold_relations)))
        print("predicted: {}\n".format(", ".join(predicted_relations)))
        
        print('---------------------------------------')
        print("QID: {}\nQuestion: {}\n".format(d, question))
        print("P: {}, R: {}, F1: {}".format(p, r, f1))
        print('---------------------------------------\n\n')

        p_tot += p
        r_tot += r
        f1_tot += f1
        q_count += 1

        print('Global: {} questions'.format(q_count))
        print("P: {}, R: {}, F1: {}".format(p_tot/q_count, r_tot/q_count, f1_score(p_tot/q_count, r_tot/q_count)))
        print('---------------------------------------\n\n')

    p_tot /= q_count
    r_tot /= q_count
    f1_tot = f1_score(p_tot, r_tot)

    print("\n\n\nFinal results:\n\t# of Qs: {}\t\nPrecision: {}\n\tRecall: {}\n\tF1: {}".format(q_count, p_tot, r_tot,
                                                                                                f1_tot))