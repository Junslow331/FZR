# -*- coding: utf-8 -*-
import json

data_path = '../data/'
dataname = 'Wiki'

rel2candidates = json.load(open(data_path + dataname + "/rel2candidates_all.json"))
e1rel_e2 = json.load(open(data_path + dataname + "/e1rel_e2_all.json"))

relation2id = json.load(open(data_path + dataname + "/relation2ids"))
entity2id = json.load(open(data_path + dataname + "/entity2id"))

def gen_test_candidates(now_tasks, mode):
    test_candidates = dict()
    for query_ in now_tasks.keys():
        # if len(now_tasks[query_]) > 500:
        #     continue
        print(len(now_tasks[query_]))
        test_candidates[query_] = dict()

        candidates = rel2candidates[query_]
        for triple in now_tasks[query_]:
            head = triple[0]
            rela = triple[1]
            true = triple[2]
            tail_candidates = []
            tail_candidates.append(true)

            for ent in candidates:
                if ent not in entity2id.keys(): # not entity2id.has_key(ent):
                    continue
                if (ent not in e1rel_e2[triple[0]+triple[1]]) and ent != true:
                    tail_candidates.append(ent)

            test_candidates[query_][str(head)+'\t'+str(rela)] = tail_candidates

    json.dump(test_candidates, open(data_path + dataname + '/' + mode + "_candidates1.json", "w"))
    print("Finish", mode, "candidates!!")
    print(len(test_candidates.keys()))
    return test_candidates

if __name__ == '__main__':
    #  dev_tasks = json.load(open(data_path + dataname + "/dev_tasks.json"))
     test_tasks = json.load(open(data_path + dataname + "/test_tasks.json"))
    #  train_tasks = json.load(open(data_path + dataname + "/train_tasks.json"))
     
    #  gen_test_candidates(dev_tasks, 'dev')
     gen_test_candidates(test_tasks, 'test')
    #  gen_test_candidates(train_tasks, 'train')