from tqdm import tqdm
import json
import random

data_path = '../data/'
dataname = 'Wiki'
train_name = 'new1'

rel2candidates = json.load(open(data_path + dataname + "/rel2candidates_all.json"))
e1rel_e2 = json.load(open(data_path + dataname + "/e1rel_e2_all.json"))

relation2id = json.load(open(data_path + dataname + "/relation2ids"))
entity2id = json.load(open(data_path + dataname + "/entity2id"))

def gen_test_candidates(now_tasks, mode):
    test_candidates = dict()
    for query_ in now_tasks.keys():
        # if len(now_tasks[query_]) > 500:
        #     continue
        # print(len(now_tasks[query_]))
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

    # json.dump(test_candidates, open(data_path + dataname + '/' + mode + "_candidates.json", "w"))
    print("Finish", mode, "candidates!!")
    return test_candidates


train_tasks = json.load(open(data_path + dataname + "/train_tasks.json"))

dev_tasks = json.load(open(data_path + dataname + "/dev_tasks.json"))

test_tasks = json.load(open(data_path + dataname + "/test_tasks.json"))


all_task = {**train_tasks, **dev_tasks, **test_tasks}

train_rel = list(train_tasks.keys())
idx_list = list(range(len(train_rel)))
random.shuffle(idx_list)
# dev_idx = idx_list[:-48]
# dev_idx = idx_list[-48-20:-48]
# train_idx = idx_list[:-48-20]

# dev_idx = idx_list[-20:]
# train_idx = idx_list[:-20]
add_idx = idx_list[-20:]
add_rel = []
for i in add_idx:
    add_rel.append(train_rel[i])

new_test_task = {}
new_dev_task = {}
new_train_task = {}

test_add_rel = list(set(list(test_tasks.keys())) | set(add_rel) | set(list(dev_tasks.keys())))
idx_list = list(range(len(test_add_rel)))
random.shuffle(idx_list)
test_idx = idx_list[-48:]
test_rel = []
for i in test_idx:
    test_rel.append(test_add_rel[i])
for rel in test_rel:
    new_test_task[rel] = all_task[rel]
print(len(test_rel))
print(len(new_test_task))


train_dev_rel = list(set(list(all_task.keys())) - set(test_rel))
idx_list = list(range(len(train_dev_rel)))
random.shuffle(idx_list)
dev_idx = idx_list[-20:]
train_idx = idx_list[:-20]

dev_rel = []
for i in dev_idx:
    dev_rel.append(train_dev_rel[i])
for rel in dev_rel:
    new_dev_task[rel] = all_task[rel]
print(len(dev_rel))
print(len(new_dev_task))

train_rel = []
for i in train_idx:
    train_rel.append(train_dev_rel[i])
for rel in train_rel:
    new_train_task[rel] = all_task[rel]
print(len(train_rel))
print(len(new_train_task))


new_test_candidates = gen_test_candidates(new_test_task, 'test')


with open('../data/Wiki/' + train_name + '_test_candidates.json', 'w', encoding='utf-8') as fw:
    json.dump(new_test_candidates, fw)

with open('../data/Wiki/' + train_name + '_test_tasks.json', 'w', encoding='utf-8') as fw:
    json.dump(new_test_task, fw)

with open('../data/Wiki/datasplit/' + train_name + '_train_tasks.json', 'w', encoding='utf-8') as fw:
    json.dump(new_train_task, fw)

with open('../data/Wiki/datasplit/' + train_name + '_dev_tasks.json', 'w', encoding='utf-8') as fw:
    json.dump(new_dev_task, fw)

