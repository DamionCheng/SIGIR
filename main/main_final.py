import torch
import numpy as np
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
setup_seed(0)
np.set_printoptions(threshold=np.inf)
import torch.optim as optim
import time

from model.JUIT import FM_fill_new
from data_loader.data_loader_final import DataLoader
import argparse
from utils.utils import Evaluate
import sys

'multi-label pre-train param:'
'Namespace(FM_tag_flag=1, K=5, alpha=10.0, batch_size=128, beta=1.0, embedding_dim=100, ' \
'epoch_begin_update_tags=0, epoch_number=100, epoch_number_update_tags=1, ' \
'gcn_activate="Sigmoid", learning_rate=0.2, mode=1, optimizer="sgd", ' \
'prediction_f=1, reg=0.001, tag_pre_train=0, tag_pred_K=10, tag_ratio=0.1, task=3, two_stages=2'



def parameter_parser():
    parser = argparse.ArgumentParser(description="Run FM_fill.")
    parser.add_argument("--data_path", type=str, default="../data/CiteULike/", help="data path")
    parser.add_argument("--embedding_dim", type=int, default=100, help="embedding dimension of the graph vertex")
    parser.add_argument("--K", type=int, default=5, help="recommending how many items to a user")
    parser.add_argument("--tag_pred_K", type=int, default=10, help="max number of tags predicted")
    parser.add_argument("--epoch_number", type=int, default=100, help="number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--optimizer", type=str, default="sgd", help="adam, sgd, adadelta, adagrad or RMSprop")
    parser.add_argument("--reg", type=float, default=0.001, help="regularation parameter")
    parser.add_argument("--alpha", type=float, default=10.0, help="weight for continue-discrete loss")
    parser.add_argument("--beta", type=float, default=1.0, help="weight for multi-label loss")
    parser.add_argument("--gcn_activate", type=str, default='Sigmoid', help="")
    parser.add_argument("--epoch_begin_update_tags", type=int, default=10000, help="")
    parser.add_argument("--epoch_number_update_tags", type=int, default=10, help="")
    parser.add_argument("--mode", type=int, default=1, help="1 is better")
    parser.add_argument("--tag_ratio", type=float, default=0.1, help="visible item tags")
    parser.add_argument("--thred", type=float, default=0.07, help="visible item tags")
    parser.add_argument("--input_part_tags", type=int, default=0,   help="")
    parser.add_argument("--tag_pre_train",       type=int,   default=1,   help="use pretrained multilable predictor")
    parser.add_argument("--ui_true_label_ratio", type=float, default=1.0, help="ui true label")
    parser.add_argument("--ui_soft_label_ratio", type=float, default=0.0, help="ui soft label")
    parser.add_argument("--it_true_label_ratio", type=float, default=0.0, help="it true label")
    parser.add_argument("--it_soft_label_ratio", type=float, default=0.0, help="it soft label")
    parser.add_argument("--distil_ratio", type=float, default=0.1, help="distill ratio")

    return parser.parse_args()


'''
our model (with boot)
our model (without boot)

MF with tags (2 stages)
MF with tags (1 stages)
MF without tags

Tag prediction
Tag prediction (boot)
'''


# our model:
# --alpha=10.0 --beta=1.0 --task=0 --input_part_tags=0 --fix_multi_label=0 --ui_true_label_ratio=1.0
# MF with tags (2 stages):
# --alpha=1.0 --beta=0.0 --task=0 --input_part_tags=0 --fix_multi_label=1 --ui_true_label_ratio=1.0
# MF with tags (1 stages):
# --alpha=1.0 --beta=0.0 --task=0 --input_part_tags=1 --ui_true_label_ratio=1.0
# MF without tags:
# --alpha=1.0 --beta=0.0 --task=0 --ui_true_label_ratio=0.0




def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class experiment():
    def __init__(self, args, model, data):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.data = data
        self.args = args
        self.new_added_item_tags = dict()
        if self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=0.99)
        elif self.args.optimizer == 'sgd':
            paras_new = []
            paras = dict(self.model.named_parameters())
            for k, v in paras.items():
                if 'bias' in k:
                    paras_new += [{'params': [v], 'lr': self.args.learning_rate, 'weight_decay': 0}]
                elif 'predictor' in k:
                    paras_new += [{'params': [v], 'lr': self.args.learning_rate, 'weight_decay': self.args.reg}]
                else:
                    paras_new += [{'params': [v], 'lr': self.args.learning_rate, 'weight_decay': self.args.reg}]
            self.optimizer = optim.SGD(paras_new, momentum=0.9)
            print(self.get_learning_rate(self.optimizer))

        elif self.args.optimizer == 'adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.args.learning_rate, rho=0.9, eps=1e-06,
                                            weight_decay=0)
        elif self.args.optimizer == 'adagrad':
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.args.learning_rate, lr_decay=0,
                                           weight_decay=0, initial_accumulator_value=0)
        elif self.args.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.args.learning_rate, alpha=0.99, eps=1e-08,
                                           weight_decay=0, momentum=0, centered=False)

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    def get_learning_rate(self, optimizer):
        lr = []
        for param_group in optimizer.param_groups:
            lr += [param_group['lr']]
        return lr



    def shuffle_FM_fill_new(self, train_users, train_items, train_tags, train_item_content, train_tag_tranable, train_ground_truth):
        train_records_num = len(train_users)
        index = np.array(range(train_records_num))
        np.random.shuffle(index)
        train_users = np.array(train_users)[index]
        train_items = np.array(train_items)[index]
        train_tags = np.array(train_tags)[index]
        train_item_content = np.array(train_item_content)[index]
        train_tag_tranable = np.array(train_tag_tranable)[index]
        train_ground_truth = np.array(train_ground_truth)[index]
        return train_users, train_items, train_tags, train_item_content, train_tag_tranable, train_ground_truth

    def run_FM_fill_new(self):
        recom_result = []
        recom_max_value = 0
        recom_best_result_index = 0
        tagging_result = []
        tagging_max_value = 0
        tagging_best_result_index = 0
        self.train_users = np.array(self.data.users)
        self.train_items = np.array(self.data.items)
        self.train_item_tags = np.array(self.data.tags)
        self.train_item_contents = np.array(self.data.item_content)
        self.train_item_users = np.array(self.data.item_users)
        self.train_tag_tranable = np.array(self.data.tag_tranable)
        self.train_ground_truth = np.array(self.data.ground_truth)
        self.evaluator_recom = Evaluate(self.args.K)
        self.evaluator_tagging = Evaluate(self.args.tag_pred_K)
        print('assert eval ...')
        auc, map, mrr, p, r, f1, hit, ndcg = self.eval_tag_FM_fill_new(10)
        print('pre-trained results: ', p, r, f1, hit, ndcg)


        for epoch in range(self.args.epoch_number):
            if epoch >= self.args.epoch_begin_update_tags and epoch % self.args.epoch_number_update_tags == 0:
                self.update_trainable_tag_labels()
            print('********************* training epoch begin *********************')
            train_record_number = len(self.data.users)
            print('training sample number: ', train_record_number)
            total_loss = 0.0
            s = time.time()
            step = 0
            max_steps = train_record_number / self.args.batch_size
            self.model.train()

            while step <= max_steps:
                if (step + 1) * self.args.batch_size > train_record_number:
                    b = train_record_number - step * self.args.batch_size
                else:
                    b = self.args.batch_size
                start = step * self.args.batch_size
                end = start + b
                if end > start:
                    self.optimizer.zero_grad()
                    loss, user_item_loss, item_tag_loss = self.model(self.train_users[start:end],
                                      self.train_items[start:end],
                                      self.train_item_tags[start:end],
                                      self.train_item_contents[start:end],
                                      self.train_item_users[start:end],
                                      self.train_tag_tranable[start:end],
                                      self.train_ground_truth[start:end])
                    total_loss += loss.cpu().detach().numpy()
                    loss.backward(retain_graph=True)
                    # torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.0)
                    # clip_gradient(self.optimizer, 1.0)
                    self.optimizer.step()
                step += 1

            print('epoch:', epoch, 'loss: ', str(total_loss))
            print('training time: \t', (time.time() - s), ' sec')
            print('********************* training epoch end *********************')


            print('&&&&&&&&&&&&&&&&&&&& test epoch begin &&&&&&&&&&&&&&&&&&&&')
            self.model.eval()
            with torch.no_grad():
                start_time = time.time()
                print('Top-N recommendation testing eval ...')
                auc_test, map_test, mrr_test, p_test, r_test, f1_test, hr_test, ndcg_test = \
                    self.performance_eval_FM_fill_new(self.model, self.data.compare_user_items_dict,
                                                  self.data.compare_user_tags_dict, self.data.compare_user_trainable_dict,
                                                  self.data.compare_ground_truth_dict,
                                                  10)

                recom_result.append([p_test, r_test, f1_test, hr_test, ndcg_test])
                if f1_test > recom_max_value:
                    recom_max_value = f1_test
                    recom_best_result_index = epoch
                print('current best testing performance [Top-N recommendation]: ' + str(recom_result[recom_best_result_index]))
                print('testing performance [Top-N recommendation]: ', p_test, r_test, f1_test, hr_test, ndcg_test)

                auc, map, mrr, p, r, f1, hit, ndcg = self.eval_tag_FM_fill_new(10)
                tagging_result.append([p, r, f1, hit, ndcg])
                if f1 > tagging_max_value:
                    tagging_max_value = f1
                    tagging_best_result_index = epoch
                print('current best testing performance [Tagging]: ' + str(tagging_result[tagging_best_result_index]))
                print('testing performance [Tagging]: ', p, r, f1, hit, ndcg)
                print('total testing time: \t', (time.time() - start_time), ' sec')
                print('&&&&&&&&&&&&&&&&&&&& test epoch end &&&&&&&&&&&&&&&&&&&&')

            sys.stdout.flush()

        print('-------------------- learning end --------------------')
        print(recom_result)
        print(recom_best_result_index)
        print('Top-N recommendation final best performance: ' + str(recom_result[recom_best_result_index]))

        print(tagging_result)
        print(tagging_best_result_index)
        print('Tagging final best performance: ' + str(tagging_result[tagging_best_result_index]))
        return recom_result[recom_best_result_index], recom_result, tagging_result[tagging_best_result_index], tagging_result

    def performance_eval_FM_fill_new(self, model, compare_user_items_dict, compare_user_tags_dict, compare_user_trainable_dict, compare_ground_truth_dict, eval_number):
        # pred: { uid: {iid : score, iid : score, ...}}
        # ground_truth: { uid: {iid : 0/1, iid : 0/1, ...}}
        test_users = list(compare_user_items_dict.keys())
        test_user_number = len(test_users)
        pred = dict()
        ground_truth = dict()

        max_steps = test_user_number / eval_number
        step = 0
        while step <= max_steps:
            if (step + 1) * eval_number > test_user_number:
                b = test_user_number - step * eval_number
            else:
                b = eval_number
            start = step * eval_number
            end = start + b
            if end > start:
                usr = test_users[start:end]
                usrs = [[int(u)] * len(compare_user_items_dict[u]) for u in usr]
                its = [compare_user_items_dict[u] for u in usr]
                tgs = [compare_user_tags_dict[u] for u in usr]

                users = torch.LongTensor(usrs).to(self.device)
                items = torch.LongTensor(its).to(self.device)
                tags = torch.FloatTensor(tgs).to(self.device)

                sample_number = len(usrs) * len(usrs[0])

                scs = model.recomendation_prediction(users.view(-1), items.view(-1), tags.view([sample_number, -1]))
                scores = scs.cpu().detach().numpy().reshape((len(users), -1))  # [len(users), len(users[0])]
                pred_tmp = {usr[i]: {its[i][j]: scores[i][j] for j in range(len(scores[i]))} for i in range(len(usr))}
                ground_truth_tmp = {usr[i]: {its[i][j]: compare_ground_truth_dict[usr[i]][j] for j in
                                             range(len(compare_ground_truth_dict[usr[i]]))} for i in range(len(usr))}

                pred.update(pred_tmp)
                ground_truth.update(ground_truth_tmp)
            step += 1
        auc, map, mrr, p, r, f1, hit, ndcg = self.evaluator_recom.evaluate(ground_truth, pred)
        return auc, map, mrr, p, r, f1, hit, ndcg

    def eval_tag_FM_fill_new(self, eval_number):
        '''
        p_our = []
        p_base = []
        K = 10
        for itm in list(self.data.item_tag_dict_not_visible.keys()):
            item = torch.LongTensor([itm]).to(self.device)

            ctn = self.data.item_content_dict[itm]
            content = torch.FloatTensor([ctn]).to(self.device)

            if itm in self.data.item_interacted_users.keys():
                ius = self.data.item_interacted_users[itm]
            else:
                ius = []
            item_users = np.zeros(self.data.user_number)
            item_users[ius] = 1
            item_users = torch.FloatTensor([item_users]).to(self.device)


            self.model.eval()
            with torch.no_grad():
                S, _ = self.model.tag_prediction(content, item, item_users)
            S = S.cpu().detach().numpy().mean(axis=0)

            our = np.argsort(S)[::-1][:K]
            ground = np.where(np.array(self.data.item_tag_dict_not_visible[itm]) == 1)[0]

            tmp = self.data.predicted_tags[itm]
            base = np.argsort(tmp)[::-1][:K]

            our_cross = len([i for i in our if i in ground])
            base_cross = len([i for i in base if i in ground])

            p_our.append(float(our_cross) / K)
            p_base.append(float(base_cross) / K)

        print 'tagging performance:'
        print 'our: ', np.array(p_our).mean()
        print 'base: ', np.array(p_base).mean()
        '''
        test_items = list(self.data.item_tag_dict_not_visible.keys())
        test_item_number = len(test_items)
        pred = dict()
        ground_truth = dict()

        max_steps = test_item_number / eval_number
        step = 0
        while step <= max_steps:
            if (step + 1) * eval_number > test_item_number:
                b = test_item_number - step * eval_number
            else:
                b = eval_number
            start = step * eval_number
            end = start + b
            if end > start:
                its = test_items[start:end]
                item = torch.LongTensor(its).to(self.device)

                ctn = [self.data.item_content_dict[itm] for itm in its]
                content = torch.FloatTensor(ctn).to(self.device)

                item_users = []
                for itm in its:
                    if itm in self.data.item_interacted_users.keys():
                        ius = self.data.item_interacted_users[itm]
                    else:
                        ius = []
                    tmp = np.zeros(self.data.user_number)
                    tmp[ius] = 1
                    item_users.append(tmp)
                item_users = torch.FloatTensor(item_users).to(self.device)

                self.model.eval()
                with torch.no_grad():
                    S, _ = self.model.tag_prediction(content, item, item_users)
                S = S.cpu().detach().numpy()

                pred_tmp = {its[i]: {j: S[i][j] for j in range(len(S[i]))} for i in range(len(its))}
                ground_truth_tmp = {its[i]: {j: self.data.item_tag_dict_not_visible[its[i]][j] for j in
                                             range(len(self.data.item_tag_dict_not_visible[its[i]]))} for i in range(len(its))}
                pred.update(pred_tmp)
                ground_truth.update(ground_truth_tmp)

            step += 1

        auc, map, mrr, p, r, f1, hit, ndcg = self.evaluator_tagging.evaluate(ground_truth, pred)
        return auc, map, mrr, p, r, f1, hit, ndcg

    def update_trainable_tag_labels(self):
        items_to_label = list(self.data.item_tag_dict_not_visible.keys())
        changed_items = 0
        for itm in items_to_label:
            item = torch.LongTensor([itm]).to(self.device)

            ctn = self.data.item_content_dict[itm]
            content = torch.FloatTensor([ctn]).to(self.device)

            if itm in self.data.item_interacted_users.keys():
                ius = self.data.item_interacted_users[itm]
            else:
                ius = []
            item_users = np.zeros(self.data.user_number)
            item_users[ius] = 1
            item_users = torch.FloatTensor([item_users]).to(self.device)

            self.model.eval()
            with torch.no_grad():
                S, _ = self.model.tag_prediction(content, item, item_users)

            S = S.cpu().detach().numpy().mean(axis=0)

            index = np.where(S > self.args.thred)[0]
            if len(index) > 0:
                S_ = np.zeros(np.shape(S))
                S_[index.tolist()] = 1
                if itm in self.new_added_item_tags.keys():
                    if (S_ != self.new_added_item_tags[itm]).any():
                        self.new_added_item_tags[itm] = S_
                        changed_items += 1
                else:
                    self.new_added_item_tags[itm] = S_
                    changed_items += 1
        print('current epoch changed item-tagging number: ', changed_items)

        print('change train item tags')
        for i in range(len(np.array(self.train_items))):
            if np.array(self.train_items)[i] in self.new_added_item_tags.keys():
                self.train_item_tags[i] = self.new_added_item_tags[np.array(self.train_items)[i]]
                self.train_tag_tranable[i] = 2




if __name__ == "__main__":
    '''
    The entrance of the application.
    (1) building dataset
    (2) defining model
    (3) training the model based on the dataset
    '''

    args = parameter_parser()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("current envirment: " + str(device))

    file_path = '../results/' + args.data_path.split('/')[-2]
    file = open(file_path, 'w')

    # (1) building dataset
    t = time.time()
    data = DataLoader(args)
    print(data.user_number)
    print(data.item_number)
    print(data.tag_number)
    print(data.word_number)

    s = time.time()
    data.generate_training_corpus_new()
    data.generate_validation_corpus_new()
    print('loading cost: ', time.time()-s)
    model = FM_fill_new(data.user_number, data.item_number, data.tag_number, data.word_number, args.embedding_dim, args).to(device)
    exp = experiment(args, model, data)
    r, r_epochs, _, _ = exp.run_FM_fill_new()


    file.writelines('parameters: ')
    file.writelines('\r\n')
    file.writelines(str(args))
    file.writelines('\r\n')
    file.writelines('results epochs: ')
    file.writelines('\r\n')
    file.writelines(str(np.array(r_epochs)))
    file.writelines('\r\n')
    file.writelines('best result: ')
    file.writelines('\r\n')
    file.writelines(str(np.array(r)))
    file.writelines('\r\n')

    print(r)




