import numpy as np
from sklearn import metrics
import torch
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(0)


'''
Some useful tools
'''

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    '''
    [scipy.sparse] sparse_mx: col, row, data, shape, ...
    [torch.sparse.FloatTensor] indices, values, shape, ...
    '''
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def to_categorical(y, num_classes=None):
    categorical = np.zeros(num_classes)
    categorical[y] = 1
    return list(categorical)


class Evaluate(object):
    def __init__(self, topk):
        self.Top_K = topk

    def auc(self, ground_truth, pred):
        result = []
        for k, v in ground_truth.items():
            if len(v) > 0 and 0 in v and 1 in v:
                v = np.array(v) + 1
                #print v
                #print pred[k]
                fpr, tpr, thresholds = metrics.roc_curve(np.array(v), np.array(pred[k]), pos_label=2)
                tmp = metrics.auc(fpr, tpr)
                #print tmp
                #raw_input()
                result.append(tmp)
        return np.array(result).mean()


    def MAP(self, ground_truth, pred):
        result = []
        for k, v in ground_truth.items():
            fit = [i[0] for i in pred[k]][:self.Top_K]
            tmp = 0
            hit = 0
            for j in range(len(fit)):
                if fit[j] in v:
                    hit += 1
                    tmp += hit / (j + 1)
            result.append(tmp)
        return np.array(result).mean()

    def MRR(self, ground_truth, pred):
        result = []
        for k, v in ground_truth.items():
            fit = [i[0] for i in pred[k]][:self.Top_K]
            tmp = 0
            for j in range(len(fit)):
                if fit[j] in v:
                    tmp = 1 / (j + 1)
                    break
            result.append(tmp)
        return np.array(result).mean()

    def NDCG(self, ground_truth, pred):
        result = []
        for k, v in ground_truth.items():
            fit = [i[0] for i in pred[k]][:self.Top_K]
            temp = 0
            Z_u = 0

            for j in range(min(len(fit), len(v))):
                Z_u = Z_u + 1 / np.log2(j + 2)
            for j in range(len(fit)):
                if fit[j] in v:
                    temp = temp + 1 / np.log2(j + 2)

            if Z_u == 0:
                temp = 0
            else:
                temp = temp / Z_u
            result.append(temp)
        return np.array(result).mean()

    def top_k(self, ground_truth, pred):
        p_total = []
        r_total = []
        f_total = []
        hit_total = []
        for k, v in ground_truth.items():
            fit = [i[0] for i in pred[k]][:self.Top_K]
            cross = float(len([i for i in fit if i in v]))
            p = cross / len(fit)
            r = cross / len(v)
            if cross > 0:
                f = 2.0 * p * r / (p + r)
            else:
                f = 0.0
            hit = 1.0 if cross > 0 else 0.0
            p_total.append(p)
            r_total.append(r)
            f_total.append(f)
            hit_total.append(hit)
        return np.array(p_total).mean(), np.array(r_total).mean(), np.array(f_total).mean(), np.array(hit_total).mean()


    def evaluate(self, ground_truth, pred):
        # input
        # pred: { uid: {iid : score, iid : score, ...}}
        # ground_truth: { uid: {iid : 0/1, iid : 0/1, ...}}


        # sorted_pred: { uid: {iid: score, iid : score, ...}}
        # index_ground_truth: { uid: [iid, iid, ...]}
        sorted_pred = {}
        index_ground_truth = {}
        for k, v in pred.items():
            sorted_pred[k] = sorted(v.items(), key=lambda item: item[1])[::-1]
            index_ground_truth[k] = list(np.array(list(ground_truth[k].keys()))[np.where(np.array(list(ground_truth[k].values()))==1)[0]])

        print('case study: ')
        k_ID = list(sorted_pred.keys())[0]
        items = [i[0] for i in sorted_pred[k_ID]][:self.Top_K]
        scores = [i[1] for i in sorted_pred[k_ID]][:self.Top_K]
        print('key: ' + str(k_ID))
        print('recommended: ' + str(items))
        print('recommended scores: ' + str(scores))
        print('real: ' + str(index_ground_truth[k_ID]))

        # auc_pred: { uid: [score, score, ...]}
        # auc_ground_truth: { uid: [0/1, 0/1, ...]}
        auc_pred = {}
        auc_ground_truth = {}
        for k, v in pred.items():
            score_v = list(v.values())
            gt_v = [ground_truth[k][i] for i in list(v.keys())]
            auc_pred[k] = score_v
            auc_ground_truth[k] = gt_v

        p, r, f1, hit = self.top_k(index_ground_truth, sorted_pred)
        map = self.MAP(index_ground_truth, sorted_pred)
        mrr = self.MRR(index_ground_truth, sorted_pred)
        ndcg = self.NDCG(index_ground_truth, sorted_pred)
        auc = self.auc(auc_ground_truth, auc_pred)
        return auc, map, mrr, p, r, f1, hit, ndcg
        #return {'auc':auc, 'map':map, 'mrr':mrr, 'precision':p, 'recall':r, 'f1':f1, 'hit-ratio':hit, 'ndcg':ndcg}



if __name__ == '__main__':
    pass




