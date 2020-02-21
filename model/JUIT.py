import torch
import torch.nn as nn
import numpy as np
import random
from torch.nn.parameter import Parameter
np.set_printoptions(threshold=np.inf)

'''
Nice results:

Namespace(K=5, alpha=10.0, batch_size=128, beta=1.0, data_path='./data/CiteULike/', 
distil_ratio=0.1, embedding_dim=100, epoch_begin_update_tags=0, epoch_number=100, 
epoch_number_update_tags=10, fix_multi_label=0, gcn_activate='Sigmoid', input_part_tags=0, 
it_soft_label_ratio=0.0, it_true_label_ratio=0.0, learning_rate=0.01, mode=1, optimizer='sgd', 
reg=0.001, tag_pre_train=1, tag_pred_K=10, tag_ratio=0.1, task=0, thred=0.07, two_stages=1, 
ui_soft_label_ratio=0.0, ui_true_label_ratio=1.0, use_tags=1)
'''

class FM_fill_new(nn.Module):
    def __init__(self, num_users, num_items, num_tags, num_word, num_factors, args):
        super(FM_fill_new, self).__init__()
        self.setup_seed(0)
        self.args = args
        self.reg = self.args.reg
        self.bi_cross_entropy = torch.nn.BCELoss()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_users = num_users
        self.num_items = num_items
        self.num_tags = num_tags
        self.num_word = num_word
        self.num_factors = num_factors

        self.user_embeddings = nn.Embedding(num_users, num_factors)
        self.item_embeddings = nn.Embedding(num_items, num_factors)
        self.tag_embeddings = nn.Embedding(num_tags, num_factors)
        self.user_biases = nn.Embedding(num_users, 1)
        self.item_biases = nn.Embedding(num_items, 1)
        self.global_bias = Parameter(torch.zeros(1))
        torch.nn.init.uniform_(self.user_embeddings.weight, a=-0.1, b=0.1)
        torch.nn.init.uniform_(self.item_embeddings.weight, a=-0.1, b=0.1)
        torch.nn.init.uniform_(self.tag_embeddings.weight, a=-0.1, b=0.1)
        torch.nn.init.uniform_(self.user_biases.weight, a=-0.1, b=0.1)
        torch.nn.init.uniform_(self.item_biases.weight, a=-0.1, b=0.1)
        torch.nn.init.uniform_(self.global_bias.data, a=-0.1, b=0.1)


        self.tag_predictor_weight = Parameter(torch.DoubleTensor(self.num_word, num_factors), requires_grad=True)
        self.tag_predictor_out_weight = Parameter(torch.DoubleTensor(num_factors, self.num_tags), requires_grad=True)
        self.user_tag_predictor_weight = Parameter(torch.DoubleTensor(self.num_users, num_factors), requires_grad=True)
        self.user_tag_predictor_out_weight = Parameter(torch.DoubleTensor(num_factors, self.num_tags), requires_grad=True)

        if self.args.tag_pre_train:
            print('tag pre-train load ...')
            self.tag_predictor_weight.data = torch.from_numpy(np.loadtxt('../main/pre_train/' + str(self.args.tag_ratio) + '_best_tag_predictor_weight.npy'))
            self.tag_predictor_out_weight.data = torch.from_numpy(np.loadtxt('../main/pre_train/' + str(self.args.tag_ratio) + '_best_tag_predictor_out_weight.npy'))
        else:
            torch.nn.init.uniform_(self.tag_predictor_weight.data, a=-0.1, b=0.1)
            torch.nn.init.uniform_(self.tag_predictor_out_weight.data, a=-0.1, b=0.1)


    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def recomendation_prediction(self, users, items, tags):
        user_embed = self.user_embeddings(users)
        item_embed = self.item_embeddings(items)
        user_bias = self.user_biases(users)
        item_bias = self.item_biases(items)
        output = self.global_bias + user_bias + item_bias

        tags_aggregate_from_true_labels = torch.matmul(tags, self.tag_embeddings.weight)
        soft_w = self.softmax(torch.mul(item_embed.unsqueeze(1), self.tag_embeddings.weight.unsqueeze(0)).sum(dim=2))
        tags_aggregate_from_c_labels = torch.matmul(soft_w, self.tag_embeddings.weight)

        enhanced_item = item_embed + \
                        self.args.ui_true_label_ratio * tags_aggregate_from_true_labels + \
                        self.args.ui_soft_label_ratio * tags_aggregate_from_c_labels
        user_item_interaction = torch.mul(user_embed, enhanced_item).sum(dim=1)
        output = output.squeeze(-1) + user_item_interaction
        return output


    def tag_prediction(self, item_contents, items, item_users):
        item_embed = self.item_embeddings(items)
        item_contents = item_contents.double()

        users_aggregate_from_true_labels = torch.matmul(item_users.double(), self.user_embeddings.weight.double())
        soft_w = self.softmax(torch.mul(item_embed.unsqueeze(1), self.user_embeddings.weight.unsqueeze(0)).sum(dim=2))
        users_aggregate_from_c_labels = torch.matmul(soft_w.double(), self.user_embeddings.weight.double())

        input_layer = torch.matmul(item_contents, self.tag_predictor_weight) + \
                      self.args.it_true_label_ratio * users_aggregate_from_true_labels + \
                        self.args.it_soft_label_ratio * users_aggregate_from_c_labels

        input_layer_non_linear = torch.sigmoid(input_layer)
        output_layer = torch.matmul(input_layer_non_linear, self.tag_predictor_out_weight) # batch * tag_number
        return self.softmax(output_layer).float(), output_layer.float()


    def user_tag_prediction(self, item_users):
        item_users = item_users.double()
        input_layer = torch.matmul(item_users, self.user_tag_predictor_weight)
        input_layer_non_linear = torch.sigmoid(input_layer)
        output_layer = torch.matmul(input_layer_non_linear, self.user_tag_predictor_out_weight) # batch * tag_number
        return self.softmax(output_layer).float(), output_layer.float()


    def forward(self, user_batch, item_batch, tag_batch, item_content_batch, item_users_batch, tag_tranable, ground_truth):
        users = torch.LongTensor(user_batch).to(self.device)
        items = torch.LongTensor(item_batch).to(self.device)
        tags = torch.FloatTensor(tag_batch).to(self.device)
        item_contents = torch.FloatTensor(item_content_batch).to(self.device)
        item_users = torch.FloatTensor(item_users_batch).to(self.device)
        ground = torch.FloatTensor(ground_truth).to(self.device)

        loss = 0.0
        user_item_loss = 0.0
        item_tag_loss = 0.0

        # compute user_item_loss
        index_origin = np.where(np.array(tag_tranable) == 1)[0]
        index_added = np.where(np.array(tag_tranable) == 2)[0]
        index = np.array(index_origin.tolist() + index_added.tolist())
        if len(index) > 0:
            users_valid = users[index]
            items_valid = items[index]
            tags_valid = tags[index]
            ground_valid = ground[index]

            item_content_valid = item_contents[index]
            item_users_valid = item_users[index]
            tag_pred_valid, _ = self.tag_prediction(item_content_valid, items_valid, item_users_valid)  # batch * tag_number

            if self.args.input_part_tags:
                recom_pred_valid = self.recomendation_prediction(users_valid, items_valid, tags_valid)
            else:
                recom_pred_valid = self.recomendation_prediction(users_valid, items_valid, tag_pred_valid)
            p_valid = torch.sigmoid(recom_pred_valid)
            recom_pred_loss_valid = self.bi_cross_entropy(p_valid, ground_valid)  # ()
            user_item_loss += recom_pred_loss_valid


        index = np.where(np.array(tag_tranable) == 0)[0]
        if len(index) > 0:
            users_not_valid = users[index]
            items_not_valid = items[index]
            tags_not_valid = tags[index]
            ground_not_valid = ground[index]

            recom_pred_not_valid = self.recomendation_prediction(users_not_valid, items_not_valid, tags_not_valid)
            p_not_valid = torch.sigmoid(recom_pred_not_valid)
            recom_pred_loss_not_valid = self.bi_cross_entropy(p_not_valid, ground_not_valid)  # ()
            user_item_loss += recom_pred_loss_not_valid


        # compute item_tag_loss
        index_origin = np.where(np.array(tag_tranable) == 0)[0]
        index_added = np.where(np.array(tag_tranable) == 2)[0]
        index = np.array(index_origin.tolist() + index_added.tolist())
        if len(index) > 0:
            multi_label_items_not_tranable = items[index]
            multi_label_item_content_not_tranable = item_contents[index]
            multi_label_item_users_not_tranable = item_users[index]
            multi_label_tags_not_tranable = tags[index]

            tag_pred_not_tranable, tag_pred_not_tranable_logist = self.tag_prediction(
                                                        multi_label_item_content_not_tranable,
                                                        multi_label_items_not_tranable,
                                                        multi_label_item_users_not_tranable)  # batch * tag_number

            tag_pred_loss_not_tranable = - torch.mul(multi_label_tags_not_tranable, torch.log(tag_pred_not_tranable)).sum(dim=1).mean()
            item_tag_loss += tag_pred_loss_not_tranable



        loss = self.args.alpha * user_item_loss + self.args.beta * item_tag_loss
        return loss, user_item_loss, item_tag_loss

