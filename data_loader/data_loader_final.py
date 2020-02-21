import pickle
import numpy as np
import random
import torch
np.set_printoptions(threshold=np.inf)


class DataLoader():
    def __init__(self, args):
        print("loading data ...")
        self.setup_seed(0)
        self.path = args.data_path
        self.args = args
        self.train_user_items_id_dict = pickle.load(open(self.path + "train_user_items_id_dict", "rb"))
        print(np.array([len(i) for i in self.train_user_items_id_dict.values()]).max())
        self.test_user_items_id_dict = pickle.load(open(self.path + "test_user_items_id_dict", "rb"))
        print(np.array([len(i) for i in self.test_user_items_id_dict.values()]).max())
        self.item_tag_dict_visible = pickle.load(open(self.path + "item_tag_id_dict_visible_"+str(self.args.tag_ratio), "rb"))
        print(len(self.item_tag_dict_visible))
        self.item_tag_dict_not_visible = pickle.load(open(self.path + "item_tag_id_dict_not_visible_"+str(self.args.tag_ratio), "rb"))
        print(len(self.item_tag_dict_not_visible))
        self.item_content_dict = pickle.load(open(self.path + "item_content_id_dict", "rb"))

        self.statistic_dict = pickle.load(open(self.path + "statistic_dict", "rb"))
        self.user_number = self.statistic_dict['user_number']
        self.item_number = self.statistic_dict['item_number']
        self.tag_number  = self.statistic_dict['tag_number']
        self.word_number = self.statistic_dict['word_number']

        self.compare_user_items_dict = dict()
        self.compare_user_tags_dict = dict()
        self.compare_user_contents_dict = dict()
        self.compare_user_trainable_dict = dict()
        self.compare_ground_truth_dict = dict()

        self.train_user_items_dict = dict()
        self.train_user_tags_dict = dict()
        self.train_user_contents_dict = dict()
        self.train_user_trainable_dict = dict()
        self.train_ground_truth_dict = dict()


    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def generate_training_corpus_new(self):
        print('loading training data ...')
        self.item_interacted_users = pickle.load(open(self.path + 'train_item_interacted_users_dict', 'rb'))

        self.users = []
        self.items = []
        self.tags = []
        self.item_content = []
        self.ground_truth = []
        self.tag_tranable = []
        self.item_users = []


        for user, items in self.train_user_items_id_dict.items():
            l = len(items)
            not_interacted_items = [i for i in self.statistic_dict['real_train_items'] if i not in items]
            neg_items = random.sample(not_interacted_items, l)

            for j in range(l):
                item = items[j]
                if item in self.item_tag_dict_visible.keys():
                    tag = self.item_tag_dict_visible[item]
                    self.tag_tranable.append(0)
                else:
                    tag = [0]*self.tag_number
                    self.tag_tranable.append(1)
                item_users = self.item_interacted_users[item]
                tmp = np.zeros(self.user_number)
                tmp[item_users] = 1
                self.item_users.append(tmp)
                self.users.append(int(user))
                self.items.append(int(item))
                self.tags.append(tag)
                self.item_content.append(self.item_content_dict[item])
                self.ground_truth.append(1)


            for j in range(len(neg_items)):
                neg_item = neg_items[j]
                if neg_item in self.item_tag_dict_visible.keys():
                    tag = self.item_tag_dict_visible[neg_item]
                    self.tag_tranable.append(0)
                else:
                    tag = [0]*self.tag_number
                    self.tag_tranable.append(1)
                if neg_item in self.item_interacted_users.keys():
                    item_users = self.item_interacted_users[neg_item]
                else:
                    item_users = []
                tmp = np.zeros(self.user_number)
                tmp[item_users] = 1
                self.item_users.append(tmp)
                self.users.append(int(user))
                self.items.append(int(neg_item))
                self.tags.append(tag)
                self.item_content.append(self.item_content_dict[neg_item])
                self.ground_truth.append(0)


    def generate_validation_corpus_new(self):
        print('loading testing data ...')
        compare_number = 100
        for uid, iids in self.test_user_items_id_dict.items():
            tmp = [i for i in self.statistic_dict['real_train_items'] if i not in iids and i not in self.train_user_items_id_dict[uid]]
            comparing_items = random.sample(tmp, compare_number - len(iids)) + iids
            random.shuffle(comparing_items)
            self.compare_user_items_dict[uid] = comparing_items
            self.compare_ground_truth_dict[uid] = [1 if i in iids else 0 for i in comparing_items]

            tags = []
            trainable = []
            for item in comparing_items:
                if item in self.item_tag_dict_visible.keys():
                    tag = self.item_tag_dict_visible[item]
                    trainable.append(1)
                else:
                    tag = [0]*self.tag_number
                    trainable.append(0)
                tags.append(tag)
            self.compare_user_tags_dict[uid] = tags
            self.compare_user_trainable_dict[uid] = trainable

            contents = []
            for item in comparing_items:
                content = self.item_content_dict[item]
                contents.append(content)
            self.compare_user_contents_dict[uid] = contents

