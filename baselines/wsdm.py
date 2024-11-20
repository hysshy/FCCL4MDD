from copy import deepcopy
from torch.nn import functional as F
import cv2
from baselines.base import BaseLearner
from utils.inc_net import  IncrementalNet
from utils.data_manager import average_weights, setup_seed
from torchvision import datasets, transforms
import skfuzzy as fuzz
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
import os
from typing import Dict
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image
from utils.WSDM_Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from utils.WSDM_Model import UNet
from utils.Scheduler import GradualWarmupScheduler
import os
from utils.synlabeldata import SynLabelDataset
import shutil
c = 2  # 聚类数目，根据图片的特征和数量选择
m = 1.1  # 模糊指数，一般取2
T=1000
Total_iters = 100000  # 训练总轮数
savePath = "data/wsdm_dermpt7"  # 聚类结果保存路径
g_batch_size = 7

train_trsf = transforms.Compose([
    transforms.Resize(size=(64, 64), interpolation=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    )
])

class WSDM(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager, syn_data_manager):
        setup_seed(self.seed)
        self._cur_task += 1
        self._total_classes = self._known_classes + \
            data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        print("Learning on {}-{}".format(self._known_classes, self._total_classes))
        self.init_dataloader(data_manager, syn_data_manager)
        self.fl_syn_train_loader = None
        self.cl_syn_train_loader = None
        self._fl_train()


    def _local_update(self, model, train_data_loader):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for _ in range(self.args["local_ep"]):
            if self.cl_syn_train_loader is not None:
                for _, (images, labels) in enumerate(self.cl_syn_train_loader):
                    images, labels = images.cuda(), labels.cuda()
                    output = model(images)["logits"]
                    loss = F.cross_entropy(output, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            for _, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                output = model(images)["logits"]
                loss = F.cross_entropy(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model, model.state_dict()

    def _train_syn(self, model, train_data_loader):
        model.train()
        print('syn')
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for _, (images, labels) in enumerate(train_data_loader):
            images, labels = images.cuda(), labels.cuda()
            output = model(images)["logits"]
            loss = F.cross_entropy(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return model.state_dict()

    # def _local_finetune(self, model, train_data_loader):
    #     model.train()
    #     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    #     for _ in range(self.args["local_ep"]):
    #         for _, (_, images, labels) in enumerate(train_data_loader):
    #             images, labels = images.cuda(), labels.cuda()
    #             fake_targets = labels - self._known_classes
    #             output = model(images)["logits"]
    #             loss = F.cross_entropy(output[:, self._known_classes :], fake_targets)
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #     return model.state_dict()
    
    def init_syn_dataloader(self, loaders, mode):
        for li in range(len(loaders)):
            data_loader = loaders[li]
            allImages = data_loader.dataset.dataset.images
            allLabels = data_loader.dataset.dataset.labels
            for gg, (rr, images, labels) in enumerate(data_loader):
                imagesList = allImages[rr]
                tartgetsList = allLabels[rr]
                if rr.shape[0] ==1:
                    imagesList = [imagesList]
                    tartgetsList = [tartgetsList]
                for i in range(len(tartgetsList)):
                    savepath = os.path.join(savePath+'_'+mode, 'task_' + str(self._cur_task), str(tartgetsList[i]))
                    os.makedirs(savepath, exist_ok=True)
                    shutil.copy(imagesList[i], savepath) 
    
    def diversity_cluster(self):
        self._network.eval()
        for li in range(len(self.local_train_loaders)):
            features = []
            imagesList = []
            tartgetsList = []
            idxsList = []
            d_label = {}
            allImages = self.local_train_loaders[li].dataset.dataset.images
            allLabels = self.local_train_loaders[li].dataset.dataset.labels
            for gg, (rr, images, labels) in enumerate(self.local_train_loaders[li]):
                rr = rr.numpy()
                idxsList.append(rr)
                imagesList.append(allImages[rr])
                tartgetsList.append(allLabels[rr])
                t_images = images.cuda()
                feature = self._network.extract_vector(t_images)
                features.append(feature.cpu().detach().numpy())
            features = np.concatenate(features, 0)  # 转置为skfuzzy要求的格式
            imagesList = np.concatenate(imagesList, 0)
            tartgetsList = np.concatenate(tartgetsList, 0)
            idxsList = np.concatenate(idxsList, 0)
            unique_labels = np.unique(tartgetsList)
            wsdm_label_num = np.max(tartgetsList) + 1 - self._known_classes
            for l in unique_labels:
                idx = np.where(tartgetsList == l)
                t_features = features[idx]
                t_imagesList = imagesList[idx]
                t_tartgetsList = tartgetsList[idx]
                t_idxsList = idxsList[idx]
                if len(idx[0]) < c:
                    labels = np.arange(len(idx[0]))
                else:
                    pca = PCA(n_components=min(t_features.shape[0], t_features.shape[1], 50))
                    t_features = pca.fit_transform(t_features)

                    t_features = np.array(t_features).T  # 转置为skfuzzy要求的格式
                    # 使用skfuzzy库的cmeans函数进行模糊聚类
                    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                        t_features, c, m, error=0.0005, maxiter=10000, init=None)
                    # 根据聚类结果，将图片分组并保存到不同的文件夹
                    labels = np.argmax(u, axis=0)  # 获取每张图片的聚类标签
                for i in range(c):
                    for j in range(len(t_imagesList)):
                        if labels[j] == i:  # 如果图片属于该聚类
                            d_label.setdefault(t_idxsList[j], i+wsdm_label_num+1)
                            # savef = savePath + '/' + 'task_'+str(self._cur_task) + '/' + 'client_'+str(li) + '/' + str(t_tartgetsList[j]) + '/' + f"cluster_{i}"
                            # os.makedirs(savef, exist_ok=True)  # 创建文件夹
                            # cv2.imwrite(savef + "/" + str(j) + '.jpg', t_imagesList[j])  # 保存图片到对应的文件夹
            savefile = self.train_wsdm(d_label, wsdm_label_num, self.local_train_loaders[li], li)
            self.replay_syn_data(unique_labels, wsdm_label_num, savefile, li)

    def train_wsdm(self, d_label, num_labels, train_loader, client_id):
        epochs = int(Total_iters/train_loader.sampler.num_samples)
        device = torch.device("cuda")
        # model setup
        net_model = UNet(T=T, num_labels=num_labels, num_shapes=c+num_labels+1,
                         pca_fcel=False,
                         embedding_type=1, ch=128,
                         ch_mult=[1, 2, 3, 4], aen=True,
                         attn=[2], num_res_blocks=2,
                         dropout=0.15).to(device)
        optimizer = torch.optim.AdamW(
            net_model.parameters(), lr=1e-4, weight_decay=1e-4)
        cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=epochs, eta_min=0, last_epoch=-1)
        warmUpScheduler = GradualWarmupScheduler(
            optimizer=optimizer, multiplier=2, warm_epoch=epochs // 10,
            after_scheduler=cosineScheduler)
        trainer = GaussianDiffusionTrainer(
            net_model, 1e-4, 0.02, T).to(device)

        # start training
        prog_bar = tqdm(range(epochs))
        for _, com in enumerate(prog_bar):
            for gg, (rr, images, cls_label) in enumerate(train_loader):
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                cls_label -= self._known_classes
                shape_label = []
                for i in range(len(cls_label)):
                    shape_label.append(d_label[rr[i].item()])
                cls_label = cls_label.to(device) + 1
                shape_label = torch.LongTensor(shape_label).to(device)
                if np.random.rand() < 0.1:
                    cls_label = torch.zeros_like(cls_label).to(device)
                    shape_label = torch.zeros_like(shape_label).to(device) + num_labels + 1
                loss = trainer(x_0, cls_label, shape_label).sum() / 1000
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), 1)
                optimizer.step()
            warmUpScheduler.step()
            info=("Task {}, Epoch {}/{}".format(
                self._cur_task, com + 1, epochs))
            prog_bar.set_description(info)
        savefile = os.path.join(savePath+'_model', 'task_'+str(self._cur_task), '_client'+str(client_id) + ".pt")
        os.makedirs(os.path.dirname(savefile), exist_ok=True)
        torch.save(net_model.state_dict(), savefile)
        return savefile

    def replay_syn_data(self, unique_labels, num_labels, ckptfile, client_id):
        # load model and evaluate
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # model setup
            model = UNet(T=T, num_labels=num_labels, num_shapes=c + num_labels + 1,
                             pca_fcel=False,
                             embedding_type=1, ch=128,
                             ch_mult=[1, 2, 3, 4], aen=True,
                             attn=[2], num_res_blocks=2,
                             dropout=0.15).to(device)
            ckpt = torch.load(ckptfile, map_location=device)
            model.load_state_dict(ckpt)
            print("model load weight done.")
            model.eval()
            sampler = GaussianDiffusionSampler(
                model, 1e-4, 0.02, T, w=0.3).to(device)
            # Sampled from standard normal distribution
            with torch.no_grad():
                prog_bar = tqdm(unique_labels)
                for _, cls_label in enumerate(prog_bar):
                    for m in range(1):
                        for shape_label in range(c):
                            cls_labelList = []
                            for i in range(0, g_batch_size):
                                cls_labelList.append(torch.ones(size=[1]).long() * cls_label)
                            cls_labels = torch.cat(cls_labelList, dim=0).long().to(device) + 1 -self._known_classes
                            shape_labelList = []
                            for i in range(0, g_batch_size):
                                shape_labelList.append(torch.ones(size=[1]).long() * shape_label)
                            shape_labels = torch.cat(shape_labelList, dim=0).long().to(device) + 2 + num_labels
                            # Sampled from standard normal distribution
                            noisyImage = torch.randn(
                                size=[g_batch_size, 3, 64, 64],
                                device=device)
                            sampledImgs = sampler(noisyImage, cls_labels, shape_labels)
                            sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
                            for i in range(len(sampledImgs)):
                                ndarr = sampledImgs[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu',
                                                                                                             torch.uint8).numpy()
                                im = Image.fromarray(ndarr)
                                savefile = savePath + '/' + 'task_'+str(self._cur_task) + '/'+ str(cls_label) + '/' + 'client'+str(client_id) + '_'  + f"diverse{shape_label}"+'_'+str(i)+'.jpg'
                                os.makedirs(savePath + '/' + 'task_'+str(self._cur_task) + '/'+ str(cls_label), exist_ok=True)  # 创建文件夹
                                im.save(savefile, format=None)
                                # img = cv2.imread(savefile)
                                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                # cv2.imwrite(savefile, img)
                    info = ("Task {}, replay_label {}/{}".format(
                        self._cur_task, cls_label, num_labels))
                    prog_bar.set_description(info)

    def init_cl_syn_loader(self):
        if self._cur_task > 0:
            #初始化持续学习data
            cl_syn_dataset_list = []
            for t in range(self._cur_task):
                syn_dataset = SynLabelDataset(savePath + '/' + 'task_'+str(t), transform=train_trsf)
                cl_syn_dataset_list.append(syn_dataset)
            cl_syn_dataset = ConcatDataset(cl_syn_dataset_list),
            self.fl_syn_train_loader = DataLoader(
                ConcatDataset(cl_syn_dataset),
                batch_size=self.args["local_bs"],
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )

    def init_fl_syn_loader(self):
        #初始化联邦学习data
        fl_syn_dataset_list = []
        for t in range(self._cur_task+1):
            syn_dataset = SynLabelDataset(savePath + '/' + 'task_'+str(t), transform=train_trsf)
            fl_syn_dataset_list.append(syn_dataset)
        self.fl_syn_train_loader = DataLoader(
            ConcatDataset(fl_syn_dataset_list),
            batch_size=self.args["local_bs"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def _fl_train(self):
        if self._cur_task > 0:
            self.init_cl_syn_loader()
        self._network.cuda()
        prog_bar = tqdm(range(self.args["com_round"]))
        for _, com in enumerate(prog_bar):
            local_weights = []
            m = max(int(self.args["num_users"]), 1)
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
            for idx in idxs_users:
                train_net = deepcopy(self._network)
                train_net, w = self._local_update(train_net, self.local_train_loaders[idx])
                if self.fl_syn_train_loader is not None:
                    w = self._train_syn(train_net, self.fl_syn_train_loader)
                # if self.fl_syn_train_loader is not None:
                #     train_net, w = self._local_update(train_net, self.local_train_loaders[idx])
                #     w = self._train_syn(train_net, self.fl_syn_train_loader)
                # else:
                #     if self._cur_task == 0:
                #         train_net, w = self._local_update(train_net, self.local_train_loaders[idx])
                #     else:
                #         w = self._local_finetune(train_net, self.local_train_loaders[idx])
                local_weights.append(deepcopy(w))
            # update global weights
            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)
            # test
            test_acc = self._compute_accuracy(self._network, self.test_loader)
            info=("Task {}, Epoch {}/{} =>  Test_accy {:.2f}".format(
                self._cur_task, com + 1, self.args["com_round"], test_acc,))
            prog_bar.set_description(info)
            if com == self.args["com_round"]//2:
                self.diversity_cluster()
                self.init_syn_dataloader(self.local_train_loaders, 'train')
                self.init_syn_dataloader(self.local_test_loader, 'test')
                self.init_fl_syn_loader()
