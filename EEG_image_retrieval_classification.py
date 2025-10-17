# val accuracy, eeg to image and text class, add mse loss
import os
import time
import torch
import numpy as np
import pandas as pd

from loss import ClipLoss
import random
import itertools
import argparse
from model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Visual stimuli reconstruction with EEG")
parser.add_argument('--dnn', default='clip', type=str)
parser.add_argument('--data_path', default='/media/siat/disk1/BCI_data/THINGS-EEG/Preprocessed_data_250Hz', type=str)
parser.add_argument('--result_path', default='./results/' , type=str)
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate for model training')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--num_sub', default=10,type=int, help='the number of subjects used in the experiments')
parser.add_argument('--batch_size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--alpha', default=0.5, type=float, help='parameters for balancing EEG_image and EEG_text')
parser.add_argument('--beta', default=2, type=float, help='parameters for balancing contrastive loss and mse loss')
parser.add_argument('--seed', default=2024, type=int, help='seed for initializing training')
parser.add_argument('--logger', default=True, type=bool)
parser.add_argument('--insubject', default=True, type=bool, help='choice for in-subject and cross-subject')
parser.add_argument('--encoder_type', default='ATMS_classification_50', type=str)
parser.add_argument('--img_encoder', default='Proj_img', type=str)
parser.add_argument('--model_type', default='ViT-H-14', type=str)
parser.add_argument('--val', default='val_acc_retrieval_classification_eeg_to_img_and_text_class_alpha0.5_both_mse2', type=str)

class EEGToImage():
    def __init__(self, args, sub, nsub):
        super(EEGToImage, self).__init__()
        self.args = args
        self.lr = args.lr
        self.epochs = args.epochs
        self.data_path = args.data_path
        self.insubject = args.insubject
        self.sub = sub
        self.nsub = nsub
        self.batch_size = args.batch_size
        self.encoder_type = args.encoder_type
        self.alpha = args.alpha
        self.beta = args.beta
        self.model_type = args.model_type
        self.val = args.val
        self.result_path = args.result_path

        os.makedirs(self.result_path + f"/output/{self.encoder_type}/{self.val}/", exist_ok=True) 
        self.log_write = open(self.result_path + f"/output/{self.encoder_type}/{self.val}/log_subject%d.txt" % self.nsub, "w")

        self.b1 = 0.5
        self.b2 = 0.999
        self.eeg_model = ATMS_classification_50().to(device)
        self.eeg_img_proj = Proj_eeg_img().to(device)
        self.eeg_text_proj = Proj_eeg_text().to(device)

        self.criterion_mse = torch.nn.MSELoss().to(device)
        self.clip_loss = ClipLoss().to(device)

    def get_eeg_data(self):
        # load traindata and validation data
        n_classes = 1654  
        samples_per_class = 10
        label_list = []  
        file_name = 'preprocessed_eeg_training.npy'
        file_path = os.path.join(self.data_path, self.sub, file_name)
        data = np.load(file_path, allow_pickle=True)
        # （16540， 4， 63， 250）——>（16540， 63， 250）
        preprocessed_eeg_data = data['preprocessed_eeg_data']
        preprocessed_eeg_data = np.mean(preprocessed_eeg_data, axis=1)   
        train_data = torch.from_numpy(preprocessed_eeg_data).float().detach()

        for i in range(n_classes):
            labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()  
            label_list.append(labels)
        train_label = torch.cat(label_list, dim=0)

        # load test data
        file_name = 'preprocessed_eeg_test.npy'
        file_path = os.path.join(self.data_path, self.sub, file_name)
        data = np.load(file_path, allow_pickle=True)
        # (200, 80, 63, 250) ——> (200, 63, 250)
        preprocessed_eeg_data = data['preprocessed_eeg_data']
        preprocessed_eeg_data = np.mean(preprocessed_eeg_data, axis=1)
        test_data = torch.from_numpy(preprocessed_eeg_data).float().detach()
        test_label = torch.from_numpy(np.arange(200)).detach()
        print('the shape of train_data is:', train_data.shape)
        print('the shape of train_label is:', train_label.shape)
        print('the shape of test_data is:', test_data.shape)
        print('the shape of test_label is:', test_label.shape)
        return train_data, train_label, test_data, test_label

    def get_image_text_data(self):
        train_feature = torch.load(os.path.join(f'{self.model_type}_detail_class_features_train.pt'), weights_only=True)
        test_feature = torch.load(os.path.join(f'{self.model_type}_detail_class_features_test.pt'), weights_only=True)
        
        img_train_features = train_feature['img_features']
        img_test_features = test_feature['img_features']
        text_train_features = train_feature['text_features']
        text_test_features = test_feature['text_features']
        class_train_features = train_feature['class_features']
        class_test_features = test_feature['class_features']
        
        # class_train_features = class_train_features.unsqueeze(1).repeat(1, 10, 1)
        # class_train_features = class_train_features.reshape(-1, class_train_features.shape[2])
        print('The shape of img_train_features is:', img_train_features.shape)
        print('The shape of img_test_features is:', img_test_features.shape)
        print('The shape of text_train_features is:', text_train_features.shape)
        print('The shape of text_test_features is:', text_test_features.shape)
        print('The shape of class_train_features is:', class_train_features.shape)
        print('The shape of class_test_features is:', class_test_features.shape)
        # (16540, 1024), (200, 1024), (16540, 1024), (200, 1024), (16540, 1024), (200, 1024)
        return img_train_features, img_test_features, text_train_features, text_test_features, class_train_features, class_test_features

    def train(self):
        # current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
        self.eeg_model.apply(weights_init_normal)
        self.eeg_img_proj.apply(weights_init_normal)
        self.eeg_text_proj.apply(weights_init_normal)

        train_data, train_label, test_data, test_label = self.get_eeg_data()
        img_train_features, img_test_features, text_train_features, text_test_features, class_train_features, class_test_features = self.get_image_text_data()

        class_features_train_all = (class_train_features[::10]).to(device).float() # (n_cls, d)
        img_features_train_all = (img_train_features[::10]).to(device).float()
        print('-----------------------------------------------------------------------')
        print('The shape of image features template is: ', img_features_train_all.shape)
        print('The shape of class features template is: ', class_features_train_all.shape)

        # shuffle the training data
        train_shuffle = np.random.permutation(len(train_data))
        train_data = train_data[train_shuffle]
        train_label = train_label[train_shuffle]
        img_train_features = img_train_features[train_shuffle]
        text_train_features = text_train_features[train_shuffle]
        class_train_features = class_train_features[train_shuffle]

        val_data = train_data[:740]
        val_label = train_label[:740]
        val_img = img_train_features[:740]
        val_text = text_train_features[:740]
        val_class = class_train_features[:740]

        train_data = train_data[740:]
        train_label = train_label[740:]
        img_train_features = img_train_features[740:]
        text_train_features = text_train_features[740:]
        class_train_features = class_train_features[740:]

        train_dataset = torch.utils.data.TensorDataset(train_data, train_label, img_train_features, text_train_features, class_train_features)
        val_dataset = torch.utils.data.TensorDataset(val_data, val_label, val_img, val_text, val_class)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label, img_test_features, text_test_features, class_test_features)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        self.optimizer = torch.optim.Adam(itertools.chain(self.eeg_model.parameters(), self.eeg_img_proj.parameters(), self.eeg_text_proj.parameters()), lr=self.lr, betas=(self.b1, self.b2))

        # train_losses, train_accuracies = [], []
        # test_losses, test_accuracies = [], []
        # v2_accs = []
        # v4_accs = []
        # v10_accs = []

        best_accuracy = 0.0
        best_loss_val = np.inf
        best_val_retrieval_accuracy = 0.0
        best_val_classification_accuracy = 0.0
        best_model_weights = None
        best_epoch = 0
        results = []  # List to store results for each epoch

        for epoch in range(self.epochs):
            self.eeg_model.train()
            self.eeg_img_proj.train()
            self.eeg_text_proj.train()
            total_train_loss = 0
            total_train_img_loss = 0
            total_train_text_loss = 0
            total_train_img_mse_loss = 0
            total_train_text_mse_loss = 0
            train_retrieval_correct = 0
            train_classification_correct = 0
            total = 0
            for batch_idx, (eeg_data, labels, img_features, text_features, class_features) in enumerate(train_loader):
                eeg_data = eeg_data.to(device).float()
                labels = labels.to(device)
                class_features = class_features.to(device).float()
                text_features = text_features.to(device).float()
                img_features = img_features.to(device).float()
                
                logit_scale = self.eeg_model.logit_scale
                eeg_features = self.eeg_model(eeg_data)
                eeg_img_features = self.eeg_img_proj(eeg_features)
                eeg_text_features = self.eeg_text_proj(eeg_features)
                
                img_loss = self.clip_loss(eeg_img_features, img_features, logit_scale)
                text_loss = (self.clip_loss(eeg_text_features, class_features, logit_scale) + self.clip_loss(eeg_text_features, text_features, logit_scale)) / 2
                img_mse_loss = self.criterion_mse(eeg_img_features, img_features)
                # text_mse_loss = self.criterion_mse(eeg_text_features, text_features)
                text_mse_loss = (self.criterion_mse(eeg_text_features, class_features) + self.criterion_mse(eeg_text_features, text_features)) / 2
                loss = self.alpha * (img_loss + img_mse_loss * self.beta) + (1 - self.alpha) * (text_loss + text_mse_loss * self.beta)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_train_loss += loss.item()
                total_train_img_loss += img_loss.item()
                total_train_text_loss += text_loss.item()
                total_train_img_mse_loss += img_mse_loss.item()
                total_train_text_mse_loss += text_mse_loss.item()
                # logits = logit_scale * eeg_features @ text_features_all.T # (n_batch, n_cls)
                
                logits_img = logit_scale * eeg_img_features @ img_features_train_all.T
                logits_text = logit_scale * eeg_text_features @ class_features_train_all.T
                # logits_single = (logits_text + logits_img) / 2.0
                # logits_text = logit_scale * eeg_features @ text_features_all.T

                predicted_img = torch.argmax(logits_img, dim=1) # (n_batch, ) \in {0, 1, ..., n_cls-1}
                train_retrieval_correct += (predicted_img == labels).sum().item()
                predicted_text = torch.argmax(logits_text, dim=1) # (n_batch, ) \in {0, 1, ..., n_cls-1}
                train_classification_correct += (predicted_text == labels).sum().item()
                
                batch_size = predicted_img.shape[0]
                total += batch_size

            train_loss = total_train_loss / (batch_idx + 1)
            train_img_loss = total_train_img_loss / (batch_idx + 1)
            train_text_loss = total_train_text_loss / (batch_idx + 1)
            train_img_mse_loss = total_train_img_mse_loss / (batch_idx + 1)
            train_text_mse_loss = total_train_text_mse_loss / (batch_idx + 1)
            train_retrieval_accuracy = train_retrieval_correct / total
            train_classification_accuracy = train_classification_correct / total

            total_val_loss = 0
            total_val = 0
            val_retrieval_correct = 0
            val_classification_correct = 0
            if (epoch + 1) % 1 == 0:
                self.eeg_model.eval()
                self.eeg_img_proj.eval()
                self.eeg_text_proj.eval()
                with torch.no_grad():
                    # * validation part
                    for batch_idx, (val_eeg_data, val_labels, val_img_features, val_text_features, val_class_features) in enumerate(val_loader):
                        val_eeg_data = val_eeg_data.to(device).float()
                        val_class_features = val_class_features.to(device).float()
                        val_text_features = val_text_features.to(device).float()
                        val_img_features = val_img_features.to(device).float()
                        val_labels = val_labels.to(device)
                        
                        val_eeg_features = self.eeg_model(val_eeg_data)
                        val_eeg_img_features = self.eeg_img_proj(val_eeg_features)
                        val_eeg_text_features = self.eeg_text_proj(val_eeg_features)
                        
                        logit_scale = self.eeg_model.logit_scale
                        val_img_loss = self.clip_loss(val_eeg_img_features, val_img_features, logit_scale)
                        val_text_loss = (self.clip_loss(val_eeg_text_features, val_class_features, logit_scale) + self.clip_loss(val_eeg_text_features, val_text_features, logit_scale)) / 2
                        val_img_mse_loss = self.criterion_mse(val_eeg_img_features, val_img_features)
                        # val_text_mse_loss = self.criterion_mse(val_eeg_text_features, val_text_features)
                        val_text_mse_loss = (self.criterion_mse(val_eeg_text_features, val_class_features) + self.criterion_mse(val_eeg_text_features, val_text_features)) / 2
                        loss = self.alpha * (val_img_loss + val_img_mse_loss * self.beta) + (1 - self.alpha) * (val_text_loss + val_text_mse_loss * self.beta)
                        total_val_loss += loss.item()
                        # logits = logit_scale * eeg_features @ text_features_all.T # (n_batch, n_cls)
                        
                        logits_img = logit_scale * val_eeg_img_features @ img_features_train_all.T
                        logits_text = logit_scale * val_eeg_text_features @ class_features_train_all.T

                        val_predicted_img = torch.argmax(logits_img, dim=1) # (n_batch, ) \in {0, 1, ..., n_cls-1}
                        val_retrieval_correct += (val_predicted_img == val_labels).sum().item()
                        val_predicted_text = torch.argmax(logits_text, dim=1) # (n_batch, ) \in {0, 1, ..., n_cls-1}
                        val_classification_correct += (val_predicted_text == val_labels).sum().item()

                        batch_size = val_predicted_img.shape[0]
                        total_val += batch_size
                        
                val_loss = total_val_loss / (batch_idx + 1)
                val_retrieval_accuracy = val_retrieval_correct / total_val
                val_classification_accuracy = val_classification_correct / total_val

                if val_retrieval_accuracy > best_val_retrieval_accuracy and val_classification_accuracy > best_val_classification_accuracy:
                    # best_loss_val = val_loss
                    best_val_retrieval_accuracy = val_retrieval_accuracy
                    best_val_classification_accuracy = val_classification_accuracy
                    best_epoch = epoch + 1
                    os.makedirs(f"./results/models/{self.encoder_type}/{self.val}/{self.sub}", exist_ok=True)             
                    torch.save(self.eeg_model.state_dict(), f"./results/models/{self.encoder_type}/{self.val}/{self.sub}/eeg_model.pth")
                    torch.save(self.eeg_img_proj.state_dict(), f"./results/models/{self.encoder_type}/{self.val}/{self.sub}/eeg_img_proj.pth") 
                    torch.save(self.eeg_text_proj.state_dict(), f"./results/models/{self.encoder_type}/{self.val}/{self.sub}/eeg_text_proj.pth")        
                    print(f"models have been saved !")
                print('Epoch:', epoch + 1,
                    '  Train loss: %.4f' % train_loss,
                    '  Train image loss: %.4f' % train_img_loss,
                    '  Train text loss: %.4f' % train_text_loss,
                    '  Train image mse loss: %.4f' % train_img_mse_loss,
                    '  Train text mse loss: %.4f' % train_text_mse_loss,
                    '  Train retrieval accuracy: %.4f' % train_retrieval_accuracy,
                    '  Train classification accuracy: %.4f' % train_classification_accuracy,
                    '  Validation loss: %.4f' % val_loss,
                    '  Validation retrieval accuracy: %.4f' % val_retrieval_accuracy,
                    '  Validation classification accuracy: %.4f' % val_classification_accuracy,
                    )
                self.log_write.write('Epoch %d: Train loss: %.4f, Train retrieval accuracy: %.4f, Train classification accuracy: %.4f, Validation loss: %.4f, Validation retrieval accuracy: %.4f, Validation classification accuracy: %.4f\n'%(epoch+1, train_loss, train_retrieval_accuracy, train_classification_accuracy, val_loss, val_retrieval_accuracy, val_classification_accuracy))

        print('The best epoch is: %d\n'% best_epoch)
        self.log_write.write('The best epoch is: %d\n'% best_epoch)

        test_loss, retrieval_test_accuracy, retrieval_top5_acc = self.evaluate_model_retrieval(test_loader, class_test_features, img_test_features, k=200)
        _, retrieval_v2_acc, _ = self.evaluate_model_retrieval(test_loader, class_test_features, img_test_features, k = 2)
        _, retrieval_v4_acc, _ = self.evaluate_model_retrieval(test_loader, class_test_features, img_test_features, k = 4)
        _, retrieval_v10_acc, _ = self.evaluate_model_retrieval(test_loader, class_test_features, img_test_features, k = 10)
        _, retrieval_v50_acc, retrieval_v50_top5_acc = self.evaluate_model_retrieval(test_loader, class_test_features, img_test_features,  k=50)
        _, retrieval_v100_acc, retrieval_v100_top5_acc = self.evaluate_model_retrieval(test_loader, class_test_features, img_test_features,  k=100)
        
        _, classification_test_accuracy, classification_top5_acc = self.evaluate_model_classification(test_loader, class_test_features, img_test_features, k=200)
        _, classification_v2_acc, _ = self.evaluate_model_classification(test_loader, class_test_features, img_test_features, k = 2)
        _, classification_v4_acc, _ = self.evaluate_model_classification(test_loader, class_test_features, img_test_features, k = 4)
        _, classification_v10_acc, _ = self.evaluate_model_classification(test_loader, class_test_features, img_test_features, k = 10)
        _, classification_v50_acc, classification_v50_top5_acc = self.evaluate_model_classification(test_loader, class_test_features, img_test_features,  k=50)
        _, classification_v100_acc, classification_v100_top5_acc = self.evaluate_model_classification(test_loader, class_test_features, img_test_features,  k=100)
        
        # test_losses.append(test_loss)
        # test_accuracies.append(test_accuracy)
        # v2_accs.append(v2_acc)
        # v4_accs.append(v4_acc)
        # v10_accs.append(v10_acc)

        results = {
        "epoch": best_epoch,
        "train_loss": train_loss,
        "train_retrieval_accuracy": train_retrieval_accuracy,
        "train_classification_accuracy": train_classification_accuracy,
        "test_loss": test_loss,
        "retrieval_test_accuracy": retrieval_test_accuracy,
        "retrieval_top5_acc": retrieval_top5_acc,
        "retrieval_v2_acc": retrieval_v2_acc,
        "retrieval_v4_acc": retrieval_v4_acc,
        "retrieval_v10_acc": retrieval_v10_acc,
        "retrieval_v50_acc": retrieval_v50_acc,
        "retrieval_v100_acc": retrieval_v100_acc,
        "retrieval_v50_top5_acc": retrieval_v50_top5_acc,
        "retrieval_v100_top5_acc": retrieval_v100_top5_acc,
        "classification_test_accuracy": classification_test_accuracy,
        "classification_top5_acc": classification_top5_acc,
        "classification_v2_acc": classification_v2_acc,
        "classification_v4_acc": classification_v4_acc,
        "classification_v10_acc": classification_v10_acc,
        "classification_v50_acc": classification_v50_acc,
        "classification_v100_acc": classification_v100_acc,
        "classification_v50_top5_acc": classification_v50_top5_acc,
        "classification_v100_top5_acc": classification_v100_top5_acc,
        }
        self.log_write.write('Best epoch %d: retrieval_test_accuracy: %.4f, retrieval_top5_acc: %.4f, retrieval_v2_acc: %.4f, retrieval_v4_acc: %.4f, retrieval_v10_acc: %.4f, retrieval_v50_acc: %.4f, retrieval_v100_acc: %.4f, retrieval_v50_top5_acc: %.4f, retrieval_v100_top5_acc: %.4f\n'%(best_epoch, retrieval_test_accuracy, retrieval_top5_acc, retrieval_v2_acc, retrieval_v4_acc, retrieval_v10_acc, retrieval_v50_acc, retrieval_v100_acc, retrieval_v50_top5_acc, retrieval_v100_top5_acc))
        self.log_write.write('classification_test_accuracy: %.4f, classification_top5_acc: %.4f, classification_v2_acc: %.4f, classification_v4_acc: %.4f, classification_v10_acc: %.4f, classification_v50_acc: %.4f, classification_v100_acc: %.4f, classification_v50_top5_acc: %.4f, classification_v100_top5_acc: %.4f\n'%(classification_test_accuracy, classification_top5_acc, classification_v2_acc, classification_v4_acc, classification_v10_acc, classification_v50_acc, classification_v100_acc, classification_v50_top5_acc, classification_v100_top5_acc))
        print(f"Train Loss: {train_loss:.4f}, Train Retrieval Accuracy: {train_retrieval_accuracy:.4f}, Train Classification Accuracy: {train_classification_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Retrieval Accuracy: {retrieval_test_accuracy:.4f}, Retrieval Top5 Accuracy: {retrieval_top5_acc:.4f}, Test Classification Accuracy: {classification_test_accuracy:.4f}, Classification Top5 Accuracy: {classification_top5_acc:.4f}")
        # print(f"Epoch {epoch + 1}/{self.epochs} - v2 Accuracy:{v2_acc} - v4 Accuracy:{v4_acc} - v10 Accuracy:{v10_acc} - v50 Accuracy:{v50_acc} - v50_top5_acc: {v50_top5_acc:.4f} - v100 Accuracy:{v100_acc} - v100_top5_acc{v100_top5_acc}")
        # # Save results to a CSV file
        # results_dir = f"./results/outputs/{self.encoder_type}/{self.sub}/{current_time}"
        # os.makedirs(results_dir, exist_ok=True)          
        # results_file = f"{results_dir}/{self.encoder_type}_{self.sub}.csv"
        
        # with open(results_file, 'w', newline='') as file:
        #     writer = csv.DictWriter(file, fieldnames=results.keys())
        #     writer.writeheader()
        #     writer.writerows(results)
        # print(f'Results saved to {results_file}')
        return results
            
    def evaluate_model_retrieval(self, dataloader, text_features_all, img_features_all, k):
        self.eeg_model.load_state_dict(torch.load(f"./results/models/{self.encoder_type}/{self.val}/{self.sub}/eeg_model.pth", weights_only=True), strict=False)
        self.eeg_img_proj.load_state_dict(torch.load(f"./results/models/{self.encoder_type}/{self.val}/{self.sub}/eeg_img_proj.pth", weights_only=True), strict=False)
        self.eeg_text_proj.load_state_dict(torch.load(f"./results/models/{self.encoder_type}/{self.val}/{self.sub}/eeg_text_proj.pth", weights_only=True), strict=False)
        self.eeg_model.eval()
        self.eeg_img_proj.eval()
        self.eeg_text_proj.eval()
        text_features_all = text_features_all.to(device).float()
        img_features_all = img_features_all.to(device).float()
        total_loss = 0
        correct = 0
        total = 0
        top5_correct = 0
        top5_correct_count = 0
        
        all_labels = set(range(text_features_all.size(0)))
        top5_acc = 0
        with torch.no_grad():
            for batch_idx, (eeg_data, labels, img_features, text_features, class_features) in enumerate(dataloader):
                eeg_data = eeg_data.to(device).float()
                labels = labels.to(device)
                img_features = img_features.to(device).float()
                text_features = text_features.to(device).float()
                class_features = class_features.to(device).float()

                eeg_features = self.eeg_model(eeg_data)
                eeg_img_features = self.eeg_img_proj(eeg_features)
                eeg_text_features = self.eeg_text_proj(eeg_features)
                
                logit_scale = self.eeg_model.logit_scale
                img_loss = self.clip_loss(eeg_img_features, img_features, logit_scale)
                text_loss = (self.clip_loss(eeg_text_features, class_features, logit_scale) + self.clip_loss(eeg_text_features, text_features, logit_scale)) / 2
                img_mse_loss = self.criterion_mse(eeg_img_features, img_features)
                # text_mse_loss = self.criterion_mse(eeg_text_features, text_features)
                text_mse_loss = (self.criterion_mse(eeg_text_features, class_features) + self.criterion_mse(eeg_text_features, text_features)) / 2
                loss = self.alpha * (img_loss + img_mse_loss * self.beta) + (1 - self.alpha) * (text_loss + text_mse_loss * self.beta)
                # loss = self.alpha * img_loss + (1 - self.alpha) * text_loss
                
                total_loss += loss.item()
                
                for idx, label in enumerate(labels):
                    
                    possible_classes = list(all_labels - {label.item()})
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                    selected_img_features = img_features_all[selected_classes]
                    selected_text_features = text_features_all[selected_classes]
                    if k==200:
                        
                        # logits_text = logit_scale * eeg_features[idx] @ selected_text_features.T                    
                        logits_img = logit_scale * eeg_img_features[idx] @ selected_img_features.T
                        logits_single = logits_img
                        # print("logits_single", logits_single.shape)
                        
                        # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                        predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) \in {0, 1, ..., n_cls-1}
                        if predicted_label == label.item():
                            # print("predicted_label", predicted_label)
                            correct += 1
                        
                        # print("logits_single", logits_single)
                        _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                            
                        
                        if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                            top5_correct_count+=1                                
                        total += 1
                    elif k == 50 or k == 100:
                        
                        selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                        # logits_text = logit_scale * eeg_features[idx] @ selected_text_features.T
                        logits_img = logit_scale * eeg_img_features[idx] @ selected_img_features.T
                        logits_single = logits_img
                        
                        predicted_label = selected_classes[torch.argmax(logits_single).item()]
                        if predicted_label == label.item():
                            correct += 1
                        _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                            
                        
                        if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                            top5_correct_count+=1                                
                        total += 1

                    elif k==2 or k==4 or k==10:
                        selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                        
                        logits_img = logit_scale * eeg_img_features[idx] @ selected_img_features.T
                        # logits_text = logit_scale * eeg_features[idx] @ selected_text_features.T
                        # logits_single = (logits_text + logits_img) / 2.0
                        logits_single = logits_img
                        # print("logits_single", logits_single.shape)
                        
                        # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                        predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) \in {0, 1, ..., n_cls-1}
                        if predicted_label == label.item():
                            correct += 1
                        total += 1
                    else:
                        print("Error.")
                        
        average_loss = total_loss / (batch_idx+1)
        accuracy = correct / total
        top5_acc = top5_correct_count / total
        return average_loss, accuracy, top5_acc
    
    def evaluate_model_classification(self, dataloader, text_features_all, img_features_all, k):
        self.eeg_model.load_state_dict(torch.load(f"./results/models/{self.encoder_type}/{self.val}/{self.sub}/eeg_model.pth", weights_only=True), strict=False)
        self.eeg_img_proj.load_state_dict(torch.load(f"./results/models/{self.encoder_type}/{self.val}/{self.sub}/eeg_img_proj.pth", weights_only=True), strict=False)
        self.eeg_text_proj.load_state_dict(torch.load(f"./results/models/{self.encoder_type}/{self.val}/{self.sub}/eeg_text_proj.pth", weights_only=True), strict=False)
        self.eeg_model.eval()
        self.eeg_img_proj.eval()
        self.eeg_text_proj.eval()
        text_features_all = text_features_all.to(device).float()
        img_features_all = img_features_all.to(device).float()
        total_loss = 0
        correct = 0
        total = 0
        top5_correct = 0
        top5_correct_count = 0
        
        all_labels = set(range(text_features_all.size(0)))
        top5_acc = 0
        with torch.no_grad():
            for batch_idx, (eeg_data, labels, img_features, text_features, class_features) in enumerate(dataloader):
                eeg_data = eeg_data.to(device).float()
                labels = labels.to(device)
                img_features = img_features.to(device).float()
                text_features = text_features.to(device).float()
                class_features = class_features.to(device).float()

                eeg_features = self.eeg_model(eeg_data)
                eeg_img_features = self.eeg_img_proj(eeg_features)
                eeg_text_features = self.eeg_text_proj(eeg_features)
                
                logit_scale = self.eeg_model.logit_scale
                img_loss = self.clip_loss(eeg_img_features, img_features, logit_scale)
                text_loss = (self.clip_loss(eeg_text_features, class_features, logit_scale) + self.clip_loss(eeg_text_features, text_features, logit_scale)) / 2
                img_mse_loss = self.criterion_mse(eeg_img_features, img_features)
                # text_mse_loss = self.criterion_mse(eeg_text_features, text_features)
                text_mse_loss = (self.criterion_mse(eeg_text_features, class_features) + self.criterion_mse(eeg_text_features, text_features)) / 2
                loss = self.alpha * (img_loss + img_mse_loss * self.beta) + (1 - self.alpha) * (text_loss + text_mse_loss * self.beta)
                # loss = self.alpha * img_loss + (1 - self.alpha) * text_loss
                
                total_loss += loss.item()
                
                for idx, label in enumerate(labels):
                    
                    possible_classes = list(all_labels - {label.item()})
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                    selected_img_features = img_features_all[selected_classes]
                    selected_text_features = text_features_all[selected_classes]
                    if k==200:
                        
                        logits_text = logit_scale * eeg_text_features[idx] @ selected_text_features.T                    
                        # logits_img = logit_scale * eeg_img_features[idx] @ selected_img_features.T
                        logits_single = logits_text
                        # print("logits_single", logits_single.shape)
                        
                        # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                        predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) \in {0, 1, ..., n_cls-1}
                        if predicted_label == label.item():
                            # print("predicted_label", predicted_label)
                            correct += 1
                        
                        # print("logits_single", logits_single)
                        _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                            
                        
                        if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                            top5_correct_count+=1                                
                        total += 1
                    elif k == 50 or k == 100:
                        
                        selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                        logits_text = logit_scale * eeg_text_features[idx] @ selected_text_features.T
                        # logits_img = logit_scale * eeg_img_features[idx] @ selected_img_features.T
                        logits_single = logits_text
                        
                        predicted_label = selected_classes[torch.argmax(logits_single).item()]
                        if predicted_label == label.item():
                            correct += 1
                        _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                            
                        
                        if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                            top5_correct_count+=1                                
                        total += 1

                    elif k==2 or k==4 or k==10:
                        selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                        
                        # logits_img = logit_scale * eeg_img_features[idx] @ selected_img_features.T
                        logits_text = logit_scale * eeg_text_features[idx] @ selected_text_features.T
                        # logits_single = (logits_text + logits_img) / 2.0
                        logits_single = logits_text
                        # print("logits_single", logits_single.shape)
                        
                        # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                        predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) \in {0, 1, ..., n_cls-1}
                        if predicted_label == label.item():
                            correct += 1
                        total += 1
                    else:
                        print("Error.")
                        
        average_loss = total_loss / (batch_idx+1)
        accuracy = correct / total
        top5_acc = top5_correct_count / total
        return average_loss, accuracy, top5_acc

def main(): 
    args = parser.parse_args()
    # seed_n = args.seed
    seed_n = np.random.randint(args.seed)

    print('seed is ' + str(seed_n))
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)  # if using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    num_sub = args.num_sub
    cal_num = 0
    retrieval_aver_acc = []
    retrieval_aver_top5 = []
    retrieval_aver_v2 = []
    retrieval_aver_v4 = []
    retrieval_aver_v10 = []
    retrieval_aver_v50 = []
    retrieval_aver_v100 = []
    retrieval_aver_v50_top5 = []
    retrieval_aver_v100_top5 = []
    
    classification_aver_acc = []
    classification_aver_top5 = []
    classification_aver_v2 = []
    classification_aver_v4 = []
    classification_aver_v10 = []
    classification_aver_v50 = []
    classification_aver_v100 = []
    classification_aver_v50_top5 = []
    classification_aver_v100_top5 = []

    subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'] 

    for i in range(num_sub):
        sub = subjects[i]
        cal_num += 1      
        eeg2image = EEGToImage(args, sub, i+1)
        results = eeg2image.train()
        retrieval_aver_acc.append(results['retrieval_test_accuracy'])
        retrieval_aver_top5.append(results['retrieval_top5_acc'])
        retrieval_aver_v2.append(results['retrieval_v2_acc'])
        retrieval_aver_v4.append(results['retrieval_v4_acc'])
        retrieval_aver_v10.append(results['retrieval_v10_acc'])
        retrieval_aver_v50.append(results['retrieval_v50_acc'])
        retrieval_aver_v100.append(results['retrieval_v100_acc'])
        retrieval_aver_v50_top5.append(results['retrieval_v50_top5_acc'])
        retrieval_aver_v100_top5.append(results['retrieval_v100_top5_acc'])
        
        classification_aver_acc.append(results['classification_test_accuracy'])
        classification_aver_top5.append(results['classification_top5_acc'])
        classification_aver_v2.append(results['classification_v2_acc'])
        classification_aver_v4.append(results['classification_v4_acc'])
        classification_aver_v10.append(results['classification_v10_acc'])
        classification_aver_v50.append(results['classification_v50_acc'])
        classification_aver_v100.append(results['classification_v100_acc'])
        classification_aver_v50_top5.append(results['classification_v50_top5_acc'])
        classification_aver_v100_top5.append(results['classification_v100_top5_acc'])
    
    retrieval_aver_acc.append(np.mean(retrieval_aver_acc))
    retrieval_aver_top5.append(np.mean(retrieval_aver_top5))
    retrieval_aver_v2.append(np.mean(retrieval_aver_v2))
    retrieval_aver_v4.append(np.mean(retrieval_aver_v4))
    retrieval_aver_v10.append(np.mean(retrieval_aver_v10))
    retrieval_aver_v50.append(np.mean(retrieval_aver_v50))
    retrieval_aver_v100.append(np.mean(retrieval_aver_v100))
    retrieval_aver_v50_top5.append(np.mean(retrieval_aver_v50_top5))
    retrieval_aver_v100_top5.append(np.mean(retrieval_aver_v100_top5))
    
    classification_aver_acc.append(np.mean(classification_aver_acc))
    classification_aver_top5.append(np.mean(classification_aver_top5))
    classification_aver_v2.append(np.mean(classification_aver_v2))
    classification_aver_v4.append(np.mean(classification_aver_v4))
    classification_aver_v10.append(np.mean(classification_aver_v10))
    classification_aver_v50.append(np.mean(classification_aver_v50))
    classification_aver_v100.append(np.mean(classification_aver_v100))
    classification_aver_v50_top5.append(np.mean(classification_aver_v50_top5))
    classification_aver_v100_top5.append(np.mean(classification_aver_v100_top5))
    column = np.arange(1, cal_num+1).tolist()
    column.append('ave')
    pd_all = pd.DataFrame(columns=column, data=[retrieval_aver_acc, retrieval_aver_top5, retrieval_aver_v2, retrieval_aver_v4, retrieval_aver_v10, retrieval_aver_v50, retrieval_aver_v100, retrieval_aver_v50_top5, retrieval_aver_v100_top5, classification_aver_acc, classification_aver_top5, classification_aver_v2, classification_aver_v4, classification_aver_v10, classification_aver_v50, classification_aver_v100, classification_aver_v50_top5, classification_aver_v100_top5])
    pd_all.to_csv(args.result_path + f'/output/{args.encoder_type}/{args.val}/result.csv')

if __name__ == '__main__':
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
    

# ##################################################################################################################################
# # val accuracy, eeg to image and text class
# # val accuracy
# import os
# import time
# import torch
# import numpy as np
# import pandas as pd

# from loss import ClipLoss
# import random
# import itertools
# import argparse
# from model import *

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# parser = argparse.ArgumentParser(description="Visual stimuli reconstruction with EEG")
# parser.add_argument('--dnn', default='clip', type=str)
# parser.add_argument('--data_path', default='/media/siat/disk1/BCI_data/THINGS-EEG/Preprocessed_data_250Hz', type=str)
# parser.add_argument('--result_path', default='./results/' , type=str)
# parser.add_argument('--lr', default=2e-4, type=float, help='learning rate for model training')
# parser.add_argument('--epochs', default=100, type=int)
# parser.add_argument('--num_sub', default=10,type=int, help='the number of subjects used in the experiments')
# parser.add_argument('--batch_size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
# parser.add_argument('--alpha', default=0.9, type=float, help='parameters for balancing EEG_image and EEG_text')
# parser.add_argument('--seed', default=2024, type=int, help='seed for initializing training')
# parser.add_argument('--logger', default=True, type=bool)
# parser.add_argument('--insubject', default=True, type=bool, help='choice for in-subject and cross-subject')
# parser.add_argument('--encoder_type', default='ATMS_classification_50', type=str)
# parser.add_argument('--img_encoder', default='Proj_img', type=str)
# parser.add_argument('--model_type', default='ViT-H-14', type=str)
# parser.add_argument('--val', default='val_acc_retrieval_classification_eeg_to_img_and_text_class_alpha0.9', type=str)

# class EEGToImage():
#     def __init__(self, args, sub, nsub):
#         super(EEGToImage, self).__init__()
#         self.args = args
#         self.lr = args.lr
#         self.epochs = args.epochs
#         self.data_path = args.data_path
#         self.insubject = args.insubject
#         self.sub = sub
#         self.nsub = nsub
#         self.batch_size = args.batch_size
#         self.encoder_type = args.encoder_type
#         self.alpha = args.alpha
#         self.model_type = args.model_type
#         self.val = args.val
#         self.result_path = args.result_path

#         os.makedirs(self.result_path + f"/output/{self.encoder_type}/{self.val}/", exist_ok=True) 
#         self.log_write = open(self.result_path + f"/output/{self.encoder_type}/{self.val}/log_subject%d.txt" % self.nsub, "w")

#         self.b1 = 0.5
#         self.b2 = 0.999
#         self.eeg_model = ATMS_classification_50().to(device)
#         self.eeg_img_proj = Proj_eeg_img().to(device)
#         self.eeg_text_proj = Proj_eeg_text().to(device)

#         self.criterion_mse = torch.nn.MSELoss().to(device)
#         self.clip_loss = ClipLoss().to(device)

#     def get_eeg_data(self):
#         # load traindata and validation data
#         n_classes = 1654
#         samples_per_class = 10
#         label_list = []
#         file_name = 'preprocessed_eeg_training.npy'
#         file_path = os.path.join(self.data_path, self.sub, file_name)
#         data = np.load(file_path, allow_pickle=True)
#         # （16540， 4， 63， 250）——>（16540， 63， 250）
#         preprocessed_eeg_data = data['preprocessed_eeg_data']
#         preprocessed_eeg_data = np.mean(preprocessed_eeg_data, axis=1)   
#         train_data = torch.from_numpy(preprocessed_eeg_data).float().detach()

#         for i in range(n_classes):
#             labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()  
#             label_list.append(labels)
#         train_label = torch.cat(label_list, dim=0)

#         # load test data
#         file_name = 'preprocessed_eeg_test.npy'
#         file_path = os.path.join(self.data_path, self.sub, file_name)
#         data = np.load(file_path, allow_pickle=True)
#         # (200, 80, 63, 250) ——> (200, 63, 250)
#         preprocessed_eeg_data = data['preprocessed_eeg_data']
#         preprocessed_eeg_data = np.mean(preprocessed_eeg_data, axis=1)
#         test_data = torch.from_numpy(preprocessed_eeg_data).float().detach()
#         test_label = torch.from_numpy(np.arange(200)).detach()
#         print('the shape of train_data is:', train_data.shape)
#         print('the shape of train_label is:', train_label.shape)
#         print('the shape of test_data is:', test_data.shape)
#         print('the shape of test_label is:', test_label.shape)
#         return train_data, train_label, test_data, test_label

#     def get_image_text_data(self):
#         train_feature = torch.load(os.path.join(f'{self.model_type}_detail_class_features_train.pt'), weights_only=True)
#         test_feature = torch.load(os.path.join(f'{self.model_type}_detail_class_features_test.pt'), weights_only=True)
        
#         img_train_features = train_feature['img_features']
#         img_test_features = test_feature['img_features']
#         text_train_features = train_feature['text_features']
#         text_test_features = test_feature['text_features']
#         class_train_features = train_feature['class_features']
#         class_test_features = test_feature['class_features']
        
#         # class_train_features = class_train_features.unsqueeze(1).repeat(1, 10, 1)
#         # class_train_features = class_train_features.reshape(-1, class_train_features.shape[2])
#         print('The shape of img_train_features is:', img_train_features.shape)
#         print('The shape of img_test_features is:', img_test_features.shape)
#         print('The shape of text_train_features is:', text_train_features.shape)
#         print('The shape of text_test_features is:', text_test_features.shape)
#         print('The shape of class_train_features is:', class_train_features.shape)
#         print('The shape of class_test_features is:', class_test_features.shape)
#         # (16540, 1024), (200, 1024), (16540, 1024), (200, 1024), (16540, 1024), (200, 1024)
#         return img_train_features, img_test_features, text_train_features, text_test_features, class_train_features, class_test_features

#     def train(self):
#         # current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
#         self.eeg_model.apply(weights_init_normal)
#         self.eeg_img_proj.apply(weights_init_normal)
#         self.eeg_text_proj.apply(weights_init_normal)

#         train_data, train_label, test_data, test_label = self.get_eeg_data()
#         img_train_features, img_test_features, text_train_features, text_test_features, class_train_features, class_test_features = self.get_image_text_data()

#         class_features_train_all = (class_train_features[::10]).to(device).float() # (n_cls, d)
#         img_features_train_all = (img_train_features[::10]).to(device).float()
#         print('-----------------------------------------------------------------------')
#         print('The shape of image features template is: ', img_features_train_all.shape)
#         print('The shape of class features template is: ', class_features_train_all.shape)

#         # shuffle the training data
#         train_shuffle = np.random.permutation(len(train_data))
#         train_data = train_data[train_shuffle]
#         train_label = train_label[train_shuffle]
#         img_train_features = img_train_features[train_shuffle]
#         text_train_features = text_train_features[train_shuffle]
#         class_train_features = class_train_features[train_shuffle]

#         val_data = train_data[:740]
#         val_label = train_label[:740]
#         val_img = img_train_features[:740]
#         val_text = text_train_features[:740]
#         val_class = class_train_features[:740]

#         train_data = train_data[740:]
#         train_label = train_label[740:]
#         img_train_features = img_train_features[740:]
#         text_train_features = text_train_features[740:]
#         class_train_features = class_train_features[740:]

#         train_dataset = torch.utils.data.TensorDataset(train_data, train_label, img_train_features, text_train_features, class_train_features)
#         val_dataset = torch.utils.data.TensorDataset(val_data, val_label, val_img, val_text, val_class)
#         test_dataset = torch.utils.data.TensorDataset(test_data, test_label, img_test_features, text_test_features, class_test_features)

#         train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
#         val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=True)
#         test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

#         self.optimizer = torch.optim.Adam(itertools.chain(self.eeg_model.parameters(), self.eeg_img_proj.parameters(), self.eeg_text_proj.parameters()), lr=self.lr, betas=(self.b1, self.b2))

#         # train_losses, train_accuracies = [], []
#         # test_losses, test_accuracies = [], []
#         # v2_accs = []
#         # v4_accs = []
#         # v10_accs = []

#         best_accuracy = 0.0
#         best_loss_val = np.inf
#         best_val_retrieval_accuracy = 0.0
#         best_val_classification_accuracy = 0.0
#         best_model_weights = None
#         best_epoch = 0
#         results = []  # List to store results for each epoch

#         for epoch in range(self.epochs):
#             self.eeg_model.train()
#             self.eeg_img_proj.train()
#             self.eeg_text_proj.train()
#             total_train_loss = 0
#             total_train_img_loss = 0
#             total_train_text_loss = 0
#             train_retrieval_correct = 0
#             train_classification_correct = 0
#             total = 0
#             for batch_idx, (eeg_data, labels, img_features, text_features, class_features) in enumerate(train_loader):
#                 eeg_data = eeg_data.to(device).float()
#                 labels = labels.to(device)
#                 class_features = class_features.to(device).float()
#                 text_features = text_features.to(device).float()
#                 img_features = img_features.to(device).float()
                
#                 logit_scale = self.eeg_model.logit_scale
#                 eeg_features = self.eeg_model(eeg_data)
#                 eeg_img_features = self.eeg_img_proj(eeg_features)
#                 eeg_text_features = self.eeg_text_proj(eeg_features)
                
#                 img_loss = self.clip_loss(eeg_img_features, img_features, logit_scale)
#                 text_loss = (self.clip_loss(eeg_text_features, class_features, logit_scale) + self.clip_loss(eeg_text_features, text_features, logit_scale)) / 2
#                 loss = self.alpha * img_loss + (1 - self.alpha) * text_loss
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
                
#                 total_train_loss += loss.item()
#                 total_train_img_loss += img_loss.item()
#                 total_train_text_loss += text_loss.item()
#                 # logits = logit_scale * eeg_features @ text_features_all.T # (n_batch, n_cls)
                
#                 logits_img = logit_scale * eeg_img_features @ img_features_train_all.T
#                 logits_text = logit_scale * eeg_text_features @ class_features_train_all.T
#                 # logits_single = (logits_text + logits_img) / 2.0
#                 # logits_text = logit_scale * eeg_features @ text_features_all.T

#                 predicted_img = torch.argmax(logits_img, dim=1) # (n_batch, ) \in {0, 1, ..., n_cls-1}
#                 train_retrieval_correct += (predicted_img == labels).sum().item()
#                 predicted_text = torch.argmax(logits_text, dim=1) # (n_batch, ) \in {0, 1, ..., n_cls-1}
#                 train_classification_correct += (predicted_text == labels).sum().item()
                
#                 batch_size = predicted_img.shape[0]
#                 total += batch_size

#             train_loss = total_train_loss / (batch_idx + 1)
#             train_img_loss = total_train_img_loss / (batch_idx + 1)
#             train_text_loss = total_train_text_loss / (batch_idx + 1)
#             train_retrieval_accuracy = train_retrieval_correct / total
#             train_classification_accuracy = train_classification_correct / total

#             total_val_loss = 0
#             total_val = 0
#             val_retrieval_correct = 0
#             val_classification_correct = 0
#             if (epoch + 1) % 1 == 0:
#                 self.eeg_model.eval()
#                 self.eeg_img_proj.eval()
#                 self.eeg_text_proj.eval()
#                 with torch.no_grad():
#                     # * validation part
#                     for batch_idx, (val_eeg_data, val_labels, val_img_features, val_text_features, val_class_features) in enumerate(val_loader):
#                         val_eeg_data = val_eeg_data.to(device).float()
#                         val_class_features = val_class_features.to(device).float()
#                         val_text_features = val_text_features.to(device).float()
#                         val_img_features = val_img_features.to(device).float()
#                         val_labels = val_labels.to(device)
                        
#                         val_eeg_features = self.eeg_model(val_eeg_data)
#                         val_eeg_img_features = self.eeg_img_proj(val_eeg_features)
#                         val_eeg_text_features = self.eeg_text_proj(val_eeg_features)
                        
#                         logit_scale = self.eeg_model.logit_scale
#                         val_img_loss = self.clip_loss(val_eeg_img_features, val_img_features, logit_scale)
#                         val_text_loss = (self.clip_loss(val_eeg_text_features, val_class_features, logit_scale) + self.clip_loss(val_eeg_text_features, val_text_features, logit_scale)) / 2

#                         loss = self.alpha * val_img_loss + (1 - self.alpha) * val_text_loss
#                         total_val_loss += loss.item()
#                         # logits = logit_scale * eeg_features @ text_features_all.T # (n_batch, n_cls)
                        
#                         logits_img = logit_scale * val_eeg_img_features @ img_features_train_all.T
#                         logits_text = logit_scale * val_eeg_text_features @ class_features_train_all.T

#                         val_predicted_img = torch.argmax(logits_img, dim=1) # (n_batch, ) \in {0, 1, ..., n_cls-1}
#                         val_retrieval_correct += (val_predicted_img == val_labels).sum().item()
#                         val_predicted_text = torch.argmax(logits_text, dim=1) # (n_batch, ) \in {0, 1, ..., n_cls-1}
#                         val_classification_correct += (val_predicted_text == val_labels).sum().item()

#                         batch_size = val_predicted_img.shape[0]
#                         total_val += batch_size
                        
#                 val_loss = total_val_loss / (batch_idx + 1)
#                 val_retrieval_accuracy = val_retrieval_correct / total_val
#                 val_classification_accuracy = val_classification_correct / total_val

#                 if val_retrieval_accuracy > best_val_retrieval_accuracy and val_classification_accuracy > best_val_classification_accuracy:
#                     # best_loss_val = val_loss
#                     best_val_retrieval_accuracy = val_retrieval_accuracy
#                     best_val_classification_accuracy = val_classification_accuracy
#                     best_epoch = epoch + 1
#                     os.makedirs(f"./results/models/{self.encoder_type}/{self.val}/{self.sub}", exist_ok=True)             
#                     torch.save(self.eeg_model.state_dict(), f"./results/models/{self.encoder_type}/{self.val}/{self.sub}/eeg_model.pth")
#                     torch.save(self.eeg_img_proj.state_dict(), f"./results/models/{self.encoder_type}/{self.val}/{self.sub}/eeg_img_proj.pth") 
#                     torch.save(self.eeg_text_proj.state_dict(), f"./results/models/{self.encoder_type}/{self.val}/{self.sub}/eeg_text_proj.pth")        
#                     print(f"models have been saved !")
#                 print('Epoch:', epoch + 1,
#                     '  Train loss: %.4f' % train_loss,
#                     '  Train image loss: %.4f' % train_img_loss,
#                     '  Train text loss: %.4f' % train_text_loss,
#                     '  Train retrieval accuracy: %.4f' % train_retrieval_accuracy,
#                     '  Train classification accuracy: %.4f' % train_classification_accuracy,
#                     '  Validation loss: %.4f' % val_loss,
#                     '  Validation retrieval accuracy: %.4f' % val_retrieval_accuracy,
#                     '  Validation classification accuracy: %.4f' % val_classification_accuracy,
#                     )
#                 self.log_write.write('Epoch %d: Train loss: %.4f, Train retrieval accuracy: %.4f, Train classification accuracy: %.4f, Validation loss: %.4f, Validation retrieval accuracy: %.4f, Validation classification accuracy: %.4f\n'%(epoch+1, train_loss, train_retrieval_accuracy, train_classification_accuracy, val_loss, val_retrieval_accuracy, val_classification_accuracy))

#         print('The best epoch is: %d\n'% best_epoch)
#         self.log_write.write('The best epoch is: %d\n'% best_epoch)

#         test_loss, retrieval_test_accuracy, retrieval_top5_acc = self.evaluate_model_retrieval(test_loader, class_test_features, img_test_features, k=200)
#         _, retrieval_v2_acc, _ = self.evaluate_model_retrieval(test_loader, class_test_features, img_test_features, k = 2)
#         _, retrieval_v4_acc, _ = self.evaluate_model_retrieval(test_loader, class_test_features, img_test_features, k = 4)
#         _, retrieval_v10_acc, _ = self.evaluate_model_retrieval(test_loader, class_test_features, img_test_features, k = 10)
#         _, retrieval_v50_acc, retrieval_v50_top5_acc = self.evaluate_model_retrieval(test_loader, class_test_features, img_test_features,  k=50)
#         _, retrieval_v100_acc, retrieval_v100_top5_acc = self.evaluate_model_retrieval(test_loader, class_test_features, img_test_features,  k=100)
        
#         _, classification_test_accuracy, classification_top5_acc = self.evaluate_model_classification(test_loader, class_test_features, img_test_features, k=200)
#         _, classification_v2_acc, _ = self.evaluate_model_classification(test_loader, class_test_features, img_test_features, k = 2)
#         _, classification_v4_acc, _ = self.evaluate_model_classification(test_loader, class_test_features, img_test_features, k = 4)
#         _, classification_v10_acc, _ = self.evaluate_model_classification(test_loader, class_test_features, img_test_features, k = 10)
#         _, classification_v50_acc, classification_v50_top5_acc = self.evaluate_model_classification(test_loader, class_test_features, img_test_features,  k=50)
#         _, classification_v100_acc, classification_v100_top5_acc = self.evaluate_model_classification(test_loader, class_test_features, img_test_features,  k=100)
        
#         # test_losses.append(test_loss)
#         # test_accuracies.append(test_accuracy)
#         # v2_accs.append(v2_acc)
#         # v4_accs.append(v4_acc)
#         # v10_accs.append(v10_acc)

#         results = {
#         "epoch": best_epoch,
#         "train_loss": train_loss,
#         "train_retrieval_accuracy": train_retrieval_accuracy,
#         "train_classification_accuracy": train_classification_accuracy,
#         "test_loss": test_loss,
#         "retrieval_test_accuracy": retrieval_test_accuracy,
#         "retrieval_top5_acc": retrieval_top5_acc,
#         "retrieval_v2_acc": retrieval_v2_acc,
#         "retrieval_v4_acc": retrieval_v4_acc,
#         "retrieval_v10_acc": retrieval_v10_acc,
#         "retrieval_v50_acc": retrieval_v50_acc,
#         "retrieval_v100_acc": retrieval_v100_acc,
#         "retrieval_v50_top5_acc": retrieval_v50_top5_acc,
#         "retrieval_v100_top5_acc": retrieval_v100_top5_acc,
#         "classification_test_accuracy": classification_test_accuracy,
#         "classification_top5_acc": classification_top5_acc,
#         "classification_v2_acc": classification_v2_acc,
#         "classification_v4_acc": classification_v4_acc,
#         "classification_v10_acc": classification_v10_acc,
#         "classification_v50_acc": classification_v50_acc,
#         "classification_v100_acc": classification_v100_acc,
#         "classification_v50_top5_acc": classification_v50_top5_acc,
#         "classification_v100_top5_acc": classification_v100_top5_acc,
#         }
#         self.log_write.write('Best epoch %d: retrieval_test_accuracy: %.4f, retrieval_top5_acc: %.4f, retrieval_v2_acc: %.4f, retrieval_v4_acc: %.4f, retrieval_v10_acc: %.4f, retrieval_v50_acc: %.4f, retrieval_v100_acc: %.4f, retrieval_v50_top5_acc: %.4f, retrieval_v100_top5_acc: %.4f\n'%(best_epoch, retrieval_test_accuracy, retrieval_top5_acc, retrieval_v2_acc, retrieval_v4_acc, retrieval_v10_acc, retrieval_v50_acc, retrieval_v100_acc, retrieval_v50_top5_acc, retrieval_v100_top5_acc))
#         self.log_write.write('classification_test_accuracy: %.4f, classification_top5_acc: %.4f, classification_v2_acc: %.4f, classification_v4_acc: %.4f, classification_v10_acc: %.4f, classification_v50_acc: %.4f, classification_v100_acc: %.4f, classification_v50_top5_acc: %.4f, classification_v100_top5_acc: %.4f\n'%(classification_test_accuracy, classification_top5_acc, classification_v2_acc, classification_v4_acc, classification_v10_acc, classification_v50_acc, classification_v100_acc, classification_v50_top5_acc, classification_v100_top5_acc))
#         print(f"Train Loss: {train_loss:.4f}, Train Retrieval Accuracy: {train_retrieval_accuracy:.4f}, Train Classification Accuracy: {train_classification_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Retrieval Accuracy: {retrieval_test_accuracy:.4f}, Retrieval Top5 Accuracy: {retrieval_top5_acc:.4f}, Test Classification Accuracy: {classification_test_accuracy:.4f}, Classification Top5 Accuracy: {classification_top5_acc:.4f}")
#         # print(f"Epoch {epoch + 1}/{self.epochs} - v2 Accuracy:{v2_acc} - v4 Accuracy:{v4_acc} - v10 Accuracy:{v10_acc} - v50 Accuracy:{v50_acc} - v50_top5_acc: {v50_top5_acc:.4f} - v100 Accuracy:{v100_acc} - v100_top5_acc{v100_top5_acc}")
#         # # Save results to a CSV file
#         # results_dir = f"./results/outputs/{self.encoder_type}/{self.sub}/{current_time}"
#         # os.makedirs(results_dir, exist_ok=True)          
#         # results_file = f"{results_dir}/{self.encoder_type}_{self.sub}.csv"
        
#         # with open(results_file, 'w', newline='') as file:
#         #     writer = csv.DictWriter(file, fieldnames=results.keys())
#         #     writer.writeheader()
#         #     writer.writerows(results)
#         # print(f'Results saved to {results_file}')
#         return results
            
#     def evaluate_model_retrieval(self, dataloader, text_features_all, img_features_all, k):
#         self.eeg_model.load_state_dict(torch.load(f"./results/models/{self.encoder_type}/{self.val}/{self.sub}/eeg_model.pth", weights_only=True), strict=False)
#         self.eeg_img_proj.load_state_dict(torch.load(f"./results/models/{self.encoder_type}/{self.val}/{self.sub}/eeg_img_proj.pth", weights_only=True), strict=False)
#         self.eeg_text_proj.load_state_dict(torch.load(f"./results/models/{self.encoder_type}/{self.val}/{self.sub}/eeg_text_proj.pth", weights_only=True), strict=False)
#         self.eeg_model.eval()
#         self.eeg_img_proj.eval()
#         self.eeg_text_proj.eval()
#         text_features_all = text_features_all.to(device).float()
#         img_features_all = img_features_all.to(device).float()
#         total_loss = 0
#         correct = 0
#         total = 0
#         top5_correct = 0
#         top5_correct_count = 0
        
#         all_labels = set(range(text_features_all.size(0)))
#         top5_acc = 0
#         with torch.no_grad():
#             for batch_idx, (eeg_data, labels, img_features, text_features, class_features) in enumerate(dataloader):
#                 eeg_data = eeg_data.to(device).float()
#                 labels = labels.to(device)
#                 img_features = img_features.to(device).float()
#                 text_features = text_features.to(device).float()
#                 class_features = class_features.to(device).float()

#                 eeg_features = self.eeg_model(eeg_data)
#                 eeg_img_features = self.eeg_img_proj(eeg_features)
#                 eeg_text_features = self.eeg_text_proj(eeg_features)
                
#                 logit_scale = self.eeg_model.logit_scale
#                 img_loss = self.clip_loss(eeg_img_features, img_features, logit_scale)
#                 text_loss = (self.clip_loss(eeg_text_features, class_features, logit_scale) + self.clip_loss(eeg_text_features, text_features, logit_scale)) / 2
#                 loss = self.alpha * img_loss + (1 - self.alpha) * text_loss
                
#                 total_loss += loss.item()
                
#                 for idx, label in enumerate(labels):
                    
#                     possible_classes = list(all_labels - {label.item()})
#                     selected_classes = random.sample(possible_classes, k-1) + [label.item()]
#                     selected_img_features = img_features_all[selected_classes]
#                     selected_text_features = text_features_all[selected_classes]
#                     if k==200:
                        
#                         # logits_text = logit_scale * eeg_features[idx] @ selected_text_features.T                    
#                         logits_img = logit_scale * eeg_img_features[idx] @ selected_img_features.T
#                         logits_single = logits_img
#                         # print("logits_single", logits_single.shape)
                        
#                         # predicted_label = selected_classes[torch.argmax(logits_single).item()]
#                         predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) \in {0, 1, ..., n_cls-1}
#                         if predicted_label == label.item():
#                             # print("predicted_label", predicted_label)
#                             correct += 1
                        
#                         # print("logits_single", logits_single)
#                         _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                            
                        
#                         if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
#                             top5_correct_count+=1                                
#                         total += 1
#                     elif k == 50 or k == 100:
                        
#                         selected_classes = random.sample(possible_classes, k-1) + [label.item()]
#                         # logits_text = logit_scale * eeg_features[idx] @ selected_text_features.T
#                         logits_img = logit_scale * eeg_img_features[idx] @ selected_img_features.T
#                         logits_single = logits_img
                        
#                         predicted_label = selected_classes[torch.argmax(logits_single).item()]
#                         if predicted_label == label.item():
#                             correct += 1
#                         _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                            
                        
#                         if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
#                             top5_correct_count+=1                                
#                         total += 1

#                     elif k==2 or k==4 or k==10:
#                         selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                        
#                         logits_img = logit_scale * eeg_img_features[idx] @ selected_img_features.T
#                         # logits_text = logit_scale * eeg_features[idx] @ selected_text_features.T
#                         # logits_single = (logits_text + logits_img) / 2.0
#                         logits_single = logits_img
#                         # print("logits_single", logits_single.shape)
                        
#                         # predicted_label = selected_classes[torch.argmax(logits_single).item()]
#                         predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) \in {0, 1, ..., n_cls-1}
#                         if predicted_label == label.item():
#                             correct += 1
#                         total += 1
#                     else:
#                         print("Error.")
                        
#         average_loss = total_loss / (batch_idx+1)
#         accuracy = correct / total
#         top5_acc = top5_correct_count / total
#         return average_loss, accuracy, top5_acc
    
#     def evaluate_model_classification(self, dataloader, text_features_all, img_features_all, k):
#         self.eeg_model.load_state_dict(torch.load(f"./results/models/{self.encoder_type}/{self.val}/{self.sub}/eeg_model.pth", weights_only=True), strict=False)
#         self.eeg_img_proj.load_state_dict(torch.load(f"./results/models/{self.encoder_type}/{self.val}/{self.sub}/eeg_img_proj.pth", weights_only=True), strict=False)
#         self.eeg_text_proj.load_state_dict(torch.load(f"./results/models/{self.encoder_type}/{self.val}/{self.sub}/eeg_text_proj.pth", weights_only=True), strict=False)
#         self.eeg_model.eval()
#         self.eeg_img_proj.eval()
#         self.eeg_text_proj.eval()
#         text_features_all = text_features_all.to(device).float()
#         img_features_all = img_features_all.to(device).float()
#         total_loss = 0
#         correct = 0
#         total = 0
#         top5_correct = 0
#         top5_correct_count = 0
        
#         all_labels = set(range(text_features_all.size(0)))
#         top5_acc = 0
#         with torch.no_grad():
#             for batch_idx, (eeg_data, labels, img_features, text_features, class_features) in enumerate(dataloader):
#                 eeg_data = eeg_data.to(device).float()
#                 labels = labels.to(device)
#                 img_features = img_features.to(device).float()
#                 text_features = text_features.to(device).float()
#                 class_features = class_features.to(device).float()

#                 eeg_features = self.eeg_model(eeg_data)
#                 eeg_img_features = self.eeg_img_proj(eeg_features)
#                 eeg_text_features = self.eeg_text_proj(eeg_features)
                
#                 logit_scale = self.eeg_model.logit_scale
#                 img_loss = self.clip_loss(eeg_img_features, img_features, logit_scale)
#                 text_loss = (self.clip_loss(eeg_text_features, class_features, logit_scale) + self.clip_loss(eeg_text_features, text_features, logit_scale)) / 2
#                 loss = self.alpha * img_loss + (1 - self.alpha) * text_loss
                
#                 total_loss += loss.item()
                
#                 for idx, label in enumerate(labels):
                    
#                     possible_classes = list(all_labels - {label.item()})
#                     selected_classes = random.sample(possible_classes, k-1) + [label.item()]
#                     selected_img_features = img_features_all[selected_classes]
#                     selected_text_features = text_features_all[selected_classes]
#                     if k==200:
                        
#                         logits_text = logit_scale * eeg_text_features[idx] @ selected_text_features.T                    
#                         # logits_img = logit_scale * eeg_img_features[idx] @ selected_img_features.T
#                         logits_single = logits_text
#                         # print("logits_single", logits_single.shape)
                        
#                         # predicted_label = selected_classes[torch.argmax(logits_single).item()]
#                         predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) \in {0, 1, ..., n_cls-1}
#                         if predicted_label == label.item():
#                             # print("predicted_label", predicted_label)
#                             correct += 1
                        
#                         # print("logits_single", logits_single)
#                         _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                            
                        
#                         if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
#                             top5_correct_count+=1                                
#                         total += 1
#                     elif k == 50 or k == 100:
                        
#                         selected_classes = random.sample(possible_classes, k-1) + [label.item()]
#                         logits_text = logit_scale * eeg_text_features[idx] @ selected_text_features.T
#                         # logits_img = logit_scale * eeg_img_features[idx] @ selected_img_features.T
#                         logits_single = logits_text
                        
#                         predicted_label = selected_classes[torch.argmax(logits_single).item()]
#                         if predicted_label == label.item():
#                             correct += 1
#                         _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                            
                        
#                         if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
#                             top5_correct_count+=1                                
#                         total += 1

#                     elif k==2 or k==4 or k==10:
#                         selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                        
#                         # logits_img = logit_scale * eeg_img_features[idx] @ selected_img_features.T
#                         logits_text = logit_scale * eeg_text_features[idx] @ selected_text_features.T
#                         # logits_single = (logits_text + logits_img) / 2.0
#                         logits_single = logits_text
#                         # print("logits_single", logits_single.shape)
                        
#                         # predicted_label = selected_classes[torch.argmax(logits_single).item()]
#                         predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) \in {0, 1, ..., n_cls-1}
#                         if predicted_label == label.item():
#                             correct += 1
#                         total += 1
#                     else:
#                         print("Error.")
                        
#         average_loss = total_loss / (batch_idx+1)
#         accuracy = correct / total
#         top5_acc = top5_correct_count / total
#         return average_loss, accuracy, top5_acc

# def main(): 
#     args = parser.parse_args()
#     seed_n = args.seed
#     # seed_n = np.random.randint(args.seed)

#     print('seed is ' + str(seed_n))
#     random.seed(seed_n)
#     np.random.seed(seed_n)
#     torch.manual_seed(seed_n)
#     torch.cuda.manual_seed(seed_n)
#     torch.cuda.manual_seed_all(seed_n)  # if using multi-GPU.
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
    
#     num_sub = args.num_sub
#     cal_num = 0
#     retrieval_aver_acc = []
#     retrieval_aver_top5 = []
#     retrieval_aver_v2 = []
#     retrieval_aver_v4 = []
#     retrieval_aver_v10 = []
#     retrieval_aver_v50 = []
#     retrieval_aver_v100 = []
#     retrieval_aver_v50_top5 = []
#     retrieval_aver_v100_top5 = []
    
#     classification_aver_acc = []
#     classification_aver_top5 = []
#     classification_aver_v2 = []
#     classification_aver_v4 = []
#     classification_aver_v10 = []
#     classification_aver_v50 = []
#     classification_aver_v100 = []
#     classification_aver_v50_top5 = []
#     classification_aver_v100_top5 = []

#     subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'] 

#     for i in range(num_sub):
#         sub = subjects[i]
#         cal_num += 1      
#         eeg2image = EEGToImage(args, sub, i+1)
#         results = eeg2image.train()
#         retrieval_aver_acc.append(results['retrieval_test_accuracy'])
#         retrieval_aver_top5.append(results['retrieval_top5_acc'])
#         retrieval_aver_v2.append(results['retrieval_v2_acc'])
#         retrieval_aver_v4.append(results['retrieval_v4_acc'])
#         retrieval_aver_v10.append(results['retrieval_v10_acc'])
#         retrieval_aver_v50.append(results['retrieval_v50_acc'])
#         retrieval_aver_v100.append(results['retrieval_v100_acc'])
#         retrieval_aver_v50_top5.append(results['retrieval_v50_top5_acc'])
#         retrieval_aver_v100_top5.append(results['retrieval_v100_top5_acc'])
        
#         classification_aver_acc.append(results['classification_test_accuracy'])
#         classification_aver_top5.append(results['classification_top5_acc'])
#         classification_aver_v2.append(results['classification_v2_acc'])
#         classification_aver_v4.append(results['classification_v4_acc'])
#         classification_aver_v10.append(results['classification_v10_acc'])
#         classification_aver_v50.append(results['classification_v50_acc'])
#         classification_aver_v100.append(results['classification_v100_acc'])
#         classification_aver_v50_top5.append(results['classification_v50_top5_acc'])
#         classification_aver_v100_top5.append(results['classification_v100_top5_acc'])
    
#     retrieval_aver_acc.append(np.mean(retrieval_aver_acc))
#     retrieval_aver_top5.append(np.mean(retrieval_aver_top5))
#     retrieval_aver_v2.append(np.mean(retrieval_aver_v2))
#     retrieval_aver_v4.append(np.mean(retrieval_aver_v4))
#     retrieval_aver_v10.append(np.mean(retrieval_aver_v10))
#     retrieval_aver_v50.append(np.mean(retrieval_aver_v50))
#     retrieval_aver_v100.append(np.mean(retrieval_aver_v100))
#     retrieval_aver_v50_top5.append(np.mean(retrieval_aver_v50_top5))
#     retrieval_aver_v100_top5.append(np.mean(retrieval_aver_v100_top5))
    
#     classification_aver_acc.append(np.mean(classification_aver_acc))
#     classification_aver_top5.append(np.mean(classification_aver_top5))
#     classification_aver_v2.append(np.mean(classification_aver_v2))
#     classification_aver_v4.append(np.mean(classification_aver_v4))
#     classification_aver_v10.append(np.mean(classification_aver_v10))
#     classification_aver_v50.append(np.mean(classification_aver_v50))
#     classification_aver_v100.append(np.mean(classification_aver_v100))
#     classification_aver_v50_top5.append(np.mean(classification_aver_v50_top5))
#     classification_aver_v100_top5.append(np.mean(classification_aver_v100_top5))
#     column = np.arange(1, cal_num+1).tolist()
#     column.append('ave')
#     pd_all = pd.DataFrame(columns=column, data=[retrieval_aver_acc, retrieval_aver_top5, retrieval_aver_v2, retrieval_aver_v4, retrieval_aver_v10, retrieval_aver_v50, retrieval_aver_v100, retrieval_aver_v50_top5, retrieval_aver_v100_top5, classification_aver_acc, classification_aver_top5, classification_aver_v2, classification_aver_v4, classification_aver_v10, classification_aver_v50, classification_aver_v100, classification_aver_v50_top5, classification_aver_v100_top5])
#     pd_all.to_csv(args.result_path + f'/output/{args.encoder_type}/{args.val}/result.csv')

# if __name__ == '__main__':
#     print(time.asctime(time.localtime(time.time())))
#     main()
#     print(time.asctime(time.localtime(time.time())))

