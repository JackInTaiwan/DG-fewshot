import os
import re
import time
import logging
import shutil
import torch
import scipy.stats
import numpy as np
import scipy as sp
import util.warmup as warmup

from tqdm import tqdm
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

from .trainer_base import TrainerBase
from model import RelationNet
from data_loader import CrossDomainSamplingDataLoader as SamplingDataLoader
from data_loader import CrossDomainSamplingEvalDataLoader

torch.backends.cudnn.deterministic = True

logger = logging.getLogger(__name__)



class RelCrossDomainSamplingTrainer(TrainerBase):
    def __init__(self, config, mode, use_cpu):
        super().__init__()

        self.config = config
        self.mode_config = config["modes"][mode]
        self.mode = mode
        self.use_cpu = use_cpu
        self.device = None
        self.report_step = None
        self.save_step = None
        self.val_step = None
        self.total_step = None
        self.global_epoch = 1
        self.global_step = 1

        # attrs need to build
        self.pid_num = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.model = None
        self.optim = None
        self.loss = None
        self.writer = None

        
    def init_attributes(self):
        use_gpu = not self.use_cpu and torch.cuda.is_available()
        self.use_cpu = not use_gpu
        self.device = 'cuda:{}'.format(self.config["cuda.id"]) if use_gpu else 'cpu'
        if self.mode in ["train", "resume"]:
            self.report_step = self.mode_config["report_step"]
            self.save_step = self.mode_config["save_step"]
            self.val_step = self.mode_config["val_step"]
            self.total_step = self.mode_config["total_step"]


    def build_model(self):
        self.model = RelationNet(self.config["model.params"])
        self.model.to(self.device)

        logging.info("| Build up the model: \n{}".format(self.model))


    def build_train_data_loader(self):
        self.train_data_loader = SamplingDataLoader(
            source_meta_list=self.mode_config["domain_dataset.source.meta"],
            root_dir=self.mode_config["domain_dataset.source.root_dir"],
            way=self.mode_config["way"],
            shot=self.mode_config["shot"],
            input_image_size=(self.mode_config["input_image_size"], self.mode_config["input_image_size"]),
            batch_size=self.mode_config["train_batch_size"],
            augmentations=self.mode_config["augmentations"],
            mode="train"
        )

    
    def build_val_data_loader(self):
        self.val_data_loader = CrossDomainSamplingEvalDataLoader(
            meta_fp=self.mode_config["domain_dataset.val.meta"],
            root_dir=self.mode_config["domain_dataset.val.root_dir"],
            input_image_size=(self.mode_config["input_image_size"], self.mode_config["input_image_size"]),
            batch_size=128,
            mode="validation"
        )


    def build_optim(self):
        # NOTE: Must init optimizer after the model is moved to expected device to ensure the
        # consistency of the optimizer state dtype
        lr = self.mode_config["lr"]
        if self.mode_config["optimizer"] == "SGD":
            self.optim = optim.SGD(self.model.parameters(), lr=lr)
        elif self.mode_config["optimizer"] == "Adam":
            self.optim = optim.Adam(self.model.parameters(), lr=lr)
        
        self.warmup_scheduler = warmup.LinearWarmup(
            self.optim,
            warmup_period=self.mode_config["warmup_period"],
        )

        # FIXME
        self.step_scheduler = StepLR(self.optim, step_size=50000,gamma=0.5)

        
    def build_loss(self):
        self.loss = nn.MSELoss(reduction="mean").to(self.device)

    
    def build_summary_writer(self):
        save_dir = os.path.join(self.config["checkpoint.save_dir"], "events")
        self.writer = SummaryWriter(save_dir)


    def build(self, domain_dataset=None):
        if self.mode == 'train':
            self.init_attributes()
            self.build_train_data_loader()
            self.build_val_data_loader()
            self.build_model()
            self.build_optim()
            self.build_loss()
            self.build_summary_writer()

        elif self.mode == 'resume':
            self.init_attributes()
            self.build_train_data_loader()
            self.build_val_data_loader()
            self.build_model()
            self.build_optim()
            self.build_loss()
            self.load_checkpoint()
            self.build_summary_writer()

        elif self.mode == 'eval':
            pass
            # self.init_attributes()
            # self.build_eval_data_loader(domain_dataset)
            # self.build_model()
            # self.load_checkpoint()
            # self.build_summary_writer()
            
        else:
            raise ValueError('Wrong mode \'{}\' is given.'.format(mode))


    def train_run(self):
        self.model.train()
        
        with tqdm(self.train_data_loader) as pbar:
            accum_time = 0
            accum_loss = 0
            accum_step = 0

            for (support_images, query_images) in pbar:
                pbar.set_description_str("|g-steps: {}|".format(self.global_step))
                
                step_time_start = time.time()

                support_images = support_images.to(self.device) # (way, self.shot, 3, H, W)
                query_images = query_images.to(self.device)     # (way, train_batch_size, 3, H, W)

                scores = self.model(support_images, query_images)   # (way * train_batch_size, way)
                labels = torch.tensor([[1 if i == j // query_images.size(1) else 0 for i in range(support_images.size(0))] for j in range(query_images.size(0) * query_images.size(1))], dtype=torch.float32)
                labels = labels.to(self.device) # (train_batch_size, way)
                loss = self.loss(scores, labels)
                loss.backward()
                self.optim.step()

                # stats
                step_time = time.time() - step_time_start
                step_time_start = time.time()
                accum_time += step_time
                accum_loss += loss.cpu().detach().item()

                pbar.set_postfix_str("loss: {:.5f}, step time: {:.2f}".format(loss.cpu().detach().item(), step_time))
                
                accum_step += 1
                self.optim.zero_grad()
                self.global_step += 1
                self.warmup_scheduler.dampen()
                # FIXME
                self.step_scheduler.step()

                if self.global_step % self.report_step == 0:
                    avg_loss = accum_loss / accum_step

                    logger.info('| Average Step Time: {:.2f} s'.format(accum_time / accum_step))
                    logger.info('| Average Step Loss: {:.5f}'.format(avg_loss))
                    
                    self.writer.add_scalar("loss", avg_loss, self.global_step)
                    self.writer.add_scalar("lr", self.optim.param_groups[0]["lr"], self.global_step)
                    self.writer.flush()
                    accum_time, accum_step, accum_loss = 0, 0, 0
                
                if self.global_step % self.save_step == 0:
                    self.save_checkpoint()

                if self.global_step % self.val_step == 0 and self.global_step >= self.mode_config["start_val_step"]:
                    self.validation_run()
                
                # Stop training if it's past total_step
                if self.global_step >= self.mode_config["total_step"]:
                    break

        # save trained model at the end
        self.save_checkpoint()


    @staticmethod
    def calculate_acc(scores, labels):
        pred_labels = torch.argmax(scores, dim=1)
        acc = torch.mean((pred_labels == labels).to(dtype=torch.float32)).unsqueeze(0)
        std = torch.std((pred_labels == labels).to(dtype=torch.float32)).unsqueeze(0)

        return acc


    def validation_run(self):
        def mean_confidence_interval(data, confidence=0.95):
            a = 1.0 * np.array(data)
            n = len(a)
            m, se = np.mean(a), scipy.stats.sem(a)
            h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
            return m,h

        logger.info("| Run validation ...")
        torch.cuda.empty_cache()

        self.model.eval()

        total_acc_list = torch.tensor([])
        
        while self.val_data_loader.update_episode():
            support_data = self.val_data_loader.get_support_images()
            support_data = support_data.to(self.device)

            self.model.keep_support_features(support_data)

            total_scores, total_labels = torch.tensor([]), torch.tensor([])
            for (query_images, query_labels) in self.val_data_loader:
                query_images = query_images.to(self.device)
                scores = self.model.inference(query_images)
                scores = scores.detach().cpu()
                total_scores = torch.cat([total_scores, scores])
                total_labels = torch.cat([total_labels, query_labels])
            
            self.model.clean_support_features()

            acc = self.calculate_acc(total_scores, total_labels)
            total_acc_list = torch.cat([total_acc_list, acc])

        self.val_data_loader.reset_episode()

        avg_acc, h = mean_confidence_interval(total_acc_list)

        
        logger.info("| g-step: {}| Acc Avg: {:.5f} C.I.: {:.5f}".format(self.global_step, avg_acc, h))
        
        self.writer.add_scalar("val_acc", avg_acc, self.global_step)
        self.writer.add_scalar("val_CI", h, self.global_step)
        self.writer.flush()

        self.model.train()

        # save val_acc if it reaches best validation acc
        if avg_acc > self.best_val_acc:
            self.best_val_acc = avg_acc
            self.save_checkpoint(save_best=True)
        
        torch.cuda.empty_cache()
        
        logger.info("| Finish validation.")
