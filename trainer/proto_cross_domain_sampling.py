import os
import re
import time
import logging
import shutil
import torch
import numpy as np
import util.warmup as warmup

from tqdm import tqdm
from torch import nn, optim
from tensorboardX import SummaryWriter

from .trainer_base import TrainerBase
from model import PrototypicalNet
from data_loader import CrossDomainSamplingDataLoader as SamplingDataLoader
from data_loader import CrossDomainSamplingEvalDataLoader

torch.backends.cudnn.deterministic = True

logger = logging.getLogger(__name__)



class ProtoCrossDomainSamplingTrainer(TrainerBase):
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
        self.model = PrototypicalNet(self.config["model.params"])
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


    def build_loss(self):
        # self.loss = ProtoDistanceLoss().to(self.device)
        # FIXME
        self.loss = nn.CrossEntropyLoss().to(self.device)

    
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

                distances = self.model(support_images, query_images)   # (way * train_batch_size, way)
                # labels = torch.tensor([[1 if i == j // query_images.size(1) else 0 for i in range(support_images.size(0))] for j in range(query_images.size(0) * query_images.size(1))], dtype=torch.float32)
                # FIXME
                labels = torch.tensor([j // query_images.size(1) for j in range(query_images.size(0) * query_images.size(1))], dtype=torch.long)

                labels = labels.to(self.device) # (train_batch_size, way)
                # loss = self.loss(distances, labels)
                # FIXME
                loss = self.loss(-distances, labels)
                loss.backward()
                self.optim.step()

                # stats
                step_time = time.time() - step_time_start
                step_time_start = time.time()
                accum_time += step_time
                accum_loss += loss.cpu().detach().item()

                pbar.set_postfix_str("loss: {:.5f}, step time: {:.2f}".format(loss.cpu().detach().item(), step_time))
                
                self.optim.zero_grad()
                self.global_step += 1
                accum_step += 1
                self.warmup_scheduler.dampen()

                if self.global_step % self.report_step == 0:
                    avg_loss = accum_loss / accum_step

                    logger.info('| Average Step Time: {:.2f} s'.format(accum_time / accum_step))
                    logger.info('| Average Step Loss: {:.5f}'.format(avg_loss))
                    
                    self.writer.add_scalar("loss", avg_loss, self.global_step)
                    self.writer.add_scalar("lr", self.optim.param_groups[0]["lr"], self.global_step)
                    self.writer.flush()
                    accum_time, accum_step, accum_loss = 0, 0, 0
                
                if self.global_step % self.save_step == 0:
                    self.save_model()

                if self.global_step % self.val_step == 0 and self.global_step >= self.mode_config["start_val_step"]:
                    self.validation_run()
                
                # Stop training if it's past total_step
                if self.global_step >= self.mode_config["total_step"]:
                    break

        # save trained model at the end
        self.save_model()



class ProtoDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, distances, labels):
        labels = labels.to(dtype=torch.bool)
        pos_loss = distances.masked_select(labels).sum()
        epsilon = 1e-7
        neg_loss = torch.sum(torch.log(torch.sum(torch.exp(-distances.masked_fill(labels, float("inf"))), dim=1) + epsilon))
        loss = (pos_loss + neg_loss) / (labels.size(0) * labels.size(1))

        return loss
        