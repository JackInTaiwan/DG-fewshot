import os
import logging
import torch

logger = logging.getLogger(__name__)



class TrainerBase:
    def __init__(self):
        super().__init__()


    def load_checkpoint(self):
        if self.mode in ["train", "resume"]:
            logger.info("| Load checkpoint from {} ...".format(self.config["checkpoint.load_path"]))

            checkpoint = torch.load(self.config["checkpoint.load_path"])

            # load model state_dict
            
            self.model.load_state_dict(checkpoint["model_state_dict"])

            # load optim state_dict
            self.optim.load_state_dict(checkpoint["optim_state_dict"])

            # load global step
            self.global_step = checkpoint["global_step"]

            # update last_step of warmup scheduler
            self.warmup_scheduler.last_step = self.global_step

            logger.info('| Model is loaded successfully from \'{}\''.format(self.config["checkpoint.load_path"]))
        
        elif self.mode == "eval":
            logger.info("| Load checkpoint from {} ...".format(self.config["checkpoint.load_path"]))

            checkpoint = torch.load(self.config["checkpoint.load_path"])
            loaded_model_state_dict = checkpoint["model_state_dict"]

            state_dict = self.model.state_dict()
            
            # load global step
            self.global_step = checkpoint["global_step"]
    

    def save_model(self):
        save_dir = self.config["checkpoint.save_dir"]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, '{}.checkpoint.pkl'.format(self.__class__.__name__))

        # Dump the state_dict of model in cpu mode
        self.model.cpu()
        model_state_dict = self.model.state_dict()
        optim_state_dict = self.optim.state_dict()

        # NOTE: do remember to move model back to right device because we just called self.model.cpu() above
        self.model.to(self.device)
        
        # Save checkpoint
        torch.save({
            "model_state_dict": model_state_dict,
            "optim_state_dict": optim_state_dict,
            "global_step": self.global_step,
            },
            save_path
        )

        logger.info('| Checkpoint is saved successfully in \'{}\''.format(save_path))


    @staticmethod
    def calculate_acc(distances, labels):
        pred_labels = torch.argmin(distances, dim=1)
        acc = torch.mean((pred_labels == labels).to(dtype=torch.float32)).unsqueeze(0)

        return acc


    def validation_run(self):
        logger.info("| Run validation ...")

        self.model.eval()

        total_acc_list = torch.tensor([])
        
        while self.val_data_loader.update_episode():
            support_data = self.val_data_loader.get_support_images()
            support_data = support_data.to(self.device)

            self.model.keep_support_features(support_data)

            total_distances, total_labels = torch.tensor([]), torch.tensor([])
            for (query_images, query_labels) in self.val_data_loader:
                query_images = query_images.to(self.device)
                distances = self.model.inference(query_images)
                distances = distances.detach().cpu()
                total_distances = torch.cat([total_distances, distances])
                total_labels = torch.cat([total_labels, query_labels])
            
            self.model.clean_support_features()

            acc = self.calculate_acc(total_distances, total_labels)
            total_acc_list = torch.cat([total_acc_list, acc])

        self.val_data_loader.reset_episode()

        avg_acc = total_acc_list.mean().item()
        
        logger.info("| Average Acc: {:.5f}".format(avg_acc))
        
        self.writer.add_scalar("val_acc", avg_acc, self.global_step)
        self.writer.flush()

        self.model.train()
        
        logger.info("| Finish validation.")