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

            # load best_val_acc if existing
            try:
                self.best_val_acc = checkpoint["best_val_acc"]
            except:
                pass

            logger.info('| Model is loaded successfully from \'{}\''.format(self.config["checkpoint.load_path"]))
        
        elif self.mode == "eval":
            logger.info("| Load checkpoint from {} ...".format(self.config["checkpoint.load_path"]))

            checkpoint = torch.load(self.config["checkpoint.load_path"])
            loaded_model_state_dict = checkpoint["model_state_dict"]

            state_dict = self.model.state_dict()
            
            # load global step
            self.global_step = checkpoint["global_step"]
    

    def save_checkpoint(self, save_best=False):
        save_dir = self.config["checkpoint.save_dir"]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if save_best:
            save_path = os.path.join(save_dir, '{}.checkpoint.best.pkl'.format(self.__class__.__name__))
        else:
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
