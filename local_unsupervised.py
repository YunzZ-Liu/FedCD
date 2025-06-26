import copy
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from networks.models import ModelFedCon  
from utils import losses                
from utils_SimPLE import sharpen        
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def update_ema_variables(model, ema_model, alpha, global_step):

    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

class UnsupervisedLocalUpdate(object):
    def __init__(self, args, idxs, n_classes):
        """
        Initializes the unsupervised client.
        
        Args:
            args: Command-line arguments or a config object.
            idxs (list): Data indices for this client.
            n_classes (int): Number of classes in the dataset.
        """
        self.args = args
        self.data_idxs = idxs
        self.n_classes = n_classes
        self.iter_num = 0
        self.max_grad_norm = args.max_grad_norm

        # --- FedCD Model Setup ---
        # 1. Student Model (self.model)
        # 2. Local Teacher Model (self.ema_model)
        # 3. Global Teacher Model (self.global_model)
        self.model = ModelFedCon(args.model, args.out_dim, n_classes=n_classes, drop_rate=args.drop_rate).cuda()
        self.ema_model = ModelFedCon(args.model, args.out_dim, n_classes=n_classes, drop_rate=args.drop_rate).cuda()
        self.global_model = ModelFedCon(args.model, args.out_dim, n_classes=n_classes, drop_rate=args.drop_rate).cuda()

        if len(args.gpu.split(',')) > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.ema_model = torch.nn.DataParallel(self.ema_model)
            self.global_model = torch.nn.DataParallel(self.global_model)

        # Local teacher parameters are not trained via backprop
        for param in self.ema_model.parameters():
            param.detach_()

        # Optimizer setup
        if args.opt == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.unsup_lr, betas=(0.9, 0.999), weight_decay=5e-4)
        elif args.opt == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.unsup_lr, momentum=0.9, weight_decay=5e-4)
        elif args.opt == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.unsup_lr, weight_decay=0.02)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.rounds, eta_min=args.min_lr)
        
        self.start = True
        self.unreliable_samples_indices = []

    def train(self, net_w, op_dict, global_round, unlabeled_idx, train_dl_local):
        """
        Main training function for the unsupervised client.
        
        Args:
            net_w (dict): State dict of the global model from the server.
            op_dict (dict): State dict of the optimizer.
            global_round (int): The current communication round.
            unlabeled_idx (int): The ID of this client.
            train_dl_local (DataLoader): The client's local data loader.
        """
        logger = logging.getLogger()
        
        # --- Initialization ---
        if self.start:
            self.ema_model.load_state_dict(copy.deepcopy(net_w))
            logger.info(f'Client {unlabeled_idx}: EMA model initialized.')
            self.start = False
        
        self.global_model.load_state_dict(copy.deepcopy(net_w))
        self.optimizer.load_state_dict(op_dict)

        self.model.train()
        self.ema_model.eval()
        self.global_model.eval()

        # --- Class Awareness Balance (CAB) - Identification Step ---
        # This is done once before local epochs, using the stable global model as proxy.
        logger.info(f'Client {unlabeled_idx}: Starting CAB identification...')
        confi_idx = self.get_confi_classes(train_dl_local, threshold=self.args.confidence_threshold)
        self.find_unreliable_samples(train_dl_local, confi_idx, top_k_range=(self.args.top_l, self.args.top_h))
        logger.info(f'Client {unlabeled_idx}: Found {len(confi_idx)} confident classes.')
        logger.info(f'Client {unlabeled_idx}: Identified {len(self.unreliable_samples_indices)} unreliable samples.')

        epoch_loss, epoch_loss_local, epoch_loss_global = [], [], []
        
        for epoch in range(self.args.local_ep):
            batch_loss, batch_loss_local, batch_loss_global = [], [], []
            
            for i, (img_index, weak_aug_batch, _) in enumerate(train_dl_local):
                weak_aug_batch = [wb.cuda() for wb in weak_aug_batch]
                img_index = img_index.cuda()

                # --- Dual Teacher Distillation (DTD) ---
                # 1. Get pseudo-labels from both teachers
                with torch.no_grad():
                    # Local Teacher
                    _, feas_tea_loc, logits_loc = self.ema_model(weak_aug_batch[0])
                    sharpened_loc = sharpen(F.softmax(logits_loc, dim=1), t=self.args.T)
                    
                    # Global Teacher
                    _, feas_tea_glo, logits_glo = self.global_model(weak_aug_batch[0])
                    sharpened_glo = sharpen(F.softmax(logits_glo, dim=1), t=self.args.T)

                # 2. Student forward pass
                _, feas_stu, logits_stu = self.model(weak_aug_batch[1])
                probs_stu = F.softmax(logits_stu, dim=1)

                # --- Knowledge Purification (Part of DTD) ---
                # Calculate variance based on KL divergence of features
                var_loc = self.cal_kl_variance(feas_stu, feas_tea_loc)
                var_glo = self.cal_kl_variance(feas_stu, feas_tea_glo)

                # --- Class Awareness Balance (CAB) - Recalibration Step ---
                weights = self.get_weights(img_index, global_round)

                # 3. Calculate distillation losses
                mse_local = losses.softmax_mse_loss(probs_stu, sharpened_loc) # (batch_size, n_classes)
                mse_global = losses.softmax_mse_loss(probs_stu, sharpened_glo)

                # Apply all components: weights from CAB and variance from DTD
                loss_local = torch.sum(mse_local * weights * var_loc) / torch.sum(weights)
                loss_global = torch.sum(mse_global * weights * var_glo) / torch.sum(weights)
                
                loss = self.args.lambda_1 * loss_global + self.args.lambda_2 * loss_local

                # --- Optimization ---
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

                # Update local teacher (EMA)
                update_ema_variables(self.model, self.ema_model, self.args.ema_decay, self.iter_num)
                self.iter_num += 1

                batch_loss.append(loss.item())
                batch_loss_local.append(loss_local.item())
                batch_loss_global.append(loss_global.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_loss_local.append(sum(batch_loss_local) / len(batch_loss_local))
            epoch_loss_global.append(sum(batch_loss_global) / len(batch_loss_global))

        if self.args.decay:
            self.scheduler.step()

        # Return updated models and stats
        return (self.model.state_dict(), self.ema_model.state_dict(), 
                sum(epoch_loss) / len(epoch_loss), 
                sum(epoch_loss_global) / len(epoch_loss_global), 
                sum(epoch_loss_local) / len(epoch_loss_local), 
                copy.deepcopy(self.optimizer.state_dict()))

    def get_confi_classes(self, dl, threshold=0.4):
        """
        FedCD: Exploits confident classes using the global model as a proxy.
        """
        self.global_model.eval()
        prediction_bank = torch.zeros(1, self.n_classes).cuda()
        
        with torch.no_grad():
            for _, weak_aug_batch, _ in dl:
                img_data = weak_aug_batch[0].cuda()
                _, _, output = self.global_model(img_data)
                prediction_bank += torch.sum(F.softmax(output, dim=1), dim=0)

        # Normalize probabilitie
        prediction_bank = prediction_bank.squeeze(0)
        if torch.max(prediction_bank) != torch.min(prediction_bank):
            bank_scaled = (prediction_bank - torch.min(prediction_bank)) / (torch.max(prediction_bank) - torch.min(prediction_bank))
        else:
            bank_scaled = torch.zeros_like(prediction_bank)

        confi_idx = torch.where(bank_scaled >= threshold)[0].tolist()
        return confi_idx

    def find_unreliable_samples(self, dl, confi_idx, top_k_range=(5, 6)):
        """
        FedCD: Identifies unreliable samples (potential rare classes).
        """
        self.global_model.eval()
        self.unreliable_samples_indices = []
        if not confi_idx: # If no confident classes, no samples can be identified
            return

        low_rank_th, high_rank_th = top_k_range
        
        with torch.no_grad():
            for img_index, weak_aug_batch, _ in dl:
                img_data = weak_aug_batch[0].cuda()
                _, _, outputs = self.global_model(img_data)
                
                # Get the rank of each class for each sample
                ranks = torch.argsort(torch.argsort(outputs, dim=1, descending=True), dim=1)
                
                for i in range(img_data.shape[0]):
                    # Check if any confident class has a low rank for this sample
                    for conf_class in confi_idx:
                        if low_rank_th <= ranks[i, conf_class] <= high_rank_th:
                            self.unreliable_samples_indices.append(img_index[i].item())
                            break # Move to the next sample

    def get_weights(self, img_index_batch, global_round):
        """
        FedCD: Recalibrates the loss function using inverse frequency weighting.
        """
        num_total_samples = len(self.data_idxs)
        num_unreliable_samples = len(self.unreliable_samples_indices)
        
        # Handle edge cases where no unreliable samples are found or all samples are unreliable
        if num_unreliable_samples == 0 or num_unreliable_samples == num_total_samples:
            # If no unreliable samples, all have equal weight.
            # The alpha_t factor is still applied.
            alpha_t = self.args.alpha_begin + (self.args.alpha_end - self.args.alpha_begin) * (global_round / self.args.rounds)
            return torch.full((img_index_batch.shape[0],), alpha_t).cuda()

        # Calculate inverse frequency weights
        weight_unreliable = num_total_samples / num_unreliable_samples
        weight_reliable = num_total_samples / (num_total_samples - num_unreliable_samples)
        
        # Create a base weight tensor for the batch
        base_weights = torch.full(img_index_batch.shape, weight_reliable).cuda()
        
        # Assign higher weight to unreliable samples in the current batch
        for i, idx in enumerate(img_index_batch):
            if idx.item() in self.unreliable_samples_indices:
                base_weights[i] = weight_unreliable
        
        # Apply the linear warmup factor alpha_t
        alpha_t = self.args.alpha_begin + (self.args.alpha_end - self.args.alpha_begin) * (global_round / self.args.rounds)
        
        # Combine base weights with the warmup factor 
        final_weights = alpha_t * base_weights
        
        # Normalize weights for the batch to stabilize training
        # This keeps the relative importance but scales the overall magnitude
        if torch.mean(final_weights) > 1e-6: # Avoid division by zero
            final_weights = final_weights / torch.mean(final_weights)

        return final_weights

    def cal_kl_variance(self, feat_student, feat_teacher):
        """
        FedCD: Knowledge Purification. 
        """

        log_prob_student = F.log_softmax(feat_student, dim=1)
        prob_teacher = F.softmax(feat_teacher, dim=1)
    
        # Calculate KL divergence for each sample in the batch
        kl_div = F.kl_div(log_prob_student, prob_teacher, reduction='none').sum(dim=1)
        
        # Return e^(-V) as the weight
        return torch.exp(-kl_div)
