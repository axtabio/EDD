from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datetime import datetime
import json
from tqdm.auto import tqdm
from copy import deepcopy
from glob import glob
from models.utils.continual_model import ContinualModel

class EDD(ContinualModel):
    NAME = "EDD"
    COMPATIBILITY = ['class-il', 'task-il']


    @staticmethod
    def get_parser(parser: ArgumentParser):
        # Basic memory resnet option
        parser.add_argument('--embedDim', type=int, default=256)

        # memory distillation option
        parser.add_argument('--lambda_memory', type= float, default=  1.0)
        parser.add_argument('--lambda_orthogonal', type=float, default=1.0, help='Weight for orthogonal loss (applied only to memory 2)')

        # Dynamic memory strategy options for memory module 1

        parser.add_argument('--memory_pruning_ratio', type=float, default=0.225)

        parser.add_argument('--ba_lr', type=float, default=1e-4)
        parser.add_argument('--ba_epochs', type=int, default=1)

        # Debug verification option (only for memory 2 orthogonality)
        parser.add_argument('--debug_verification', type=int, default=0, choices=[0, 1], help='Enable cosine similarity method verification every 200 steps')
        parser.add_argument('--every_step', type=int, default=200, help='Step interval for debugging verification')

        return parser       

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Memory distillation parameters
        self.lambda_memory = self.args.lambda_memory

        
        # Orthogonal loss parameters (only for memory 2)
        self.lambda_orthogonal = self.args.lambda_orthogonal
        
        # (Batch Adaptation) parameters
        self.ba_lr = self.args.ba_lr
        self.ba_epochs = self.args.ba_epochs
        
        # Network and optimizer storage
        self.old_net = None
        self.optimizer_ta = None
        
        # Tracking variables
        self.step_count = 0
        
        # Instantiate buffers for multi-task learning
        self.eye = torch.eye(self.dataset.N_CLASSES_PER_TASK *
                             self.dataset.N_TASKS).to(self.device)
        
        # Log directory
        self.run_dir = f"./edd/{datetime.now():%Y%m%d_%H%M%S}"
        os.makedirs(self.run_dir, exist_ok=True)

        # Debug verification flag (only for memory 2)
        self.debug_verification = self.args.debug_verification
        self.every_step = self.args.every_step
        
        # Loss scale tracking for analysis
        self.loss_scales = {
            'ce_loss': [],
            'memory_distillation_loss': [],
            'orthogonal_loss_memory_2': []
        }
    
    def create_memory_strategies(self):
        """
        based on args, create memory_strategies dictionary.
        
        Returns:
            dict: memory_strategies dictionary
        """
        memory_strategies = {}
        
        # Memory 1 strategy setting
        try:
            memory_strategies['1'] = {
                'pruning_type': 'Class',
                'expanding_type': 'fix',
                'pruning_ratio': self.args.memory_pruning_ratio
            }
        except AttributeError as e:
            print(f"Warning: Error setting Memory 1 strategy: {e}. Using default.")
        
        # Memory 2 strategy setting
        try:
            memory_strategies['2'] = {
                'pruning_type': 'Class',
                'expanding_type': 'fix',
                'pruning_ratio': self.args.memory_pruning_ratio
            }
        except AttributeError as e:
            print(f"Warning: Error setting Memory 2 strategy: {e}. Using default.")
        
        # if memory_strategies is empty, return None (use default)
        if not memory_strategies:
            print("No memory strategies configured, using default behavior.")
            return None
            
        print(f"Created memory strategies: {memory_strategies}")
        return memory_strategies

    def get_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=self.args.n_epochs, eta_min=0)

    def observe(self, inputs, labels, not_aug_inputs=None,logits=None, epoch=None):
        pc = self.current_task * self.dataset.N_CLASSES_PER_TASK
        ac = (self.current_task + 1) * self.dataset.N_CLASSES_PER_TASK

        t_logits = None
        # student network output
        s_logits, s_feature, s_memory_attn, s_memory_out = self.net(inputs, returnt='both')
        # teacher network output
        if self.current_task > 0:
            with torch.no_grad():
                t_logits, t_feature, t_memory_attn, t_memory_out = self.old_net(inputs, returnt='both')
                t_logits = torch.sigmoid(t_logits)
        
        # Calculate memory losses separately
        memory_distillation_loss = torch.tensor(0.0, device=s_logits.device)
        orthogonal_loss_memory_2 = torch.tensor(0.0, device=s_logits.device)
        
        # 1. Traditional Memory Distillation Loss (Teacher-Student)
        if self.current_task > 0 and self.old_net is not None:
            # Select data source based on memory_value
            s_data = s_memory_attn
            t_data = t_memory_attn
            loss_fn = self.cal_loss_cosine
            
            # Collect losses to sum them safely
            distillation_losses = []
            
            # Calculate loss for both memory modules
            # distillation for common memory part between Teacher and Student
            for mem_idx, (s_mem, t_mem) in enumerate(zip(s_data, t_data)):
                if s_mem is not None and t_mem is not None:
                    # Attention maps: (batch, HW, L) - Teacher size
                    teacher_memory_size = t_mem.shape[-1]  # L dimension
                    if s_mem.shape[-1] >= teacher_memory_size:
                        s_mem_teacher_part = s_mem[:, :, :teacher_memory_size]
                        distillation_losses.append(loss_fn(s_mem_teacher_part, t_mem))
                    else:
                        print(f"Warning: Student memory {mem_idx+1} size ({s_mem.shape[-1]}) < Teacher size ({teacher_memory_size})")

            
            # Sum all distillation losses
            if distillation_losses:
                memory_distillation_loss = sum(distillation_losses)

        # 2. Orthogonal Loss for Memory 2 Only (Frozen-Unfrozen Memory)
        if self.current_task > 0:
            # verbose output only occasionally (every 200 steps)
            verbose = self.debug_verification and (self.step_count % self.every_step == 0)
            # debugging verification every 200 steps (if enabled)
            debug_verification = self.debug_verification and (self.step_count % self.every_step == 0)
            
            orthogonal_loss_memory_2 = self.cal_loss_orth_cosine_memory_2(verbose=verbose, debug_verification=debug_verification)

        self.opt.zero_grad()
        loss_ce = self.get_mc_loss(s_logits, t_logits, labels, pc, ac)
        
        # Combine all losses (only memory 2 orthogonal loss)
        total_memory_loss = (self.lambda_memory * memory_distillation_loss + 
                           self.lambda_orthogonal * orthogonal_loss_memory_2)
        
        # Loss scale tracking and analysis
        ce_loss_val = loss_ce.item()
        
        # Safely extract scalar values from losses (handle both tensor and scalar cases)
        if isinstance(memory_distillation_loss, torch.Tensor):
            mem_dist_loss_val = memory_distillation_loss.item()
        else:
            mem_dist_loss_val = float(memory_distillation_loss)
            
        if isinstance(orthogonal_loss_memory_2, torch.Tensor):
            orth_loss_val = orthogonal_loss_memory_2.item()
        else:
            orth_loss_val = float(orthogonal_loss_memory_2)
        
        # Track loss scales for analysis
        self.loss_scales['ce_loss'].append(ce_loss_val)
        self.loss_scales['memory_distillation_loss'].append(mem_dist_loss_val)
        self.loss_scales['orthogonal_loss_memory_2'].append(orth_loss_val)
        
        # Keep only recent history (last 100 steps)
        for key in self.loss_scales:
            if len(self.loss_scales[key]) > 100:
                self.loss_scales[key] = self.loss_scales[key][-100:]
        
        # Detailed loss logging and adaptive lambda suggestions
        if self.debug_verification and self.step_count % self.every_step == 0:
            print(f"Task {self.current_task+1}, Step {self.step_count}:")
            print(f"  CE Loss: {ce_loss_val:.6f}")
            if self.current_task > 0:
                print(f"  Memory Distillation Loss: {mem_dist_loss_val:.6f} (weight: {self.lambda_memory})")
                print(f"  Orthogonal Loss (Memory 2): {orth_loss_val:.6f} (weight: {self.lambda_orthogonal})")
                print(f"  Total Memory Loss: {total_memory_loss:.6f}")
                # Loss scale analysis and adaptive lambda suggestions
                if len(self.loss_scales['ce_loss']) > 10:  # Need some history for analysis
                    self._analyze_loss_scales_and_suggest_lambdas()

        # Step counter for debugging
        self.step_count += 1

        loss = loss_ce + total_memory_loss
        loss.backward()

        self.opt.step()
        if hasattr(self, 'custom_scheduler'):
            self.custom_scheduler.step()
        return loss.item()

    def get_mc_loss(self, s_logits: torch.Tensor, t_logits: torch.Tensor, labels: torch.Tensor, pc: int, ac: int) -> torch.Tensor:
        s_logits = s_logits[:, :ac]
        if self.current_task == 0:
            # Compute loss on the current task
            targets = self.eye.to(labels.device)[labels][:, :ac]
            loss = F.binary_cross_entropy_with_logits(s_logits, targets)
            assert loss >= 0
        else:
            targets = self.eye.to(labels.device)[labels][:, pc:ac]
            comb_targets = torch.cat((t_logits[:, :pc], targets), dim=1)
            loss = F.binary_cross_entropy_with_logits(s_logits, comb_targets)
            assert loss >= 0

        return loss

    def begin_task(self, dataset):
        # network prepare
        self.net.eval()
        if self.current_task > 0 and self.old_net is not None:
            try:
                self.old_net.train()
                for ba_epoch in tqdm(range(self.ba_epochs), desc="BA Pre-Training Epochs", leave=True):
                    pbar = tqdm(dataset.train_loader, desc=f"BA Pre-Training Batch (Epoch {ba_epoch+1}/{self.ba_epochs})", leave=False)
                    for data in pbar:
                        inputs = data[0] # 0 : data 1 : label 2: no aug
                        inputs = inputs.to(self.device)
                        _ = self.old_net(inputs)
                        if self.optimizer_ba is not None:
                            self.optimizer_ba.zero_grad()
                            self.optimizer_ba.step()
                        else:
                            print("Warning: BA optimizer is None, skipping BA step")
                            break
                self.old_net.eval()
            except Exception as e:
                print(f"Error during BA pre-training: {e}")
        elif self.current_task > 0:
            print("Warning: BA requested but old_net is None")
            
        self.net.train()

        # prepare main optimizer
        self.opt = self.get_optimizer()
        self.custom_scheduler = self.get_scheduler()


    def end_task(self, dataset):
        # Dynamic memory management with strategies (after current task)
        memory_strategies = self.create_memory_strategies()
        current_task_classes = self.dataset.N_CLASSES_PER_TASK
        total_classes = self.dataset.N_CLASSES_PER_TASK * self.dataset.N_TASKS
        
        print(f"End of task {self.current_task}, applying dynamic memory management...")
        
        # Call dynamic_memory with strategies
        try:
            self.net.dynamic_memory(
                memory_strategies=memory_strategies,
                current_task_classes=current_task_classes,
                total_classes=total_classes
            )
        except Exception as e:
            print(f"Error in dynamic_memory: {e}")
            print("Continuing without dynamic memory management...")
        
        # Measure orthogonality after dynamic memory management (only memory 2)
        if self.debug_verification and self.current_task > 0:
            self.measure_orthogonality_memory_2()
        
        # snapshot and freeze 
        self.old_net = deepcopy(self.net.eval())
        self.net.train()

        try:
            for m in self.old_net.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    if m.weight is not None:
                        m.weight.requires_grad = True
                    if m.bias is not None:
                        m.bias.requires_grad = True
            params = [p for p in self.old_net.parameters() if p.requires_grad]
            if params:  # create optimizer only if params is not empty
                self.optimizer_ba = torch.optim.SGD(params, lr=self.ba_lr, weight_decay=0.0)
            else:   
                print("Warning: No parameters found for BA optimizer")
        except Exception as e:
            print(f"Error setting up BA optimizer: {e}")
            self.optimizer_ba = None

    def cal_loss_cosine(self, fs, ft):
        # flatten both tensors (reshape handles non-contiguous tensors)
        fs_flat = fs.flatten(start_dim=1)
        ft_flat = ft.flatten(start_dim=1)
        
        # normalize and compute cosine similarity
        fs_norm = F.normalize(fs_flat, p=2, dim=1)
        ft_norm = F.normalize(ft_flat, p=2, dim=1)
        
        # cosine loss = 1 - cosine similarity
        cos_sim = F.cosine_similarity(fs_norm, ft_norm, dim=1)
        loss = (1 - cos_sim).mean()
        
        return loss

    def cal_loss_mse(self, fs, ft):
        # flatten both tensors (reshape handles non-contiguous tensors)
        fs_flat = fs.flatten(start_dim=1)
        ft_flat = ft.flatten(start_dim=1)
        
        # MSE loss
        loss = F.mse_loss(fs_flat, ft_flat)
        
        return loss

    def cal_loss_orth_cosine_memory_2(self, verbose=False, debug_verification=False):
        """
        Calculate orthogonality loss between Frozen memory and unfrozen memory for Memory 2 only.
        Calculate cosine similarity in the same way as F.cosine_similarity
        and force orthogonality (cosine similarity = 0).
        """
        # Initialize as tensor to ensure consistent return type
        device = next(self.net.parameters()).device
        
        # Only check memory '2' (memory 2 only)
        key = '2'
        if key in self.net.memory_modules:
            try:
                memory = self.net.memory_modules[key]
                
                # check if frozen_indices exists
                if hasattr(memory, 'frozen_indices') and len(memory.frozen_indices) > 0:
                    current_L = memory.keys.shape[0]
                    
                    # separate frozen and unfrozen indices
                    frozen_indices = memory.frozen_indices
                    all_indices = torch.arange(current_L, device=memory.keys.device)
                    unfrozen_mask = torch.ones(current_L, dtype=torch.bool, device=memory.keys.device)
                    unfrozen_mask[frozen_indices] = False
                    unfrozen_indices = all_indices[unfrozen_mask]
                    
                    if len(unfrozen_indices) > 0:
                        # Frozen memory part (create unit vector with L2 normalization)
                        frozen_keys = F.normalize(memory.keys[frozen_indices], p=2, dim=1)
                        frozen_values = F.normalize(memory.values[frozen_indices], p=2, dim=1)
                        
                        # Unfrozen memory part (create unit vector with L2 normalization)
                        unfrozen_keys = F.normalize(memory.keys[unfrozen_indices], p=2, dim=1)
                        unfrozen_values = F.normalize(memory.values[unfrozen_indices], p=2, dim=1)
                        
                        # Method 1: All-pairs cosine similarity using matrix multiplication
                        # (more efficient and consider all pairs)
                        keys_cosine_sim = torch.mm(frozen_keys, unfrozen_keys.T)
                        values_cosine_sim = torch.mm(frozen_values, unfrozen_values.T)
                        
                        # Orthogonality loss: minimize |cosine_similarity|
                        # Perfect orthogonality: cosine_similarity = 0
                        keys_orth_loss = torch.mean(torch.abs(keys_cosine_sim))
                        values_orth_loss = torch.mean(torch.abs(values_cosine_sim))
                        
                        # Debug verification: Compare matrix method with F.cosine_similarity
                        if debug_verification:
                            print(f"\n🔍 DEBUG VERIFICATION - Memory {key} (Step {self.step_count}):")
                            print(f"   Keys verification:")
                            keys_loss1, keys_loss2 = self._verify_cosine_methods(
                                memory.keys[frozen_indices], memory.keys[unfrozen_indices]
                            )
                            print(f"   Values verification:")
                            values_loss1, values_loss2 = self._verify_cosine_methods(
                                memory.values[frozen_indices], memory.values[unfrozen_indices]
                            )
                            print(f"   Combined loss comparison:")
                            print(f"     Matrix method: {(keys_loss1 + values_loss1).item():.8f}")
                            print(f"     F.cosine method: {(keys_loss2 + values_loss2).item():.8f}")
                            print(f"     Total difference: {abs((keys_loss1 + values_loss1 - keys_loss2 - values_loss2).item()):.8f}")
                        
                        memory_orth_loss = keys_orth_loss + values_orth_loss
                        
                        if verbose:
                            # additional statistics
                            max_keys_sim = keys_cosine_sim.abs().max().item()
                            max_values_sim = values_cosine_sim.abs().max().item()
                            avg_keys_sim = keys_cosine_sim.abs().mean().item()
                            avg_values_sim = values_cosine_sim.abs().mean().item()
                            
                            print(f"Memory {key} cosine orthogonal loss - Keys: {keys_orth_loss.item():.6f} (max: {max_keys_sim:.6f}), "
                                  f"Values: {values_orth_loss.item():.6f} (max: {max_values_sim:.6f})")
                            print(f"  Avg |cosine similarity| - Keys: {avg_keys_sim:.6f}, Values: {avg_values_sim:.6f}")
                        
                        return memory_orth_loss
                    else:
                        if verbose:
                            print(f"Memory {key} - No unfrozen memory to compute orthogonality")
                        return torch.tensor(0.0, device=device)
                else:
                    if verbose:
                        print(f"Memory {key} - No frozen memory to compute orthogonality")
                    return torch.tensor(0.0, device=device)
                    
            except Exception as e:
                print(f"Error in cal_loss_orth_cosine_memory_2 for memory {key}: {e}")
                return torch.tensor(0.0, device=device)
        else:
            if verbose:
                print(f"Memory {key} - Not found in memory modules")
            return torch.tensor(0.0, device=device)
    
    def measure_orthogonality_memory_2(self):
        """
        Measure orthogonality between frozen and unfrozen parts of Memory 2 and print the result.
        Use cosine similarity-based measurement similar to cal_loss_orth_cosine_memory_2.
        """
        print("\n" + "="*60)
        print(f"📊 ORTHOGONALITY MEASUREMENT - Memory 2 Only - Task {self.current_task}")
        print("="*60)
        
        key = '2'
        if key in self.net.memory_modules:
            try:
                memory = self.net.memory_modules[key]
                
                if hasattr(memory, 'frozen_indices') and len(memory.frozen_indices) > 0:
                    current_L = memory.keys.shape[0]
                    frozen_indices = memory.frozen_indices
                    
                    # calculate unfrozen indices
                    all_indices = torch.arange(current_L, device=memory.keys.device)
                    unfrozen_mask = torch.ones(current_L, dtype=torch.bool, device=memory.keys.device)
                    unfrozen_mask[frozen_indices] = False
                    unfrozen_indices = all_indices[unfrozen_mask]
                    
                    if len(unfrozen_indices) > 0:
                        # Frozen and Unfrozen memory parts (create unit vectors with L2 normalization)
                        frozen_keys = F.normalize(memory.keys[frozen_indices], p=2, dim=1)
                        frozen_values = F.normalize(memory.values[frozen_indices], p=2, dim=1)
                        unfrozen_keys = F.normalize(memory.keys[unfrozen_indices], p=2, dim=1)
                        unfrozen_values = F.normalize(memory.values[unfrozen_indices], p=2, dim=1)
                        
                        # Calculate cosine similarity (inner product of normalized vectors)
                        keys_cosine_sim = torch.mm(frozen_keys, unfrozen_keys.T)
                        values_cosine_sim = torch.mm(frozen_values, unfrozen_values.T)
                        
                        # Calculate orthogonality metric (based on cosine similarity)
                        keys_avg_sim = keys_cosine_sim.abs().mean().item()
                        values_avg_sim = values_cosine_sim.abs().mean().item()
                        keys_max_sim = keys_cosine_sim.abs().max().item()
                        values_max_sim = values_cosine_sim.abs().max().item()
                        
                        # RMS (Root Mean Square) cosine similarity for overall measurement
                        keys_rms_sim = torch.sqrt(torch.mean(keys_cosine_sim ** 2)).item()
                        values_rms_sim = torch.sqrt(torch.mean(values_cosine_sim ** 2)).item()
                        
                        # Orthogonality score (1 = perfect orthogonal, 0 = completely aligned)
                        orthogonality_score = 1 - (keys_avg_sim + values_avg_sim) / 2
                        
                        print(f"🔍 Memory {key} ({len(frozen_indices)} frozen, {len(unfrozen_indices)} unfrozen):")
                        print(f"   Keys   - Avg |cosine|: {keys_avg_sim:.6f}, Max |cosine|: {keys_max_sim:.6f}, RMS: {keys_rms_sim:.6f}")
                        print(f"   Values - Avg |cosine|: {values_avg_sim:.6f}, Max |cosine|: {values_max_sim:.6f}, RMS: {values_rms_sim:.6f}")
                        print(f"   🎯 Orthogonality Score: {orthogonality_score:.6f} (1.0 = perfect orthogonal)")
                        
                        # performance feedback
                        if keys_max_sim > 0.5 or values_max_sim > 0.5:
                            print(f"   ⚠️  High similarity detected! Consider increasing lambda_orthogonal")
                        elif orthogonality_score > 0.95:
                            print(f"   ✅ Excellent orthogonality achieved!")
                        elif orthogonality_score > 0.9:
                            print(f"   👍 Good orthogonality achieved!")
                    else:
                        print(f"🔍 Memory {key}: No unfrozen memory to measure orthogonality")
                else:
                    print(f"🔍 Memory {key}: No frozen memory to measure orthogonality")
                    
            except Exception as e:
                print(f"❌ Error measuring orthogonality for memory {key}: {e}")
        else:
            print(f"🔍 Memory {key}: Not found in memory modules")
        
        print("="*60)

    def _analyze_loss_scales_and_suggest_lambdas(self):
        """
        Analyze the size of losses and suggest adaptive lambda values.
        """
        import numpy as np
        
        # Calculate recent averages (last 20 steps)
        recent_window = 20
        ce_avg = np.mean(self.loss_scales['ce_loss'][-recent_window:])
        mem_dist_avg = np.mean(self.loss_scales['memory_distillation_loss'][-recent_window:])
        orth_avg = np.mean(self.loss_scales['orthogonal_loss_memory_2'][-recent_window:])
        
        print(f"\n📊 LOSS SCALE ANALYSIS (Recent {recent_window} steps):")
        print(f"  CE Loss (baseline): {ce_avg:.6f}")
        print(f"  Memory Distillation: {mem_dist_avg:.6f} (raw)")
        print(f"  Orthogonal Loss (Memory 2): {orth_avg:.6f} (raw)")
        
        # Current weighted contributions
        current_mem_contrib = self.lambda_memory * mem_dist_avg
        current_orth_contrib = self.lambda_orthogonal * orth_avg
        
        print(f"  Current Weighted Contributions:")
        print(f"    Memory Distillation: {current_mem_contrib:.6f} (λ={self.lambda_memory})")
        print(f"    Orthogonal (Memory 2): {current_orth_contrib:.6f} (λ={self.lambda_orthogonal})")
        
        # 🎯 Adaptive Lambda Suggestions
        print(f"\n🎯 ADAPTIVE LAMBDA SUGGESTIONS:")
        
        # Target: Memory distillation loss should be 10-50% of CE loss
        if mem_dist_avg > 0:
            target_mem_contrib_range = (0.1 * ce_avg, 0.5 * ce_avg)  # 10-50% of CE loss
            suggested_lambda_mem_low = target_mem_contrib_range[0] / mem_dist_avg
            suggested_lambda_mem_high = target_mem_contrib_range[1] / mem_dist_avg
            print(f"  Memory Distillation λ: {suggested_lambda_mem_low:.1f} - {suggested_lambda_mem_high:.1f}")
            print(f"    (to achieve 10-50% of CE loss: {target_mem_contrib_range[0]:.4f} - {target_mem_contrib_range[1]:.4f})")
        else:
            print(f"  Memory Distillation λ: N/A (raw loss is 0)")
        
        # Target: Orthogonal loss should be 5-20% of CE loss  
        if orth_avg > 0:
            target_orth_contrib_range = (0.05 * ce_avg, 0.2 * ce_avg)  # 5-20% of CE loss
            suggested_lambda_orth_low = target_orth_contrib_range[0] / orth_avg
            suggested_lambda_orth_high = target_orth_contrib_range[1] / orth_avg
            print(f"  Orthogonal (Memory 2) λ: {suggested_lambda_orth_low:.1f} - {suggested_lambda_orth_high:.1f}")
            print(f"    (to achieve 5-20% of CE loss: {target_orth_contrib_range[0]:.4f} - {target_orth_contrib_range[1]:.4f})")
        else:
            print(f"  Orthogonal (Memory 2) λ: N/A (raw loss is 0)")
        
        # Balance analysis
        total_mem_loss = current_mem_contrib + current_orth_contrib
        mem_ce_ratio = total_mem_loss / ce_avg if ce_avg > 0 else 0
        
        print(f"\n⚖️  BALANCE ANALYSIS:")
        print(f"  Total Memory Loss / CE Loss Ratio: {mem_ce_ratio:.2f}")
        if mem_ce_ratio < 0.1:
            print(f"  🔴 Memory losses are too small compared to CE loss")
            print(f"  💡 Consider increasing lambda values")
        elif mem_ce_ratio > 1.0:
            print(f"  🔴 Memory losses are dominating CE loss")
            print(f"  💡 Consider decreasing lambda values")
        else:
            print(f"  🟢 Loss balance looks reasonable")
        


    def _verify_cosine_methods(self, frozen_vecs, unfrozen_vecs):
        """
        Debugging: Compare matrix multiplication and F.cosine_similarity results
        """
        # Method 1: Matrix multiplication (current method)
        frozen_norm = F.normalize(frozen_vecs, p=2, dim=1)
        unfrozen_norm = F.normalize(unfrozen_vecs, p=2, dim=1)
        cosine_matrix = torch.mm(frozen_norm, unfrozen_norm.T)
        loss1 = torch.mean(torch.abs(cosine_matrix))
        
        # Method 2: F.cosine_similarity (for verification only)
        cosine_list = []
        max_diff = 0.0
        
        for i, f_vec in enumerate(frozen_norm):
            for j, u_vec in enumerate(unfrozen_norm):
                cos_sim = F.cosine_similarity(f_vec.unsqueeze(0), u_vec.unsqueeze(0), dim=1)
                cosine_list.append(cos_sim.abs())
                # Track maximum difference
                diff = abs(cos_sim.item() - cosine_matrix[i, j].item())
                max_diff = max(max_diff, diff)
        
        loss2 = torch.stack(cosine_list).mean()
        
        print(f"     Matrix: {loss1.item():.8f} | F.cosine: {loss2.item():.8f} | Diff: {abs(loss1.item() - loss2.item()):.2e} | MaxElementDiff: {max_diff:.2e}")
        
        return loss1, loss2
