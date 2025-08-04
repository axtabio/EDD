import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from backbone import register_backbone, MammothBackbone
from backbone.ResNetBlock import BasicBlock, conv3x3
import os
import matplotlib.pyplot as plt

class memoryModule(nn.Module):
    def __init__(self, L=50, channel=128, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.keys = nn.Parameter(torch.randn(L, channel))
        self.values = nn.Parameter(torch.randn(L, channel))
        nn.init.xavier_uniform_(self.keys)
        nn.init.xavier_uniform_(self.values)
        self.softmax = nn.Softmax(dim=1)
        self.channel = channel
        
        # save iniital value
        self.register_buffer('initial_keys', self.keys.data.clone())
        self.register_buffer('initial_values', self.values.data.clone())
        
        # track frozen indices
        self.register_buffer('frozen_indices', torch.tensor([], dtype=torch.long))
        
        # save hook handles (avoid duplicate registration)
        self._hook_handles = []
    
    def _clear_hooks(self):
        """Remove existing gradient hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
    
    # input of size (b,C,H,W) 
    def forward(self, x: Tensor, normality=False) -> Tensor:
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1).permute(0, 2, 1)  # (batch, HW, channels)

        if not normality:
            keys = F.normalize(self.keys, p=2, dim=1).unsqueeze(0).expand(b, -1, -1) # (1, L, C)
            queries = F.normalize(x_flat, p=2, dim=2) # (batch, HW, C)
        else:
            keys = F.normalize(self.values, p=2, dim=1).unsqueeze(0).expand(b, -1, -1) # (1, L, C)
            queries = F.normalize(x_flat, p=2, dim=2) # (batch, HW, C)

        # compute cosine similarity
        attn = torch.bmm(queries, keys.transpose(1, 2))  # (batch, HW, L)
        attn = self.softmax(attn) # (batch, HW, L)
        
        # retrieve
        values = self.values.unsqueeze(0).expand(b, -1, c)  # (batch, L, C)
        out_flat = torch.bmm(attn, values)  # (batch, HW, C)

        out = out_flat.permute(0, 2, 1).view(b, c, h, w) # (batch, C, H, W)

        return out, attn
    
class MemResNet(MammothBackbone):
    def __init__(self, block, num_blocks, num_classes, nf, embedDim,use_memory_layers=[1, 2], concat='memory', initial_conv_k=3):
        super(MemResNet,self).__init__()
        self.return_prerelu = False
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.embedDim = embedDim  # save embedDim

        if initial_conv_k != 3:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.conv1 = nn.Conv2d(3, nf * 1 * block.expansion, kernel_size=initial_conv_k, stride=2, padding=3, bias=False)
        else:
            self.conv1 = conv3x3(3, nf * 1 * block.expansion)
        self.bn1 = nn.BatchNorm2d(nf * 1 * block.expansion)


        if isinstance(use_memory_layers, str):
            self.use_memory_layers = [int(x.strip()) for x in use_memory_layers.split(',') if x.strip()]
        self.concat = concat  # 'resnet', 'memory', 'both', 'agumented'
        self.memory_modules = nn.ModuleDict()

        if 1 in self.use_memory_layers:
            self.memory_modules['1'] = memoryModule(L=embedDim, channel=nf * 1 * block.expansion)
        if 2 in self.use_memory_layers:
            self.memory_modules['2'] = memoryModule(L=embedDim, channel=nf * 2 * block.expansion)

        self.layer1 = self._make_layer(block, nf * 1 * block.expansion, num_blocks[0], stride=1)
        in_planes_layer2 = self.layer1[-1].bn2.num_features
        
        if 1 in self.use_memory_layers and self.concat == 'both':
            in_planes_layer2 *= 2
        
        self.layer2 = self._make_layer(block, nf * 2 * block.expansion, num_blocks[1], stride=2, in_planes=in_planes_layer2)
        
        in_planes_layer3 = self.layer2[-1].bn2.num_features
        if 2 in self.use_memory_layers and self.concat == 'both':
            in_planes_layer3 *= 2
        self.layer3 = self._make_layer(block, nf * 4 * block.expansion, num_blocks[2], stride=2, in_planes=in_planes_layer3)
        # layer4 is the same as the original
        self.layer4 = self._make_layer(block, nf * 8 * block.expansion, num_blocks[3], stride=2) if len(num_blocks) == 4 else nn.Identity()
        self.feature_dim = nf * 8 * block.expansion if len(num_blocks) == 4 else nf * 4 * block.expansion
        self.classifier = nn.Linear(self.feature_dim, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride, in_planes=None):
        if in_planes is not None:
            self.in_planes = in_planes
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, returnt='out'):
        out_0 = self.bn1(self.conv1(x))
        out_0 = nn.functional.relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)
        out_1 = self.layer1(out_0)
        mem_attn = []
        memory_out = []
        if 1 in self.use_memory_layers:
            mem_1_out, mem_1_attn  = self.memory_modules['1'](out_1)
            mem_attn.append(mem_1_attn)
            memory_out.append(mem_1_out)
            if self.concat == 'resnet':
                out_1_cat = out_1
            elif self.concat == 'memory':
                out_1_cat = mem_1_out
            elif self.concat == 'both':
                out_1_cat = torch.cat([out_1, mem_1_out], dim=1)
            elif self.concat == 'agumented':
                out_1_cat = out_1 + mem_1_out
            else:
                raise ValueError("concat option must be one of 'resnet', 'memory', 'both', 'agumented'")
        else:
            out_1_cat = out_1
        out_2 = self.layer2(out_1_cat)
        if 2 in self.use_memory_layers:
            mem_2_out,mem_2_attn = self.memory_modules['2'](out_2)
            mem_attn.append(mem_2_attn)
            memory_out.append(mem_2_out)
            if self.concat == 'resnet':
                out_2_cat = out_2
            elif self.concat == 'memory':
                out_2_cat = mem_2_out
            elif self.concat == 'both':
                out_2_cat = torch.cat([out_2, mem_2_out], dim=1)
            elif self.concat == 'agumented':
                out_2_cat = out_2 + mem_2_out
            else:
                raise ValueError("concat option must be one of 'resnet', 'memory', 'both', 'agumented'")
        else:
            out_2_cat = out_2
        out_3 = self.layer3(out_2_cat)
        out_4 = self.layer4(out_3)
        feature = nn.functional.avg_pool2d(out_4, out_4.shape[2])
        feature = feature.view(feature.size(0), -1)
        out = self.classifier(feature)

        if returnt == 'features':
            return feature
        elif returnt == 'memory':
            return out, mem_attn, memory_out
        elif returnt == 'both':
            # return logit, feature, memory features
            return (out, feature, mem_attn, memory_out)
        elif returnt == 'out':
            return out
        elif returnt == 'full':
            return out, [out_0, out_1, out_2, out_3, out_4]
    
    def dynamic_memory(self, memory_strategies=None, pruning_type='None', expanding_type='None', pruning_ratio=0.1, 
                      current_task_classes=None, total_classes=None):
        """
        Dynamic memory management with pruning and expanding capabilities.
        
        Args:
            memory_strategies (dict): Dictionary with memory-specific strategies
                Example: {'1': {'pruning_type': 'Class', 'expanding_type': 'fix'},
                         '2': {'pruning_type': 'Class', 'expanding_type': 'fix'}}
            pruning_type (str): Default pruning type - 'None', 'Top-K', 'Class'
            expanding_type (str): Default expanding type - 'None', 'fix', 'embedDim'
            pruning_ratio (float): Default ratio for Top-K pruning
            current_task_classes (int): number of classes in current task (for Class pruning)
            total_classes (int): total number of classes across all tasks (for Class pruning)
        """
        
        for memory_key, memory in self.memory_modules.items():
            try:
                device = memory.keys.device
                dtype = memory.keys.dtype
                current_L = memory.keys.shape[0]
                
                # set memory-specific strategies (get from dictionary or use default)
                if memory_strategies and memory_key in memory_strategies:
                    strategy = memory_strategies[memory_key]
                    current_pruning_type = strategy.get('pruning_type', pruning_type)
                    current_expanding_type = strategy.get('expanding_type', expanding_type)
                    current_pruning_ratio = strategy.get('pruning_ratio', pruning_ratio)
                else:
                    current_pruning_type = pruning_type
                    current_expanding_type = expanding_type
                    current_pruning_ratio = pruning_ratio
                
                # check and modify conditions
                # 1. if pruning is none, expanding must be embedDim/None
                if current_pruning_type == 'None' and current_expanding_type == 'fix':
                    print(f"Warning: Memory {memory_key} - 'fix' expanding not allowed with 'None' pruning. Setting to 'None'.")
                    current_expanding_type = 'None'
                
                
                # check range
                if not (0.0 <= current_pruning_ratio <= 1.0):
                    print(f"Warning: Memory {memory_key} - Invalid pruning_ratio {current_pruning_ratio}. Clipping to [0.0, 1.0]")
                    current_pruning_ratio = max(0.0, min(1.0, current_pruning_ratio))
                
                print(f"Memory {memory_key} strategy - Pruning: {current_pruning_type}, Expanding: {current_expanding_type}")
                
                # Step 1: Pruning
                newly_frozen_count = 0
                if current_pruning_type != 'None':
                    # calculate change compared to initial value
                    if hasattr(memory, 'initial_keys') and hasattr(memory, 'initial_values'):
                        # calculate change of keys and values (use L2 norm)
                        keys_change = torch.norm(memory.keys.data - memory.initial_keys, p=2, dim=1)
                        values_change = torch.norm(memory.values.data - memory.initial_values, p=2, dim=1)
                        
                        # total change (keys + values)
                        total_change = keys_change + values_change
                        
                        # normalize to 0~1 (more stable way)
                        if total_change.max() > total_change.min():
                            normalized_change = (total_change - total_change.min()) / (total_change.max() - total_change.min())
                        else:
                            normalized_change = torch.zeros_like(total_change)
                        
                        # determine pruning ratio
                        if current_pruning_type == 'Top-K':
                            freeze_ratio = current_pruning_ratio
                        elif current_pruning_type == 'Class':
                            if current_task_classes is not None and total_classes is not None and total_classes > 0:
                                freeze_ratio = current_task_classes / total_classes
                            else:
                                print(f"Warning: Memory {memory_key} - Class pruning requires valid current_task_classes and total_classes")
                                freeze_ratio = 0.0
                        
                        # select indices of top change (exclude already frozen)
                        unfrozen_mask = torch.ones(current_L, dtype=torch.bool, device=device)
                        if len(memory.frozen_indices) > 0:
                            unfrozen_mask[memory.frozen_indices] = False
                        
                        unfrozen_indices = torch.where(unfrozen_mask)[0]
                        if len(unfrozen_indices) > 0 and freeze_ratio > 0:
                            unfrozen_changes = normalized_change[unfrozen_indices]
                            num_to_freeze = max(1, int(len(unfrozen_indices) * freeze_ratio))  # at least 1
                            num_to_freeze = min(num_to_freeze, len(unfrozen_indices))  # limit to available indices
                            
                            if num_to_freeze > 0:
                                # select indices of top change
                                _, top_indices = torch.topk(unfrozen_changes, num_to_freeze)
                                new_frozen_indices = unfrozen_indices[top_indices]
                                
                                # combine with existing frozen indices
                                updated_frozen_indices = torch.cat([memory.frozen_indices, new_frozen_indices])
                                memory.frozen_indices = updated_frozen_indices
                                newly_frozen_count = num_to_freeze
                                
                                print(f"Memory {memory_key} - Pruned {num_to_freeze} slots. Total frozen: {len(memory.frozen_indices)}")
                    else:
                        print(f"Warning: Memory {memory_key} - Initial memory values not found. Skipping pruning.")
                
                # Step 2: Expanding
                expand_size = 0
                if current_expanding_type == 'fix':
                    # expand by newly frozen memory
                    expand_size = newly_frozen_count
                elif current_expanding_type == 'embedDim':
                    # expand by embedDim
                    expand_size = self.embedDim
                
                if expand_size > 0:
                    # create new keys/values (use Xavier initialization)
                    new_keys_data = torch.randn(expand_size, memory.keys.shape[1], device=device, dtype=dtype)
                    new_values_data = torch.randn(expand_size, memory.values.shape[1], device=device, dtype=dtype)
                    
                    # Xavier initialization
                    nn.init.xavier_uniform_(new_keys_data)
                    nn.init.xavier_uniform_(new_values_data)
                    
                    new_keys = nn.Parameter(new_keys_data)
                    new_values = nn.Parameter(new_values_data)
                    
                    # combine with existing keys/values
                    combined_keys = torch.cat([memory.keys.data, new_keys.data], dim=0)
                    combined_values = torch.cat([memory.values.data, new_values.data], dim=0)
                    
                    # replace with new parameters
                    memory.keys = nn.Parameter(combined_keys)
                    memory.values = nn.Parameter(combined_values)
                    
                    # expand initial values (new part is initialized with new values)
                    new_initial_keys = torch.cat([memory.initial_keys, new_keys.data.clone()], dim=0)
                    new_initial_values = torch.cat([memory.initial_values, new_values.data.clone()], dim=0)
                    memory.initial_keys = new_initial_keys
                    memory.initial_values = new_initial_values
                    
                    print(f"Memory {memory_key} - Expanded by {expand_size} slots. New size: {memory.keys.shape[0]}")
                
                # Step 3: set gradient (remove existing hooks and register new ones)
                memory._clear_hooks()
                
                # set all parameters to trainable
                memory.keys.requires_grad = True
                memory.values.requires_grad = True
                
                # set gradient hook for frozen indices
                if len(memory.frozen_indices) > 0:
                    def create_freeze_hook(frozen_idx):
                        def freeze_hook(grad):
                            if grad is not None:
                                grad_copy = grad.clone()
                                grad_copy[frozen_idx] = 0  # set gradient of frozen memory to 0
                                return grad_copy
                            return grad
                        return freeze_hook
                    
                    # register hook for each parameter and save handle
                    frozen_indices_cpu = memory.frozen_indices.cpu() if memory.frozen_indices.is_cuda else memory.frozen_indices
                    
                    keys_handle = memory.keys.register_hook(create_freeze_hook(memory.frozen_indices))
                    values_handle = memory.values.register_hook(create_freeze_hook(memory.frozen_indices))
                    
                    memory._hook_handles.extend([keys_handle, values_handle])
                    
                    print(f"Memory {memory_key} - Frozen {len(memory.frozen_indices)} indices: {frozen_indices_cpu.tolist()}")
                
            except Exception as e:
                print(f"Error processing Memory {memory_key}: {str(e)}")
                print(f"Skipping Memory {memory_key} processing...")
                continue

@register_backbone("EDD")
def MemResNetBlock_flexible_pruning(num_classes: int, embedDim: int = 100, nf: int = 64,use_memory_layers="1, 2", concat='memory'
):
    return MemResNet(
        BasicBlock, [2, 2, 2, 2], num_classes, nf, embedDim,
        use_memory_layers=use_memory_layers, concat=concat
    )