from transformers import LlamaForCausalLM
from .configuration_asvd_llama import ASVDLlamaConfig
import torch.nn as nn

class ASVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        # 這裡是實際用到 rank 的地方
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        self.ALinear = nn.Linear(rank, out_features, bias=bias)

    def forward(self, input):
        return self.ALinear(self.BLinear(input))

class ASVDLlamaForCausalLM(LlamaForCausalLM):
    config_class = ASVDLlamaConfig

    def __init__(self, config: ASVDLlamaConfig):
        # 先執行父類初始化
        super().__init__(config)
        
        # 取得 config 中的 rank 字典
        self.truncation_ranks = getattr(config, "truncation_ranks", {})
        if not self.truncation_ranks:
            return

        # 1. 先收集所有需要被替換的 Linear 層 (避免在迭代中修改結構)
        layers_to_replace = []
        for name, module in self.named_modules():
            if name in self.truncation_ranks and isinstance(module, nn.Linear):
                layers_to_replace.append((name, module))

        # 2. 執行替換
        for name, old_module in layers_to_replace:
            # 這裡正確提取對應這一個 layer 名稱的 rank
            current_rank = self.truncation_ranks[name]
            
            # 建立新的 ASVD 層
            new_layer = ASVDLinear(
                in_features=old_module.in_features,
                out_features=old_module.out_features,
                rank=current_rank, # 確實傳入 rank
                bias=old_module.bias is not None
            )

            # 3. 定位父節點並替換
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = self.get_submodule(parent_name) if parent_name else self
            setattr(parent, child_name, new_layer)
            
        # 釋放不再需要的參考
        del layers_to_replace



# from transformers import LlamaForCausalLM
# from .configuration_asvd_llama import ASVDLlamaConfig
# import torch.nn as nn

# class ASVDLinear(nn.Module):
#     def __init__(self, in_features, out_features, rank, bias=True):
#         super().__init__()
#         self.BLinear = nn.Linear(in_features, rank, bias=False)
#         self.ALinear = nn.Linear(rank, out_features, bias=bias)

#     def forward(self, input):
#         return self.ALinear(self.BLinear(input))

# class ASVDLlamaForCausalLM(LlamaForCausalLM):
#     config_class = ASVDLlamaConfig
#     def __init__(self, config:ASVDLlamaConfig):
#         super().__init__(config)
#         self.truncation_ranks=config.truncation_ranks

#         full_name_dict = {module: name for name, module in self.named_modules()}
#         linear_info = {}
#         modules = [self]
#         while len(modules) > 0:
#             submodule = modules.pop()
#             for name, raw_linear in submodule.named_children():
#                 if isinstance(raw_linear, nn.Linear):
#                     full_name = full_name_dict[raw_linear]
#                     linear_info[raw_linear] = {
#                         "father": submodule,
#                         "name": name,
#                         "full_name": full_name,
#                     }
#                 else:
#                     modules.append(raw_linear)


#         for name,module in self.named_modules():
#             if name in self.truncation_ranks:
#                 info=linear_info[module]
#                 new_layer=ASVDLinear(module.in_features,module.out_features,self.truncation_ranks[name],bias=module.bias is not None)
#                 setattr(info["father"], info["name"], new_layer)
         



# from transformers import LlamaForCausalLM
# from .configuration_asvd_llama import ASVDLlamaConfig
# import torch.nn as nn

# class ASVDLinear(nn.Module):
#     def __init__(self, in_features, out_features, rank, bias=True):
#         super().__init__()
#         # 保留原本的 A/B Linear
#         self.BLinear = nn.Linear(in_features, rank, bias=False)
#         self.ALinear = nn.Linear(rank, out_features, bias=bias)

#     def forward(self, input):
#         return self.ALinear(self.BLinear(input))

#     # --- 新增方法：把 A/B 拆出成獨立層，方便 GPTQ 量化 ---
#     def export_to_linear(self):
#         return nn.ModuleDict({
#             "BLinear": self.BLinear,
#             "ALinear": self.ALinear
#         })

# class ASVDLlamaForCausalLM(LlamaForCausalLM):
#     config_class = ASVDLlamaConfig

#     def __init__(self, config: ASVDLlamaConfig):
#         super().__init__(config)
#         self.truncation_ranks = getattr(config, "truncation_ranks", {})

#         if not self.truncation_ranks:
#             return

#         # 1. 找出需要替換的 Linear 層
#         layers_to_replace = []
#         for name, module in self.named_modules():
#             if name in self.truncation_ranks and isinstance(module, nn.Linear):
#                 layers_to_replace.append((name, module))

#         # 2. 執行替換，並在 module 層級把 A/B 拆開
#         for name, old_module in layers_to_replace:
#             rank = self.truncation_ranks[name]
#             new_layer = ASVDLinear(
#                 in_features=old_module.in_features,
#                 out_features=old_module.out_features,
#                 rank=rank,
#                 bias=old_module.bias is not None
#             )

#             # 定位父節點
#             parent_name = ".".join(name.split(".")[:-1])
#             child_name = name.split(".")[-1]
#             parent = self.get_submodule(parent_name) if parent_name else self

#             # 替換原始層
#             setattr(parent, child_name, new_layer)

#             # 3. 直接把 BLinear/ALinear 暴露到父層，讓 GPTQ 能掃描到
#             setattr(parent, f"{child_name}_BLinear", new_layer.BLinear)
#             setattr(parent, f"{child_name}_ALinear", new_layer.ALinear)

#         del layers_to_replace


'''
# # from transformers import LlamaForCausalLM
# # from .configuration_asvd_llama import ASVDLlamaConfig
# # import torch.nn as nn

# # class ASVDLinear(nn.Module):
# #     def __init__(self, in_features, out_features, rank, bias=True):
# #         super().__init__()
# #         self.BLinear = nn.Linear(in_features, rank, bias=False)
# #         self.ALinear = nn.Linear(rank, out_features, bias=bias)

# #     def forward(self, input):
# #         return self.ALinear(self.BLinear(input))

# # class ASVDLlamaForCausalLM(LlamaForCausalLM):
# #     config_class = ASVDLlamaConfig
# #     def __init__(self, config:ASVDLlamaConfig):
# #         super().__init__(config)
# #         self.truncation_ranks=config.truncation_ranks

# #         full_name_dict = {module: name for name, module in self.named_modules()}
# #         linear_info = {}
# #         modules = [self]
# #         while len(modules) > 0:
# #             submodule = modules.pop()
# #             for name, raw_linear in submodule.named_children():
# #                 if isinstance(raw_linear, nn.Linear):
# #                     full_name = full_name_dict[raw_linear]
# #                     linear_info[raw_linear] = {
# #                         "father": submodule,
# #                         "name": name,
# #                         "full_name": full_name,
# #                     }
# #                 else:
# #                     modules.append(raw_linear)


# #         for name,module in self.named_modules():
# #             if name in self.truncation_ranks:
# #                 info=linear_info[module]
# #                 new_layer=ASVDLinear(module.in_features,module.out_features,self.truncation_ranks[name],bias=module.bias is not None)
# #                 setattr(info["father"], info["name"], new_layer)
         
# from transformers import LlamaForCausalLM
# from .configuration_asvd_llama import ASVDLlamaConfig
# import torch.nn as nn

# class ASVDLinear(nn.Module):
#     def __init__(self, in_features, out_features, rank, bias=True):
#         super().__init__()
#         # 這裡是實際用到 rank 的地方
#         self.BLinear = nn.Linear(in_features, rank, bias=False)
#         self.ALinear = nn.Linear(rank, out_features, bias=bias)

#     def forward(self, input):
#         return self.ALinear(self.BLinear(input))

# class ASVDLlamaForCausalLM(LlamaForCausalLM):
#     config_class = ASVDLlamaConfig

#     def __init__(self, config: ASVDLlamaConfig):
#         # 先執行父類初始化
#         super().__init__(config)
        
#         # 取得 config 中的 rank 字典
#         self.truncation_ranks = getattr(config, "truncation_ranks", {})
#         if not self.truncation_ranks:
#             return

#         # 1. 先收集所有需要被替換的 Linear 層 (避免在迭代中修改結構)
#         layers_to_replace = []
#         for name, module in self.named_modules():
#             if name in self.truncation_ranks and isinstance(module, nn.Linear):
#                 layers_to_replace.append((name, module))

#         # 2. 執行替換
#         for name, old_module in layers_to_replace:
#             # 這裡正確提取對應這一個 layer 名稱的 rank
#             current_rank = self.truncation_ranks[name]
            
#             # 建立新的 ASVD 層
#             new_layer = ASVDLinear(
#                 in_features=old_module.in_features,
#                 out_features=old_module.out_features,
#                 rank=current_rank, # 確實傳入 rank
#                 bias=old_module.bias is not None
#             )

#             # 3. 定位父節點並替換
#             parent_name = ".".join(name.split(".")[:-1])
#             child_name = name.split(".")[-1]
#             parent = self.get_submodule(parent_name) if parent_name else self
#             setattr(parent, child_name, new_layer)
            
#         # 釋放不再需要的參考
#         del layers_to_replace


# # from transformers import LlamaForCausalLM
# # from .configuration_asvd_llama import ASVDLlamaConfig
# # import torch.nn as nn

# # class ASVDLinear(nn.Module):
# #     def __init__(self, in_features, out_features, rank, bias=True):
# #         super().__init__()
# #         # 保留原本的 A/B Linear
# #         self.BLinear = nn.Linear(in_features, rank, bias=False)
# #         self.ALinear = nn.Linear(rank, out_features, bias=bias)

# #     def forward(self, input):
# #         return self.ALinear(self.BLinear(input))

# #     # --- 新增方法：把 A/B 拆出成獨立層，方便 GPTQ 量化 ---
# #     def export_to_linear(self):
# #         return nn.ModuleDict({
# #             "BLinear": self.BLinear,
# #             "ALinear": self.ALinear
# #         })

# # class ASVDLlamaForCausalLM(LlamaForCausalLM):
# #     config_class = ASVDLlamaConfig

# #     def __init__(self, config: ASVDLlamaConfig):
# #         super().__init__(config)
# #         self.truncation_ranks = getattr(config, "truncation_ranks", {})

# #         if not self.truncation_ranks:
# #             return

# #         # 1. 找出需要替換的 Linear 層
# #         layers_to_replace = []
# #         for name, module in self.named_modules():
# #             if name in self.truncation_ranks and isinstance(module, nn.Linear):
# #                 layers_to_replace.append((name, module))

# #         # 2. 執行替換，並在 module 層級把 A/B 拆開
# #         for name, old_module in layers_to_replace:
# #             rank = self.truncation_ranks[name]
# #             new_layer = ASVDLinear(
# #                 in_features=old_module.in_features,
# #                 out_features=old_module.out_features,
# #                 rank=rank,
# #                 bias=old_module.bias is not None
# #             )

# #             # 定位父節點
# #             parent_name = ".".join(name.split(".")[:-1])
# #             child_name = name.split(".")[-1]
# #             parent = self.get_submodule(parent_name) if parent_name else self

# #             # 替換原始層
# #             setattr(parent, child_name, new_layer)

# #             # 3. 直接把 BLinear/ALinear 暴露到父層，讓 GPTQ 能掃描到
# #             setattr(parent, f"{child_name}_BLinear", new_layer.BLinear)
# #             setattr(parent, f"{child_name}_ALinear", new_layer.ALinear)

# #         del layers_to_replace


from transformers import LlamaForCausalLM
from .configuration_asvd_llama import ASVDLlamaConfig
import torch.nn as nn
import torch

# 引入 GPTQ 專用的量化線性層
try:
    from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2QuantLinear
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
except ImportError:
    # 如果沒安裝或環境不支援，回退到標準 Linear (僅用於非量化狀態)
    QuantLinear = nn.Linear

class ASVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        # 預設使用標準 Linear
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        self.ALinear = nn.Linear(rank, out_features, bias=bias)

    def forward(self, input):
        return self.ALinear(self.BLinear(input))

class ASVDLlamaForCausalLM(LlamaForCausalLM):
    config_class = ASVDLlamaConfig

    def __init__(self, config: ASVDLlamaConfig):
        super().__init__(config)
        
        self.truncation_ranks = getattr(config, "truncation_ranks", {})
        if not self.truncation_ranks:
            return

        # 檢查是否為量化配置
        is_quantized = getattr(config, "quantization_config", None) is not None

        layers_to_replace = []
        for name, module in self.named_modules():
            if name in self.truncation_ranks and isinstance(module, nn.Linear):
                layers_to_replace.append((name, module))

        for name, old_module in layers_to_replace:
            current_rank = self.truncation_ranks[name]
            
            # 建立 ASVD 層容器
            new_layer = ASVDLinear(
                in_features=old_module.in_features,
                out_features=old_module.out_features,
                rank=current_rank,
                bias=old_module.bias is not None
            )

            # --- 關鍵改動：如果是量化模型，將 BLinear 替換為 QuantLinear ---
            if is_quantized:
                # 從 config 讀取 GPTQ 參數 (預設 bits=4, group_size=128)
                q_cfg = config.quantization_config
                bits = getattr(q_cfg, "bits", 4)
                group_size = getattr(q_cfg, "group_size", 128)
                sym = getattr(q_cfg, "sym", True)
                desc_act = getattr(q_cfg, "desc_act", True)
                # 檢查對齊：如果不能被 32 整除，則不使用 Triton
                if current_rank % 32 != 0:
                    print(f"⚠️  Rank {current_rank} 無法被 32 整除，切換至 TorchQuantLinear")
                    TargetLayer = TorchQuantLinear
                else:
                    TargetLayer = TritonV2QuantLinear
                # 重新定義 BLinear 為 QuantLinear，這樣它才能接收 qweight 等數據
                new_layer.BLinear = TargetLayer(
                    bits=bits,
                    group_size=group_size,
                    in_features=old_module.in_features,
                    out_features=current_rank,
                    bias=False,
                    sym=sym,
                    desc_act=desc_act
                )
                # ALinear 通常保持 FP16 (因為我們在 dynamic 排除它了)，所以維持 nn.Linear

            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = self.get_submodule(parent_name) if parent_name else self
            setattr(parent, child_name, new_layer)
            
        del layers_to_replace'''