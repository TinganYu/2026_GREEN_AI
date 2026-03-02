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
