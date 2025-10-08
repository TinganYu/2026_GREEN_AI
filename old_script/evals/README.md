# 模型量化測試報告

## Llama-3.2-1B-Instruct 測試結果 (Few-shot)

| 模型         | 準確率 (Accuracy) | 備註               |
| ------------ | ----------------- | ------------------ |
| Baseline     | 0.4503            | 未量化模型         |
| BnB-4bit     | 0.2782            | BitsAndBytes 4-bit 量化 |
| AWQ-4bit     | 0.3723            | AWQ 4-bit 量化     |
| GPTQ-4bit    | 待測試            | GPTQ 4-bit 量化    |
