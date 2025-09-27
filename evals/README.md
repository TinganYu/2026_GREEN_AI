# 模型量化測試報告

## Llama-3.2-1B-Instruct 測試結果 (Few-shot)

| 模型         | 準確率 (Accuracy) | 備註               |
| ------------ | ----------------- | ------------------ |
| Baseline     | 0.4503            | 未量化模型         |
| BnB-4bit     | 0.2782            | BitsAndBytes 4-bit 量化 |
| AWQ-4bit     | 0.3723            | AWQ 4-bit 量化     |
| GPTQ-4bit    | 待測試            | GPTQ 4-bit 量化    |

---

## 測試說明

### 1. 測試模型
- 測試的模型為 `llama-3.2-1b-it`，使用不同的量化方法進行測試：
  - **Baseline**：未量化的原始模型。
  - **BnB-4bit**：使用 [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) 進行 4-bit 量化。
  - **AWQ-4bit**：使用 [AWQ](https://github.com/mit-han-lab/awq) 進行 4-bit 量化。
  - **GPTQ-4bit**：使用 [GPTQ](https://github.com/IST-DASLab/gptq) 進行 4-bit 量化（測試結果待補充）。

---

### 2. 測試方法
- **測試目標**：評估模型在數學推理問題上的表現。
- **數據集**：使用 `openai/gsm8k` 測試集，包含數百條數學問題。
- **測試方式**：
  - 使用 Few-shot Prompt 測試模型的數學推理能力。
  - 計算準確率 (Accuracy)：正確回答的問題數量佔總問題數量的比例。

---

### 3. 測試輸出
- 測試過程中生成的輸出結果存放於 `output` 資料夾中：
  - `llama-3.2-1b-it.txt`：Baseline 測試輸出。
  - `llama-3.2-1b-it-bnb-4bit.txt`：BnB-4bit 測試輸出。
  - `llama-3.2-1b-it-awq-4bit.txt`：AWQ-4bit 測試輸出。
  - `gemma-3-1b-it.txt`：其他模型 Baseline 測試輸出。
  - `gemma-3-1b-it-bnb-4bit.txt`：其他模型 BnB-4bit 測試輸出。

---

### 4. 環境需求
- **Python 版本**：3.8+
- **依賴套件**：
  - `transformers`
  - `torch`
  - `datasets`
  - `bitsandbytes`
  - `auto-gptq`
  - `awq`

---

### 5. 執行方式
1. 打開 `eval.ipynb` 測試腳本。
2. 修改 `model_name` 參數以切換不同的模型進行測試。
3. 在 Jupyter Notebook 中執行腳本，生成測試結果。

---

## 待辦事項
- [ ] 補充 GPTQ 4-bit 測試結果。
- [ ] 增加更多量化方法的測試。
- [ ] 優化測試流程，提升測試效率。