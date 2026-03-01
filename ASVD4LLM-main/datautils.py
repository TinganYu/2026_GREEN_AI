import os
import numpy as np
import torch
from datasets import load_dataset
import random
import io
import json

"""
doc https://huggingface.co/docs/datasets/loading
doc https://huggingface.co/docs/datasets/process
doc https://huggingface.co/blog/llama2#how-to-prompt-llama-2
"""


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def sample_train_loaders(name, tokenizer, nsamples=128, seed=0, seqlen=2048):
    set_seed(seed)
    if "wikitext2" in name:
        traindata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train",
            trust_remote_code=True
        )
        traindata = "\n\n".join(traindata["text"])
    elif "c4" in name:
        traindata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
            trust_remote_code=True
        )
        traindata = "\n\n".join(traindata["text"])
    elif "gsm8k" in name:
        ds = load_dataset("openai/gsm8k", "main", split="train", trust_remote_code=True)
        ds = ds.shuffle(seed=seed)
        trainloader = []
        for i in range(min(nsamples, len(ds))):
            ex = ds[i]
            txt = ex.get("question", "") + " " + ex.get("answer", "")
            enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=seqlen)
            inp = enc.input_ids[:, :seqlen]
            trainloader.append(inp)
        return trainloader
    else:
        raise NotImplementedError

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, len(traindata) - seqlen * 2 - 1)
        j = i + seqlen * 2
        # breakpoint()
        trainenc = tokenizer(traindata[i:j], return_tensors="pt")
        inp = trainenc.input_ids[:, :seqlen]
        trainloader.append(inp)
    return trainloader


def get_redpajama_train(tokenizer, percent=10, seed=3, batch_size=128, max_length=2048):
    def tokenization(example):
        return tokenizer(example["text"], truncation=True, max_length=max_length)

    if percent != 100:
        split = f"train[:{int(850000*percent/100)}]"
    else:
        split = "train"
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split=split,trust_remote_code=True)

    processed_dataset = dataset.map(tokenization, batched=True, batch_size=batch_size, num_proc=os.cpu_count())
    return processed_dataset


def get_english_quote(dataset_name, tokenizer):
    data = load_dataset(dataset_name,trust_remote_code=True)
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    return data["train"]


def get_qat_dataset(name, tokenizer, data_percent):
    if name == "red_pajama":
        data = get_redpajama_train(tokenizer, data_percent)

    elif name == "Abirate/english_quotes":
        data = get_english_quote(name, tokenizer)
    else:
        raise NotImplementedError
    data = data.shuffle()
    return data


llama_chat_format = """<s>[INST] <<SYS>>
"Below is an instruction that describes a task. Write a response that appropriately completes the request."
<</SYS>>

{{ instruction }} [/INST] {{ response }} </s>
"""


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def get_calib_data(name, tokenizer, model_id, nsamples, seqlen=2048, seed=3, use_bos=False):
    print(f" get_ptq_calib_data {name}, nsamples={nsamples}, seqlen={seqlen}, {seed}")
    cache_file = f"cache/{name}_{model_id.replace('/','_')}_{nsamples}_{seqlen}_{seed}_bos{use_bos}.pt"
    print(f"cache_file={cache_file}")
    if not os.path.exists("cache"):
        os.makedirs("cache")
    if os.path.exists(cache_file):
        traindataset = torch.load(cache_file)
        return traindataset
    # set seeds for reproducibility of sampling
    set_seed(seed)
    if name == "c4":
        traindata = load_dataset(
            "allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train",trust_remote_code=True
        )
        tot_text = "\n\n".join(traindata["text"])
    elif name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",trust_remote_code=True)
        tot_text = "\n\n".join(traindata["text"])
    elif name == "ptb":
        traindata = load_dataset("ptb_text_only", "penn_treebank", split="train",trust_remote_code=True)
        tot_text = "\n\n".join(traindata["sentence"])
    elif name == "alpaca":
        # this is for chat models
        data_path = "data/alpaca_data.json"
        list_data_dict = jload(data_path)
        traindataset = []
        selected_data_dict = random.sample(list_data_dict, nsamples)
        for example in selected_data_dict:
            if example.get("input", "") == "":
                s = llama_chat_format.format(instruction=example["instruction"], response=example["output"])
                trainenc = tokenizer(s, return_tensors="pt")
                inp = trainenc.input_ids[:, :seqlen]
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        return traindataset
    elif name == "gsm8k":
        # GSM8K is a QA dataset with 'question' and 'answer' fields; sample and tokenize
        ds = load_dataset("openai/gsm8k", "main", split="train", trust_remote_code=True)
        ds = ds.shuffle(seed=seed)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        selected = ds.select(range(min(nsamples, len(ds))))
        traindataset = []
        for ex in selected:
            q = ex.get("question", "")
            a = ex.get("answer", "")
            txt = q + " " + a
            if use_bos:
                txt = tokenizer.bos_token + txt
            # Pad and truncate to seqlen for consistent tensor shapes
            trainenc = tokenizer(txt, return_tensors="pt", padding="max_length", truncation=True, max_length=seqlen)
            inp = trainenc.input_ids
            attention_mask = trainenc.attention_mask
            traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        torch.save(traindataset, cache_file)
        return traindataset
    elif name == "selfgen":
        raise NotImplementedError

    else:
        raise NotImplementedError
    print(f"tot_text={len(tot_text)}")
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        txt = tot_text[i:j]
        ind = txt.find(".")
        txt = txt[ind + 1 :].strip()
        if use_bos:
            txt = tokenizer.bos_token + txt
        trainenc = tokenizer(txt, return_tensors="pt")
        inp = trainenc.input_ids[:, :seqlen]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    torch.save(traindataset, cache_file)
    return traindataset


def get_eval_loaders(name, tokenizer):
    if "wikitext2" in name:
        testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
            trust_remote_code=True
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    if "ptb" in name:
        valdata = load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="validation",
            trust_remote_code=True
        )
        testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
        return testenc
    if "c4" in name:
        testdata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
            trust_remote_code=True
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    if "gsm8k" in name:
        testdata = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
        texts = [q + " " + a for q, a in zip(testdata["question"], testdata["answer"])]
        testenc = tokenizer("\n\n".join(texts), return_tensors="pt", truncation=True)
        return testenc
    raise NotImplementedError
