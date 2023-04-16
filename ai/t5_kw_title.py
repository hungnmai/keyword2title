import re
import os
import torch
from path import ROOT_PATH
from transformers import T5ForConditionalGeneration, T5Tokenizer

torch.set_num_threads(1)

model_path = os.path.join(ROOT_PATH, "models/checkpoint-2024")

kw_model = T5ForConditionalGeneration.from_pretrained(model_path)
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    kw_model = kw_model.to(device)

kw_tokenizer = T5Tokenizer.from_pretrained(model_path,
                                           bos_token='<s>',
                                           eos_token='</s>',
                                           pad_token='<pad>')


def get_value_mask(txt, k):
    regex = r"(?<=" + k + ")(.*?)(?=<extra_i)"

    matches = re.finditer(regex, txt, re.MULTILINE)
    value = ""
    for matchNum, match in enumerate(matches, start=1):
        value = match.group()
    return value


def _post_process(title: str) -> str:
    title = re.sub(' +', ' ', title.strip())
    sents = [sent[0].upper() + sent[1:] for sent in re.split("\. | \. ", title)]
    sents = [sent.strip() for sent in sents]
    title = ". ".join(sents)
    title = title.strip()

    return title


def gen_title_from_kw(keywords):
    mask = {}
    input_txt = "<extra_id_1> "
    mask["<extra_id_1>"] = ""
    for i, tok in enumerate(keywords, 2):
        input_txt += tok + " <extra_id_{}> ".format(i)
        extra_id = "<extra_id_{}>".format(i)
        mask[extra_id] = ""

    input_tokens = kw_tokenizer(input_txt,
                                add_special_tokens=True,
                                return_tensors="pt").input_ids
    min_length = len(input_tokens) + 24
    max_length = len(input_tokens) + 96

    generate_tokens = kw_model.generate(input_tokens,
                                        no_repeat_ngram_size=1,
                                        bos_token_id=kw_tokenizer.bos_token_id,
                                        eos_token_id=kw_tokenizer.eos_token_id,
                                        pad_token_id=kw_tokenizer.pad_token_id,
                                        max_length=max_length,
                                        min_length=min_length,
                                        do_sample=True,
                                        repetition_penalty=5.0,
                                        top_p=0.95,
                                        top_k=10,
                                        temperature=0.95,
                                        num_return_sequences=5,
                                        )
    decoder_txt = kw_tokenizer.decode(generate_tokens[0], skip_special_tokens=False)
    # print("decoder_txt: ", decoder_txt)
    for k, v in mask.items():
        v = get_value_mask(decoder_txt, k)
        mask[k] = v
        input_txt = input_txt.replace(k, v)
    # print("mask: ", mask)

    return input_txt


def get_longest_title(keywords):
    longest_title = ""
    for i in range(1):
        title = gen_title_from_kw(keywords)
        if len(title.split()) > len(longest_title.split()):
            longest_title = title

    return _post_process(longest_title)
