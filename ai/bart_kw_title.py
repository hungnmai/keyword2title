import os
import json
import torch
import random
import datetime
from nltk import word_tokenize
from path import ROOT_PATH
from ai import process_data
from transformers import BartForConditionalGeneration, BartTokenizer

MODEL_PATH = os.path.join(ROOT_PATH, "models/checkpoint-7048")
torch.set_num_threads(1)


class BGenerator(object):
    def __init__(self, save_path):
        self.model = BartForConditionalGeneration.from_pretrained(save_path)
        self.tokenizer = BartTokenizer.from_pretrained(save_path)
        self.model.eval()

    def predict(self, kw, N, use_beam_search=False, verbose=False):
        input_ids = get_encoded_kw(kw, self.tokenizer)
        if verbose:
            print('input_ids: ', input_ids)
            print('length of input: ', len(input_ids))
            print('decode: ', self.tokenizer.decode(input_ids))
        ori_leng = len(input_ids)
        input_ids = torch.LongTensor([input_ids])
        force_words_ids = []
        for w in kw:
            wids = self.tokenizer([' %s' % w, w], add_special_tokens=False).input_ids
            force_words_ids.append(wids)
        # force_words_ids = [ for w in kw]
        if verbose:
            print('force_words_ids: ', force_words_ids)
        min_length = min(ori_leng + 16, 60)
        max_length = 80
        if not use_beam_search:
            output_sequences = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                temperature=0.9,
                top_k=10,
                top_p=0.95,
                num_beams=1,
                num_beam_groups=1,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                num_return_sequences=N,
                repetition_penalty=1.5,
                # decoder_start_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        else:
            output_sequences = self.model.generate(
                force_words_ids=force_words_ids,
                input_ids=input_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=10,
                do_sample=False,
                temperature=0.95,
                top_k=10,
                top_p=0.95,
                max_length=max_length,
                min_length=min_length,
                num_return_sequences=N,
                repetition_penalty=1.4,
                # decoder_start_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        answers = []
        output_sequences = output_sequences.tolist()
        for i in range(len(output_sequences)):
            wids = output_sequences[i]
            answer = self.tokenizer.decode(wids, skip_special_tokens=True)
            answer = answer.strip()
            answers.append(answer)
            if verbose:
                print('leng = %d ' % len(wids), ', wids: ', wids)
                print('---answer: ', answer)
        return answers


MODEL = BGenerator(MODEL_PATH)


def get_encoded_kw(kw, b_tokenizer):
    result = []
    start, sep, end = b_tokenizer.encode('<sep>')
    for w in kw:
        wids = b_tokenizer.encode(' ' + w)[1:-1]
        result += wids + [sep]
    result = [start] + result[:-1] + [end]
    return result


def get_score_of_interval(interval, score_dic):
    if score_dic is None:
        return interval[1] - interval[0] + 1
    return score_dic[interval]


def find_nearest_interval_index(current_index, intervals):
    index = current_index - 1
    if index == -1:
        return -1
    start = intervals[current_index][0]
    while index >= 0:
        if intervals[index][1] < start:
            return index
        index -= 1
    return -1


def get_longest_non_overlaping_intervals(intervals, score_dic=None):
    arr = sorted(intervals, key=lambda element: (element[1], element[0]))
    # print 'sorted array: ',arr
    s = {}
    trace = {}
    trace[-1] = []
    s[-1] = 0
    arr_len = len(arr)
    for i in range(0, arr_len):
        nearest_index = find_nearest_interval_index(i, arr)
        # print 'nearest of %s is %s'%(str(arr[i]), str(arr[nearest_index]))
        temp_sol = s[nearest_index] + get_score_of_interval(arr[i], score_dic)
        if temp_sol > s[i - 1]:
            s[i] = temp_sol
            trace[i] = list(trace[nearest_index])
            trace[i].append(i)
        else:
            trace[i] = list(trace[i - 1])
            s[i] = s[i - 1]
        # find the interval maximum index j such that arr[j].end < start
        # print 's=',s
        # print 'trace=', trace
    result = []
    # print trace
    for key in trace[arr_len - 1]:
        result.append(arr[key])
    return result


def normalize_text(text):
    words = word_tokenize(text)
    return ' '.join(words)


def locate_words_in_text(word, text):
    result = []
    index = 0
    w_leng = len(word)
    while index < len(text):
        m_index = text.find(word, index)
        if m_index >= 0:
            end_index = m_index + w_leng - 1
            result.append((m_index, end_index))
            index = end_index + 1
        else:
            break
    return result


def locate_answer_inside(kw, ori_text, verbose=False):
    text = normalize_text(ori_text)
    a_pairs = []
    for w in kw:
        pairs = locate_words_in_text(w, text)
        a_pairs.extend(pairs)
    if verbose:
        print('all pairs: ', a_pairs)
    indices = get_longest_non_overlaping_intervals(a_pairs)
    if verbose:
        print('indices of kw in text: ', indices)
    # mapping from kw --> indices, kw 1 --> (x, y), kw2
    result = {}  # indices of kw --> (x, y)
    ind_set = set([i for i in range(len(kw))])
    for start, end in indices:
        if len(ind_set) == 0:
            break
        surface = text[start: end + 1]
        for ind in ind_set:
            if kw[ind] == surface:
                ind_set.discard(ind)
                result[ind] = (start, end)
                break
    return result


def get_longest_text(texts):
    if len(texts) == 1:
        return texts[0]
    m_index = 0
    for i in range(1, len(texts)):
        if len(texts[m_index]) < len(texts[i]):
            m_index = i
    return texts[m_index]


def fix_url_change(title, kw):
    # first find URL inside kw
    urls = []


def generate_from_kw(kw, n=1, use_beam_search=False, verbose=False):
    n_kw = [normalize_text(w) for w in kw]
    l_kw = [w.lower().strip() for w in n_kw]
    if verbose:
        print('normalize and lower kw: ', l_kw)
    texts = MODEL.predict(l_kw, n, use_beam_search=use_beam_search)
    result = []
    failed = []
    for text in texts:
        if verbose:
            print('generated text: ', text)
        l_dic = locate_answer_inside(l_kw, text, verbose)
        if verbose:
            print('located kw in generated text: ', l_dic)
        if len(l_dic) == len(l_kw):  # contain all
            result.append(text)
        else:
            failed.append(text)
    if len(result) > 0:
        # choose the longest from satisfied generations
        return get_longest_text(result)
    # choose the
    return get_longest_text(failed)


def read_json(data_path):
    with open(data_path, 'r') as f:
        text = f.read()
        data = json.loads(text)
        return data


def stat_token_lengths():
    data_path = 'span/train_span.json'
    dev_path = 'span/dev_span.json'
    # model_path = '/Users/khaimai/Downloads/checkpoint-6250'
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    data = read_json(data_path)
    data.extend(read_json(dev_path))
    print('data size: ', len(data))
    count_dic = {}
    ave = 0
    count = 0
    tok_count = {}
    for item in data:
        title = item['title']
        words = title.split()
        for word in words:
            t_key = word.lower().strip()
            tok_count[t_key] = tok_count.get(t_key, 0) + 1
        # t_len = len(words)
        count += 1
        # count_dic[t_len] = count_dic.get(t_len, 0) + 1
        tokens = tokenizer.encode(title)
        t_len = len(tokens)
        ave += t_len
        count_dic[t_len] = count_dic.get(t_len, 0) + 1
    keys = sorted(list(count_dic.keys()))
    for key in keys:
        print('leng = %d, %d' % (key, count_dic[key]))
    print('overall length: ', ave / count)

    top_freg = sorted(tok_count.items(), key=lambda x: - x[1])
    for item, count in top_freg:
        if count > 100:
            print('%s: %d' % (item, count))


def count_miss_gen(data_path):
    with open(data_path, 'r') as f:
        text = f.read()
        items = json.loads(text)
    N = 5
    g_count, f_count = 0, 0
    failed_count = 0
    random.shuffle(items)
    items = items[:1000]
    print('number of items: ', len(items))
    t1 = datetime.datetime.now()
    count = 0
    size = len(items)
    all_failed_cases = []
    kw_count = 0
    for item in items:
        spans = item['spans']
        for kw in spans:
            g_result, failed = generate_from_kw(kw, N)
            g_count += len(g_result)
            f_count += len(failed)
            if len(g_result) == 0:
                failed_count += 1
                print('failed at kw = ', kw, ' title=', item['title'], ' failed: ', failed)
            kw_count += 1
        count += 1
        t2 = datetime.datetime.now()
        ave = (t2 - t1).seconds / count
        if count % 50 == 1:
            print('ave time for item: %f seconds, remaining time: %f seconds' % (ave, ave * (size - count)))
    print('g_count = %d, f_count = %d, all-fail=%d, kw_count=%d' % (g_count, f_count, failed_count, kw_count))


def main():
    cases = [
        'erectile dysfunction products from Leki na potencje, viagra',
        'drugs that restore impaired erectile function under conditions of sexual stimulation',
        'The pharmacy Leki na potencje recommends that you coordinate the dosage with your doctor'
    ]
    for case in cases:
        kw = case.split(',')
        print('-------------------------------------------')
        title = generate_from_kw(kw)
        print('result for: ', case)
        print('title: ', title)


def test_if_generation_contain():
    cases = [
        ['drugs that restore impaired erectile function under conditions of sexual stimulation'],
        ['The pharmacy Leki na potencje recommends that you coordinate the dosage with your doctor'],
        ['erectile dysfunction products from Leki na potencje', 'viagra'],
        ['drugs that restore impaired erectile function', 'conditions', 'sexual stimulation'],
        ['The pharmacy Leki na potencje', 'coordinate', 'dosage with your doctor']
    ]
    N = 10
    iter_num = 3
    g_count, f_count = 0, 0
    failed_count = 0
    for case in cases:
        print('test for case: ', case)
        for i in range(iter_num):
            g_result, failed = generate_from_kw(case, N)
            g_count += len(g_result)
            f_count += len(failed)
            if len(failed) > 0:
                print('failed response: ', failed)
            if len(g_result) == 0:
                failed_count += 1
    print('g_count = %d, f_count = %d, all-fail=%d' % (g_count, f_count, failed_count))


def debug():
    kw = ["faq 's about"]
    text = "faq's about penis enlargement - how to find the best one for you and your woman!"
    locate_answer_inside(kw, text, verbose=True)


def generate_multiple():
    kw = ['viagra', 'https://Arabmenshealth.com/sildenafil.html', 'sildenafil']
    tit_set = set()
    for i in range(100):
        tit = generate_from_kw(kw)
        if tit not in tit_set:
            tit_set.add(tit)
            print(tit)
    # print('number of distincts: ', len(tit_set))


def test_gen():
    print('input: kw1, kw2, ...')
    while True:
        kw = input().split(',')
        print('kw: ', kw)
        for i in range(5):
            tit = generate_from_kw(kw)
            print(tit)


def valid_title(spans, title):
    for tok in spans:
        if tok not in title:
            return False
    if title.lower().count("http") > 1:
        return False
    return True


def capitalize_after_period(text):
    leng = len(text)
    for i in range(leng):
        if text[i] == '.' and i + 1 < leng:
            if text[i + 1] == ' ':
                if i + 2 < leng:
                    text = text[:i + 2] + text[i + 2].upper() + text[i + 3:]
    return text


def generate_title_strict(keyword_normalized, keywords_dict, domains, max_iter=5):
    last_title = None
    for i in range(max_iter):
        title = generate_from_kw(keyword_normalized)
        title = process_data.post_processing(title=title, domains=domains, keyword_normalized=keywords_dict)
        if valid_title(keyword_normalized, title):
            return title, i
        last_title = title
    return last_title, None


def gen(keywords, verbose=True):
    keywords_dict, domains = process_data.pre_processing(keywords)
    title, count = generate_title_strict(keywords, keywords_dict, domains)
    title = capitalize_after_period(title)
    msg = "BART-" + "Keywords: " + ",".join(keywords) + " | Title: " + title
    return title, count


if __name__ == '__main__':
    # test_gen()
    # stat_token_lengths()
    generate_multiple()
