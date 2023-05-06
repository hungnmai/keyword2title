import os
import re
import random
import validators
from path import AI_PATH

# load bad words and good words
try:
    with open(os.path.join(AI_PATH, "blacklist.txt"), "r") as fp:
        blacklist = [word.strip().lower() for word in fp.readlines()]

    with open(os.path.join(AI_PATH, "whitelist.txt"), "r") as fp:
        whitelist = [word.strip() for word in fp.readlines()]
except Exception as e:
    print("Error: ", str(e))
    print("Could not load blacklist and whitelist files!")
    blacklist = []
    whitelist = []


def check_domain(s: str) -> bool:
    """
    Check string is domain
    :param s: input
    :return: True or False
    """
    if validators.domain(s):
        return True


def check_url(s: str) -> bool:
    """
    Check string is url
    :param s: input
    :return: True or False
    """
    if re.match(
            r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
            s):
        return True
    return False


def regex_url(s: str) -> bool:
    """
    Check string is domain or url
    :param s: input
    :return: True or False
    """
    if check_url(s) or check_domain(s):
        return True
    return False


def pre_processing(kws: list) :
    """
    Preprocess data before generating
    :param kws: list of keywords
    :return:
    """
    result = {}
    domains = []

    for tok in kws:
        origin = tok.strip()
        tok = tok.strip().lower().strip("/")
        result[tok] = origin

        if regex_url(tok):
            domains.append(tok)

    return result, domains


def post_processing(title: str, domains: list, keyword_normalized: dict) -> str:
    """
    Post processing after generating
    :param title: title được tạo ra bởi model
    :param domains: danh sách url hoặc domain
    :param keyword_normalized: {token_normalized, token origin}
    }
    :return:
    """
    title = title.replace("http ", "http").replace(": /", ":/").replace("https ", "https")

    tokens = title.split()
    for domain in domains:
        for tok in tokens:
            if domain in tok and len(tok) > len(domain):
                title = title.replace(tok, domain)
                tokens = title.split()

    for domain in domains:
        if domain in tokens:
            idx = tokens.index(domain)
            invalid_tokens = []
            for i in range(idx + 1, len(tokens)):
                tok = tokens[i]
                if "/" in tok or "." in tok:
                    invalid_tokens.append(tok)
                else:
                    break
            for i in range(idx - 1, -1, -1):
                tok = tokens[i]
                if "/" in tok or "." in tok or "www" in tok or "http" in tok:
                    invalid_tokens.append(tok)
                else:
                    break
            valid_tokens = []

            for tok in tokens:
                if tok not in invalid_tokens:
                    valid_tokens.append(tok)

            title = " ".join(valid_tokens)

    title = re.sub("\s+", " ", title)
    items = sorted(list(keyword_normalized.items()), key=lambda x: - len(x[0]))
    for tok, origin in items:
        # print(tok, origin)
        title = title.replace(tok, origin)
        # print(title)

    first_tok = title.split()[0]
    if regex_url(first_tok):
        return title
    return title.strip()


def replace_blacklist(kw: list, title: str) -> str:
    """
    Replace bad words in keyword by good words
    :param kw: list of keywords
    :param title: title after model generate
    :return: new title
    """
    kw = [tok.strip().lower() for tok in kw]
    words = title.split()
    list_words = []
    for word in words:
        tok = word.lower()
        if tok in blacklist and tok not in kw:
            word = random.choice(whitelist)
        else:
            for _b in blacklist:
                if _b in tok:
                    word = random.choice(whitelist)
        list_words.append(word)
    return " ".join(list_words)


def upper_title(title: str, spans: list = []) -> str:
    """
    Upper title
    :param title:
    :param spans:
    :return:
    """
    spans = [span.lower().strip() for span in spans]
    tokens = title.split()
    tokens_cleaned = []
    for tok in tokens:
        if not check_url(tok) and tok not in spans and "com" not in tok:
            if "." in tok and tok[-1] != ".":
                tok = ". ".join(tok.split("."))

        tokens_cleaned.append(tok)
    title = " ".join(tokens_cleaned)

    title = re.sub(' +', ' ', title.strip())
    sents = [sent[0].upper() + sent[1:] for sent in re.split("\. | \. ", title)]
    sents = [sent.strip() for sent in sents]
    title = ". ".join(sents)
    title = title.strip()

    return title


