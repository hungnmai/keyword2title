import logging
from fastapi import FastAPI
from ai.bart_kw_title import gen
from ai.process_data import replace_blacklist, upper_title

app = FastAPI()

logging.basicConfig(filename="app.log", format='%(asctime)s - %(message)s', level=logging.INFO)


@app.get("/")
@app.get("/ping")
def ping():
    return "I am alive!"


@app.get("/generate_title_from_kw_v1")
def generate_title_from_kw(spans: str):
    keywords = [kw.strip() for kw in spans.split(",") if len(kw.strip()) > 0]
    title, count = gen(keywords)
    title = replace_blacklist(keywords, title)
    title = upper_title(title, keywords)
    return {
        "spans": spans,
        "title": title,
    }
