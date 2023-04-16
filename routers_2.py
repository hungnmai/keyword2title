import logging
from fastapi import FastAPI
from ai.t5_kw_title import get_longest_title
from ai.process_data import replace_blacklist, upper_title

app = FastAPI()

logging.basicConfig(filename="app.log", format='%(asctime)s - %(message)s', level=logging.INFO)


@app.get("/")
@app.get("/ping")
def ping():
    return "I am alive!"


@app.get("/generate_title_from_kw_v2")
def generate_title_from_kw_v2(spans: str):
    try:
        spans = [kw.strip() for kw in spans.split(",") if len(kw.strip()) > 0]
        title = get_longest_title(spans)
        title = replace_blacklist(spans, title)
        title = upper_title(title, spans)
        logging.info("T5-" + "Keywords: " + ",".join(spans) + " | Title: " + title)

        return {
            "spans": spans,
            "title": title.strip(),
        }

    except Exception as e:
        logging.error("T5: " + str(e))
