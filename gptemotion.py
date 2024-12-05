import os
import logging
from os import listdir, path

import polars as pl
from openai import OpenAI
from diskcache import Cache
from dotenv import load_dotenv
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__package__)


class Result(BaseModel):
    情感体验: str
    事件认知: str
    行为反应: str


load_dotenv()

DATA_PATH = os.environ["DATA_PATH"]
OUTPUT_PATH = os.environ["OUTPUT_PATH"]
RESULT_PATH = os.environ["RESULT_PATH"]

OPENAI = OpenAI()

cache = Cache("temp")


def process_with_gpt(text: str) -> Result | None:
    context = [
        {"role": "system", "content": "输出必须在五字以内，不包含标点符号"},
        {
            "role": "user",
            "content": text,
        },
    ]

    try:
        completion = OPENAI.beta.chat.completions.parse(
            messages=context,
            model="gpt-4o",
            n=1,
            temperature=0.2,
            response_format=Result,
            timeout=10.0,
        )
        content = completion.choices[0].message.parsed
        return content
    except KeyboardInterrupt:
        exit(1)
    except Exception as e:
        logger.error("failed to process %s:" % text, e)
        return None


def process(text: str) -> dict | Result:
    DEFAULT = {"情感体验": None, "事件认知": None, "行为反应": None}

    if text is None:
        return DEFAULT

    logger.info("started process for text %s" % text.__repr__())

    if text in cache and cache[text] is None:
        cache.pop(text)
    if text in cache:
        result = cache[text]
    else:
        result = process_with_gpt(text)
        if result is not None:
            cache[text] = result
    logger.info(f"{text.__repr__()}: {result}")
    return result.model_dump() if result is not None else DEFAULT


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    def find_excels(path: str):
        for filename in listdir(path):
            if not filename.endswith(".xlsx"):
                continue
            yield filename

    for filename in find_excels(DATA_PATH):
        input_path = path.join(DATA_PATH, filename)
        output_path = path.join(OUTPUT_PATH, filename)
        df = pl.read_excel(input_path)

        df = df.filter(pl.col("评论内容").str.len_bytes() != 0)
        df = df.with_columns(
            pl.col("评论内容")
            .map_elements(process, return_dtype=pl.Struct, strategy="threading")
            .alias("result")
        ).unnest("result")
        df.write_excel(output_path)

    total: pl.DataFrame = pl.concat(
        map(
            pl.read_excel,
            map(lambda p: path.join(OUTPUT_PATH, p), find_excels(OUTPUT_PATH)),
        )
    )

    os.makedirs(RESULT_PATH, exist_ok=True)
    for field in ["情感体验", "事件认知", "行为反应"]:
        total[field].value_counts(sort=True).with_columns(
            pl.col("count")
            .map_elements(lambda c: c * 100 / len(total), return_dtype=pl.Float64)
            .round(2)
            .alias("perc")
        ).write_excel(f"{RESULT_PATH}/{field}.xlsx")
    logger.info("processed %d comments" % len(total))


if __name__ == "__main__":
    main()
