import os
import logging
from os import listdir, path

import polars as pl
from openai import OpenAI
from cachier import cachier
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

OPENAI = OpenAI()


@cachier()
def process_with_gpt(text: str) -> Result | None:
    context = [
        {"role": "system", "content": "必须以四字以内描述，简短为佳"},
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
        )
        content = completion.choices[0].message.parsed
        return content
    except KeyboardInterrupt:
        exit(1)
    except Exception as e:
        print(f"failed to process {text}:", e)
        return None


def process(text: str) -> dict | Result:
    DEFAULT = {"情感体验": None, "事件认知": None, "行为反应": None}

    result = process_with_gpt(text)
    logger.info(f"{text.__repr__()}: {result}")
    return result.model_dump() if result is not None else DEFAULT


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    for filename in (
        filename for filename in listdir(DATA_PATH) if filename.endswith(".xlsx")
    ):
        input_path = path.join(DATA_PATH, filename)
        output_path = path.join(OUTPUT_PATH, filename)
        df = pl.read_excel(input_path)

        df = df.limit(50)
        df = df.with_columns(
            pl.col("评论内容")
            .map_elements(process, return_dtype=pl.Struct, strategy="threading")
            .alias("result")
        ).unnest("result")
        df.write_excel(output_path)


if __name__ == "__main__":
    main()
