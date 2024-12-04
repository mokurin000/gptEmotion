import os
from os import listdir, path

import polars as pl
from openai import OpenAI
from cachier import cachier
from dotenv import load_dotenv
from pydantic import BaseModel

from tqdm import tqdm

PROGRESS_BAR = None


class Result(BaseModel):
    情感体验: str
    事件认知: str
    行为反应: str


load_dotenv()

DATA_PATH = os.environ["DATA_PATH"]
OUTPUT_PATH = os.environ["OUTPUT_PATH"]

OPENAI = OpenAI()

os.environ["POLARS_MAX_THREADS"] = "32"


@cachier()
def process_with_gpt(text: str) -> Result | None:
    context = [
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
    if PROGRESS_BAR is not None:
        PROGRESS_BAR.update()
    return result.model_dump() if result is not None else DEFAULT


def main():
    global PROGRESS_BAR

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    for filename in (
        filename for filename in listdir(DATA_PATH) if filename.endswith(".xlsx")
    ):
        input_path = path.join(DATA_PATH, filename)
        output_path = path.join(OUTPUT_PATH, filename)
        df = pl.read_excel(input_path)

        total = len(df)
        with tqdm(total=total, desc=filename.removesuffix(".xlsx")) as pbar:
            PROGRESS_BAR = pbar
            df = df.with_columns(
                pl.col("评论内容")
                .map_elements(process, return_dtype=pl.Struct, strategy="threading")
                .alias("result")
            ).unnest("result")
        PROGRESS_BAR = None
        df.write_excel(output_path)


if __name__ == "__main__":
    main()
