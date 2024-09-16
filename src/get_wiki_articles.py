# %%
# 参考
# https://github.com/Azure-Samples/graphrag-accelerator/blob/main/notebooks/get-wiki-articles.py
# 実行：ターミナルで以下を実行
# python3 script/get_wiki_articles.py

# %%
# !/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This script downloads a few sample wikipedia articles that can be used for demo or quickstart purposes in conjunction with the solution accelerator.
"""

import argparse
import os

import wikipedia

wikipedia.set_lang("ja")
# %%

choose_article = [
    "豊臣秀吉",
    "織田信長",
    "徳川家康",
    "明智光秀",
]


def main():
    parser = argparse.ArgumentParser(description="Wikipedia Download Script")
    parser.add_argument(
        "--directory",
        help="Directory to download sample wikipedia articles to.",
        default="/workspace/input",
    )
    parser.add_argument(
        "--short-summary",
        help="Retrieve short summary article content.",
        action="store_true",
    )
    parser.add_argument(
        "--num-articles",
        help="Number of wikipedia articles to download. Default=5",
        default=5,
        choices=range(1, 6),
        type=int,
    )
    args = parser.parse_args()
    os.makedirs(args.directory, exist_ok=True)
    for state in choose_article[0:args.num_articles]:
        try:
            title = wikipedia.page(state).title.lower().replace(" ", "_")
            content = (
                wikipedia.page(state).summary
                if args.short_summary
                else wikipedia.page(state).content
            )
            content = content.strip()
            filename = os.path.join(args.directory, f"{title}_wiki_article.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Saving wiki article '{title}' to {filename}")
        except Exception:
            print(f"Error fetching wiki article {title}")

# %%


if __name__ == "__main__":
    main()
