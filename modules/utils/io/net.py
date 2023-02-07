from pathlib import Path

import requests


def get_string(link_or_path: str):
    if link_or_path.startswith("http://") or link_or_path.startswith("https://"):
        with requests.Session() as session:
            content = session.get(link_or_path).content.decode("utf-8")
    elif Path(link_or_path).exists():
        with open(link_or_path, "r") as f:
            content = f.read()
    else:
        raise ValueError(f'"{link_or_path}" is not a valid link or path')

    return content
