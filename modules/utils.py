def rename_keys(source: dict, key_dict: dict):
    return {key_dict.get(k, k): v for k, v in source.items()}
