def rename_keys(source: dict, key_dict: dict):
    return {key_dict.get(k, k): v for k, v in source.items()}


def get_class(name: str):
    import importlib
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)
