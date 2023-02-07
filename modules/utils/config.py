from omegaconf import ListConfig, DictConfig


def enumerate_dict_config(conf: ListConfig, recurse=True):
    for item in conf:
        if isinstance(item, DictConfig):
            yield item
        elif recurse:
            assert isinstance(conf, ListConfig)
            yield from enumerate_dict_config(item)


def search_key(conf: ListConfig | DictConfig, key: str, recurse=True):
    if isinstance(conf, DictConfig):
        value = conf.get(key)
        if value is not None:
            yield value

        if not recurse:
            return

        for item in conf.values():
            if not (isinstance(item, ListConfig) or isinstance(item, DictConfig)):
                continue

            yield from search_key(item, key, True)
    elif recurse:
        assert isinstance(conf, ListConfig)
        for item in enumerate_dict_config(conf, False):
            yield from search_key(item, key, True)
