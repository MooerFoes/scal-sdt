import psutil


def physical_core_count():
    return psutil.cpu_count(logical=False)
