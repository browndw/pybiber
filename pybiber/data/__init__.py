
from importlib_resources import files as _files

sources = {
    "micusp_mini": _files("pybiber") / "data/micusp_mini.parquet",
}


def __dir__():
    return list(sources)


def __getattr__(k):
    import polars as pl

    f_path = sources.get("micusp_mini")

    return pl.read_parquet(f_path)
