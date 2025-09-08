from importlib_resources import files as _files

# Map logical names to packaged data file paths
sources = {
    "micusp_mini": _files("pybiber") / "data/micusp_mini.parquet",
    "biber_loadings": _files("pybiber") / "data/biber_loadings.csv",
}


def __dir__():
    return list(sources)


def __getattr__(k):
    import polars as pl

    f_path = sources.get(k)
    if f_path is None:
        raise AttributeError(f"module 'pybiber.data' has no attribute '{k}'")

    path_str = str(f_path)
    if path_str.endswith(".parquet"):
        return pl.read_parquet(f_path)
    if path_str.endswith(".csv"):
        return pl.read_csv(f_path)
    raise AttributeError(
        f"Unsupported resource type for '{k}': {path_str}"
    )
