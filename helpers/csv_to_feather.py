from helpers.read_file import read_csv


def write_csv_to_feather(csv_path: str, feather_path: str, **kwargs) -> None:
    df = read_csv(csv_path, **kwargs)
    df.to_feather(feather_path)


if __name__ == "__main__":
    import os.path as osp

    DATA_PATH_BASE = "./data"

    TRAIN_PATH_CSV = osp.join(DATA_PATH_BASE, "train.csv")
    TRAIN_PATH_FEATHER = osp.join(DATA_PATH_BASE, "train.fth")

    KWARGS = {"parse_dates": []}

    for pair in [
        (TRAIN_PATH_CSV, TRAIN_PATH_FEATHER),
    ]:
        write_csv_to_feather(*pair, **KWARGS)
