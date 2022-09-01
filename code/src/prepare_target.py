import pandas as pd
import numpy as np
import glob
import math
from utils import TargetUtils as tu


DATA_PATH = "data/ojdamage_rus.xls"
REGION = "Субъект Российской Федерации "
CUR_REGION = "Тамбовская область"
EVENTNAME_COL = "Название явления "
STARTDATE = "Дата начала "
EVENTTYPE = "Град"
MONTH_COL = "Month"
YEAR_COL = "Year"
TARGET = "target"

IMPORTANT_COLS = [
        "BEGIN_DATE_TIME",
        "BEGIN_LAT",
        "BEGIN_LON",
    ]


class MeteoruDB:

    def get_target(self, data_path = DATA_PATH, region = CUR_REGION):
        data = pd.read_excel(data_path)
        hail_data = data[data[EVENTNAME_COL] == EVENTTYPE].reset_index().drop(columns="index")[[STARTDATE, REGION]]
        hail_data = hail_data[hail_data[REGION] == region].reset_index().drop(columns="index")
        hail_data[STARTDATE] = hail_data[[STARTDATE]].apply(tu.short_date_format, axis=1)
        hail_data = hail_data.drop_duplicates()
        hail_data[STARTDATE] = pd.to_datetime(hail_data[STARTDATE], format="%m.%Y")  # , dayfirst = True)
        hail_data = hail_data.sort_values(by=[STARTDATE])
        hail_data = hail_data.drop(columns=[REGION])
        hail_data[TARGET] = np.ones(hail_data.shape[0], dtype=int)
        hail_data = hail_data.set_index([STARTDATE])
        idx = pd.date_range(min(hail_data.index), max(hail_data.index), freq='MS')
        hail_data = hail_data.reindex(idx, fill_value=0)
        return hail_data


class NoaaDB:

    def get_grid(dataframe: pd.DataFrame, lat: tuple, long: tuple, year: int):
        """

        Args:
            dataframe:
            lat:
            long:
            year:

        Returns:

        Получение сетки из данных о штормовых событиях
        """
        num_of_days = 365

        if tu.is_leap_year(year):
            num_of_days = 366

        step = 0.25
        lat_grid = np.arange(27, 37 + step, step)[::-1]
        long_grid = np.arange(-109, -93. + step, step)

        lat_to_idx = {}.fromkeys(lat_grid)
        long_to_idx = {}.fromkeys(lat_grid)

        for i, lat_ in enumerate(lat_grid):
            lat_to_idx[lat_] = i
        for j, long_ in enumerate(long_grid):
            long_to_idx[long_] = j

        hail_df = dataframe[dataframe.EVENT_TYPE == "Hail"].reset_index().drop(columns=["index"])[IMPORTANT_COLS]
        hail_df["BEGIN_DATE_TIME"] = pd.to_datetime(hail_df["BEGIN_DATE_TIME"])
        hail_df["DOY"] = hail_df["BEGIN_DATE_TIME"].dt.dayofyear
        hail_df = hail_df.drop(columns=["BEGIN_DATE_TIME"])
        hail_df = hail_df[hail_df["BEGIN_LAT"] < lat[1]]
        hail_df = hail_df[hail_df["BEGIN_LAT"] > lat[0]]
        hail_df = hail_df[hail_df["BEGIN_LON"] < long[1]]
        hail_df = hail_df[hail_df["BEGIN_LON"] > long[0]]
        hail_df = hail_df.reset_index().drop(columns=["index"])
        hail_df = hail_df.apply(tu.round_coord, axis=1)
        hail_df = hail_df.drop_duplicates().reset_index().drop(columns=["index"])
        hail_np = hail_df.to_numpy()
        grid = np.zeros((num_of_days, len(lat_grid), len(long_grid)))

        for row in hail_np:
            grid[int(row[2]), lat_to_idx[row[0]], long_to_idx[row[1]]] = 1.0
            try:
                grid[int(row[2]), lat_to_idx[row[0]] + 1, long_to_idx[row[1]]] = 1.0
                grid[int(row[2]), lat_to_idx[row[0]] - 1, long_to_idx[row[1]]] = 1.0
                grid[int(row[2]), lat_to_idx[row[0]], long_to_idx[row[1]] + 1] = 1.0
                grid[int(row[2]), lat_to_idx[row[0]], long_to_idx[row[1]] - 1] = 1.0
                grid[int(row[2]), lat_to_idx[row[0]] + 1, long_to_idx[row[1]] + 1] = 1.0
                grid[int(row[2]), lat_to_idx[row[0]] - 1, long_to_idx[row[1]] + 1] = 1.0
                grid[int(row[2]), lat_to_idx[row[0]] + 1, long_to_idx[row[1]] - 1] = 1.0
                grid[int(row[2]), lat_to_idx[row[0]] - 1, long_to_idx[row[1]] - 1] = 1.0
            except IndexError:
                pass
        return grid


def prepare_target_grid(path: str, lat: tuple, long: tuple):
    """

    Args:
        path:
        lat:
        long:

    Returns:

    Подготовка таргетной сетки

    """
    target_paths = sorted(glob.glob(path + "/*"))
    grids = []
    years = [i for i in range(2016, 2022)]
    for path, year in zip(target_paths, years):
        dataframe = pd.read_csv(path)
        grids.append(tu.get_grid(dataframe, lat, long, year))
    grids = np.concatenate(grids, axis=0)
    return grids



