import math
import numpy as np
from matplotlib import pyplot as plt

class TargetUtils:

    @staticmethod
    def short_date_format(row):
        row[0] = row[0][3:]
        return row

    @staticmethod
    def is_leap_year(year):
        """Determine whether a year is a leap year."""
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    def round_to25(n: float):
        """

        Args:
            n:

        Returns:

        Вспомогательная функция для специального округления для сетки геоданных
        """
        floor = math.floor(n)
        if abs(n - floor) <= 0.125:
            return floor
        elif abs(n - (floor + 0.125)) <= 0.125:
            return floor + 0.25
        elif abs(n - (floor + 0.5)) <= 0.125:
            return floor + 0.5
        elif abs(n - (floor + 0.75)) <= 0.125:
            return floor + 0.75
        else:
            return floor + 1
    
    def round_coord(self, row):
        """

        Args:
            row:

        Returns:

        Вспомогательная функция для преобразования координат сетки.
        """
        row[0] = self.round_to25(row[0])
        row[1] = self.round_to25(row[1])
        return row


def deg_min_to_dec(degrees, minutes):
  return degrees + minutes / 60.

def get_htc(tmean, pr, plot=True, x_pos=0, y_pos=0):
  """
  Computes Hydro-thermal coefficient for 3d arrays of tmean and pr
  
  Arguments:
    tmean (3darray): mean temperatures monthly
    pr (3darray): total precipitations monthly
    plot (bool): if to plot an example at coords x_pos, y_pos
    x_pos (int): latitude index
    y_pos (int): longitude index
  """
  #mask for months with temp greater than 10 degees C
  tmask = tmean >= 10

  tmean_masked = tmean * tmask
  pr_masked = pr * tmask

  htc = np.empty((pr_masked.shape[0] // 12, pr_masked.shape[1], pr_masked.shape[2]))
  for i in range(htc.shape[0]):
    htc[i, :, :] =  10 * pr_masked[12 * i : 12 * (i + 1), :, :].sum(axis = 0) / (30 * tmean_masked[12 * i : 12 * (i + 1), :, :].sum(axis = 0))
  if plot:
    plt.figure(figsize=(16, 9))

    plt.plot(htc[:, x_pos, y_pos])
    plt.legend()
    plt.ylabel('HTC value')
    plt.xlabel('Year')
    plt.grid()

  return htc