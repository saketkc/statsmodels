from __future__ import print_function
from statsmodels.compat import lzip
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

prestige = sm.datasets.get_rdataset("Duncan", "car", cache=True).data
prestige_model = ols("prestige ~ education -1", data=prestige).fit()


fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(prestige_model, "education", fig=fig)
fig.savefig('prestige~education-1.png')

prestige_model = ols("prestige ~ education", data=prestige).fit()
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(prestige_model, "education", fig=fig)
fig.savefig('prestige~income.png')

