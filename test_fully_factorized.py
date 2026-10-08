import numpy as np
import pandas as pd
from synthyverse.generators import UnivariateGenerator


x_cat = np.random.choice(5, size=(1000, 2))
x_num = np.random.normal(size=(1000, 2))
x_num_2 = np.random.choice(50, (1000,))
x = np.column_stack((x_num, x_num_2, x_cat))
df = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])

self = UnivariateGenerator()
self.fit(df, discrete_features=["x3", "x4"])
self.generate(1000)


