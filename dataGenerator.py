import pandas as pd
import numpy as np
np.random.seed(42)

n = 500
tv = np.random.randint(0, 200, size=n)
radio = np.random.randint(0, 50, size=n)
social_media = np.random.randint(0, 100, size=n)


sales = 10 + 0.5 * tv + 0.8 * radio + 0.3 * social_media + np.random.normal(0, 5, size=n)

data = {
    'TV': tv,
    'Radio': radio,
    'Social_Media': social_media,
    'Sales': sales
}
df = pd.DataFrame(data)

df.to_csv('sales_data.csv', index=False)
