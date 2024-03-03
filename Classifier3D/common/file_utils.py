import pandas as pd
import os


def save_csv(file, data, name=None):
    if not os.path.exists(file):
        os.mknod(file)

    data = pd.DataFrame(columns=name, data=data)
    data.drop_duplicates(inplace=True)
    data.to_csv(file, index=False)