from pathlib import Path

import pandas as pd

main_path = Path("./data/raw/data_raw.xlsx")
sheet_names = ["UX", "GP", "AS"]
df = pd.DataFrame()
for name in sheet_names:
    data = pd.read_excel(main_path, sheet_name=name)[
        ["Комментарии", "Эмоциональная окраска"]
    ]
    df = pd.concat([df, data])

print(df.info())
df.to_csv("./data/interim/data_read.csv")
