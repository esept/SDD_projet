# Introduction 
Hugging Face platform is a renowned website in the field of artificial intelligence, highly regarded by developers and researchers. This platform is a hub where users share applications based on Transformer models, showcasing not only the current state of technology but also the future trends in artificial intelligence.

For this reason, I decided to utilize the datasets and model information available on the Hugging Face website for my research. By leveraging the Hugging Face Hub API, I have accessed all dataset information and data on models with more than 10 downloads from the site. This raw data includes rich details about "language," "tasks," and "timing." By analyzing this data in-depth, I believe insights can be gained into the characteristics of models for different task types, their peak development periods, and the advancements in artificial intelligence on Transformer architecture.

# Collect Data with `huggingface_hub`
```
from huggingface_hub import HfApi
import pandas as pd

api = HfApi()

# collect dataset's info
df_data = pd.DataFrame()
i = 0

for dataset in api.list_datasets(): # collect all dataset's information 
  i += 1
  data_info = {
      "name": dataset.id,
      "author": dataset.author, 
      "CreateTime": dataset.created_at,
      "Last Modified": dataset.last_modified,
      ...
  }

  df_data = pd.concat([df_data,pd.DataFrame([data_info])],ignore_index=True)
df_data.to_csv("./datasets.csv",index = False)


df = pd.DataFrame()
for model in api.list_models():
  if model.downloads > 10:  # collect times of download's model superieur than 10 
    model_data = {
        "Name": model.id,
        "Create_Time": model.created_at,
        "Last_Modified": model.last_modified,
        "private": model.private,
        "gated": model.gated,
        "Disabled": model.disabled,
        "Downloads": model.downloads,
				...
    }
    df = pd.concat([df,pd.DataFrame([model_data])],ignore_index=True)

df.to_csv("./df_10.csv",index = False)
```