#!/usr/bin/env python
# coding: utf-8

import wandb
## read .env file
import dotenv

dotenv.load_dotenv()


# login by key in .env file
WANDB_API_KEY = dotenv.dotenv_values()["WANDB_API_KEY"]
wandb.login(key=WANDB_API_KEY)


import wandb
import pandas as pd

# Connect to the existing run
api = wandb.Api()
url='stefano-gamba-social-politecnico-di-torino/FL_Dino_CIFAR100_base_new_batchsize64/runs/FL_Dino_Baseline_non_iid_5_4_zt'
run = api.run(url)

# Download history
history_df = run.history()


history_df


filtered_df = history_df[history_df['_step'] <= 130]
filtered_df


# Keep only steps ≤ 150
# filtered_df = history_df[history_df['_step'] <= 130]
filtered_df = history_df

project_name='FL_Dino_CIFAR100_base_new_batchsize64'
name = 'FL_Dino_Baseline_non_iid_5_4'
# Start a new run
new_run = wandb.init(project=project_name, name=name,id=name)

# Re-log filtered data
for _, row in filtered_df.iterrows():
    # Remove internal W&B columns if present
    step=int(row['_step'])
    row = row.drop(labels=[col for col in row.index if col.startswith('_')], errors='ignore')
    print(step)
    print(row.to_dict())
    wandb.log(row.to_dict(), step=step)

new_run.finish()




