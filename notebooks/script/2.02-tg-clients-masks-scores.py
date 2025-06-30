#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import flwr
import torch
import dotenv

from fl_g13.fl_pytorch import build_fl_dependencies

from fl_g13.fl_pytorch.editing import get_client_masks, aggregate_by_sum, aggregate_masks, save_mask
from fl_g13.modeling import load_or_create

from fl_g13.architectures import BaseDino


dotenv.load_dotenv()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

build_fl_dependencies()


CHECKPOINT_DIR = dotenv.dotenv_values()['CHECKPOINT_DIR']

J = 8
partition_type = 'shard'
shards = [1, 10, 50]
mask_type = 'local'
mask_sparsity = 0.7
mask_rounds = 3
client_batch_size = 1


for s in shards:
    mask_name = f'{s}_{J}_{mask_type}_{mask_sparsity}_{mask_rounds}.pth'

    partition_name = 'iid' if partition_type == 'iid' else 'non-iid'
    model_save_path = CHECKPOINT_DIR + f"/fl/{partition_name}/{s}_{J}"

    model, _ = load_or_create(
        path=model_save_path,
        model_class=BaseDino,
        model_config=None,
        optimizer=None,
        scheduler=None,
        device=DEVICE,
        verbose=True
    )
    model.to(DEVICE)

    unfreeze_blocks = 12
    model.unfreeze_blocks(unfreeze_blocks)
    
    print(model_save_path, mask_name)

    masks, scores, _ = get_client_masks(
        ## config client data set params
        client_partition_type=partition_type,        # 'iid' or 'shard' for non-iid dataset
        client_num_partitions=100,                  # equal to number of client
        client_num_shards_per_partition=s,
        client_batch_size=client_batch_size,

        ## config get mask params
        mask_model=model,
        mask_sparsity=mask_sparsity,
        mask_type=mask_type,
        mask_rounds=mask_rounds,
        return_scores = True # Always return the scores
    )
    
    sum_mask = aggregate_by_sum(masks)
    print(f"Saving sum mask to: {CHECKPOINT_DIR + f'/masks/sum_{mask_name}'}")
    save_mask(sum_mask, CHECKPOINT_DIR + f'/masks/sum_{mask_name}')
    
    union_mask = aggregate_masks(masks, 'union')
    print(f"Saving union mask to: {CHECKPOINT_DIR + f'/masks/union_{mask_name}'}")
    save_mask(union_mask, CHECKPOINT_DIR + f'/masks/union_{mask_name}')
    
    intersection_mask = aggregate_masks(masks, 'intersection')
    print(f"Saving intersection mask to: {CHECKPOINT_DIR + f'/masks/intersection_{mask_name}'}")
    save_mask(intersection_mask, CHECKPOINT_DIR + f'/masks/intersection_{mask_name}')
    
    sum_scores = aggregate_by_sum(scores)
    for k in sum_scores.keys():
        sum_scores[k] /= 100 # average across clients
    print(f"Saving average fisher scores to: {CHECKPOINT_DIR + f'/masks/avgscores_{mask_name}'}")    
    save_mask(sum_scores, CHECKPOINT_DIR + f'/masks/avgscores_{mask_name}')

