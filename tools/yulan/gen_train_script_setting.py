import random
import json
import sys
import os
from IPython import embed
from tqdm import tqdm
from pprint import pprint

    

def concat_all_path(paths):
    cur_path = paths[0]
    for i in range(1, len(paths)):
        cur_path = os.path.join(cur_path, paths[i])
    return cur_path



BASE_PATH = "/fs/archive/share/yulan/data"

# # 28 sources: [cn:11, en:16, multi:1]
all_data_paths = [
    "zhihu-qa-cn/v1",
    "wiki-baike-cn/v3",
    "bdbk-baike-cn/v2",
    "tiger-baike-cn/v2",
    "legal_case-law-cn/v2",
    "cc-web-cn/v2",
    "cc100-web-cn/v2",
    "clueweb22-web-cn/v2",
    "cbooks-book-cn/v2",
    "bestsellers-book-cn/v2",
    "cicg-news-cn/v3",
    "stack_exchange-qa-en/v2",
    "wiki-baike-en/v2",
    "refinedweb-web-en/v1",
    "cc-web-en/v2",
    "cc100-web-en/v2",
    "clueweb22-web-en/v2",
    "ccstories-web-en/v2",
    "openwebtext2-web-en/v2",
    "c4-web-en/v1",
    "books3-book-en/v1",
    "gutenberg-book-en/v2",
    "realnews-news-en/v2",
    "ccnews-news-en/v1",
    "cicg-news-en/v2",
    "arxiv-paper-en/v2",
    "github-code-en/v1",
    "wiki-baike-multi/v1",
]


# Set your expected total disk size for each dataset during the whole training.
expected_disk_sizes = [60, 3.5, 45.5, 31, 60, 53.8, 10.4, 5.8, 81.8, 8.2, 290, 40, 40, 1171.73, 9.9, 1.4, 0.15, 0.82, 66, 750, 121.4, 28.6, 91, 1, 138, 100, 240, 50]

# get the real disk size for each datset of the current training stage.
mode = "train"  # set this
assert mode in ["train", "val"]

stage_idx = 0   # set this
stage_ratio = 0.2 # set this
mode_ratio = 0.001 if mode == "val" else 0.998

real_disk_sizes = []
for data_path in all_data_paths:
    name, version = data_path.split('/')
    fpath = concat_all_path([BASE_PATH, name, 'training', str(stage_idx), '{}.jsonl'.format(mode)])
    real_disk_sizes.append(os.path.getsize(fpath))

expected_disk_sizes = [mode_ratio * stage_ratio * x * 1024 * 1024 * 1024 for x in expected_disk_sizes]

# This weight is used to multiply the tokens_per_dataset, which reflect 
# If the mode is training, this weight should not be different from too much with the below reference training weight. If it does, there mush be some bugs.
# Training reference weights: [1.07, 2.06, 2.07, 2.07, 1.0, 1.05, 1.05, 1.05, 1.6, 1.61, 1.01, 1.03, 2.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.19, 1.19, 1.0, 1.0, 1.0, 1.03, 1.0, 1.0]
# this weights for training should not be different from too much with the reference weight. If it does, there mush be some bugs. Note that val is ok to different from the reference weight.
weights = [expected_disk_sizes[i] / real_disk_sizes[i] for i in range(len(expected_disk_sizes))]
pprint(weights) # just for checking

# get this from the trainig log.
tokens_per_dataset = [
         3882973059,   
         123100879,    
         1543137856,   
         1089551512,   
         4080838760,   
         3489491150,   
         642364406,    
         379386272,    
         3964565194,   
         367769664,    
         19025801608,  
         2382465757,   
         861030794,
         64886714388,    
         516767002,    
         70762579,     
         6905526,      
         39366963,     
         3441393335,
         39602233014,   
         5873654754,   
         1445536400,   
         4859519942,   
         54400596,     
         7328814962,   
         6915773312,   
         17046333298,  
         3120967670
     ]   


assert len(tokens_per_dataset) == len(weights)
total_tokens = [tokens_per_dataset[i] * weights[i] for i in range(len(weights))]    # weighted tokens per dataset
print("total tokens (B) used (has been weighted): {}".format(sum(total_tokens) / 1e9))   # total tokens used, which has been weighted. 

real_weights_to_write = [round(x / sum(total_tokens)*100, 6) for x in total_tokens]
print("You should write the below weight in the `{}-data-weight` of the training script (setup.yml).".format(mode))
print(real_weights_to_write)
