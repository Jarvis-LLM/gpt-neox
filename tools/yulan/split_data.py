import random
import json
import sys
import os
from IPython import embed
from tqdm import tqdm
import logging
import datetime

import logging
import datetime

# Create a custom formatter
class CustomFormatter(logging.Formatter):
    def format(self, record):
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        record.asctime = current_time
        return super().format(record)

# Create a logger and set the logging level
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a logging handler
handler = logging.StreamHandler()
formatter = CustomFormatter('%(asctime)s %(levelname)s: %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)





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




def count_num_line(file_name):
    from itertools import (takewhile, repeat)
    buffer = 1024 * 1024
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)

def concat_all_path(paths):
    cur_path = paths[0]
    for i in range(1, len(paths)):
        cur_path = os.path.join(cur_path, paths[i])
    return cur_path


def process_dataset(data_path, output_path, num_used_docs, expected_total_training_size, ratio, d_line):
    '''
    - data_path: the path contains the whole data
    - output_path: the path contains the training, validation, and test datasets for the current stage.
    - num_used_docs: the number of docs that has been used (trained). 
    - expected_total_training_size: the expected total training size.
    - ratio: the percent of the whole dataset that need to be extracted
    - d_line: stored the known num_lines of text file. Key: fpath, Value: num_line (int)
    '''
    
    logging.info('Processing: {} ...'.format(data_path))
    logging.info('Will be saved at: {}'.format(output_path))
    
    filenames = os.listdir(data_path)
    # only reserve json files
    filenames = [filename for filename in filenames if filename.split('.')[-1] == 'jsonl']
    filenames = sorted(filenames)
    total_size = 0
    start_line_idx = 0
    start_file_idx = 0
    accu_num_line = 0
    
    for i, filename in enumerate(filenames):
        fpath = os.path.join(data_path, filename)
        total_size += os.path.getsize(fpath)
        
        if accu_num_line > num_used_docs:
            continue
        if fpath in d_line:
            cur_num_line = d_line[fpath]
        else:
            cur_num_line = count_num_line(fpath)
            d_line[fpath] = cur_num_line
    
        logging.info("**The number of lines in {} is {}".format(fpath, cur_num_line))
        accu_num_line += cur_num_line
        
        if accu_num_line > num_used_docs:
            start_file_idx = i
            start_line_idx = cur_num_line - (accu_num_line - num_used_docs)
   
    assert accu_num_line > num_used_docs
    

    
    logging.info('\n------------------------------------------------------------\n')
    logging.info('real total data size: {}'.format(total_size))
    logging.info('expected total data size: {}'.format(expected_total_training_size))
    
    total_size = min(total_size, expected_total_training_size)

    needed_size = ratio * total_size
    train_needed_size = 0.998 * needed_size
    val_needed_size = 0.001 * needed_size
    test_needed_size = 0.001 * needed_size

    logging.info('needed ratio: {}, needed data size: {}'.format(ratio, needed_size))
    logging.info('num used docs: {}'.format(num_used_docs))
    logging.info('start file idx: {}, filename: {}'.format(start_file_idx,  os.path.join(data_path, filenames[start_file_idx])))
    logging.info('start line idx: {}'.format(start_line_idx))
    
    train_file = os.path.join(output_path, 'train.jsonl')
    val_file = os.path.join(output_path, 'val.jsonl')
    test_file = os.path.join(output_path, 'test.jsonl')
    
    logging.info('\n------------------------------------------------------------\n')
    logging.info('Performing data extraction...')
    logging.info('Train file: {}'.format(train_file))
    logging.info('Validation file: {}'.format(val_file))
    logging.info('Test file: {}'.format(test_file))
    
    with open(train_file, "w") as f_train, open(val_file, 'w') as f_val, open(test_file, 'w') as f_test:
        ok = False
        num_extracted_line = 0
        cur_line_idx = 0
        
        while start_file_idx < len(filenames):
            fpath = os.path.join(data_path, filenames[start_file_idx])
            logging.info('extracting from: {} ...'.format(fpath))
            
            with open(fpath, 'r') as fr:
                for line in tqdm(fr):
                    if cur_line_idx < start_line_idx:
                        cur_line_idx += 1
                        continue
                    
                    cur_train_size =  os.path.getsize(train_file)
                    cur_val_size = os.path.getsize(val_file)
                    cur_test_size = os.path.getsize(test_file)
                    
                    if cur_train_size < 0.7 * train_needed_size:
                        rnd = random.randint(1, 10000)
                        # val
                        if rnd >= 1 and rnd <= 10 and cur_val_size < val_needed_size:
                            f_val.write(line)
                        # test
                        elif rnd >= 11 and rnd <= 20 and cur_test_size < test_needed_size:
                            f_test.write(line)       
                        # train              
                        else:
                            f_train.write(line)
                    elif cur_val_size >= val_needed_size and cur_test_size >= test_needed_size:
                        f_train.write(line)
                    elif cur_val_size < cur_test_size:
                        f_val.write(line)
                    else:
                        f_test.write(line)

                    num_extracted_line += 1
                    cur_size = os.path.getsize(train_file) + os.path.getsize(val_file) + os.path.getsize(test_file) 
                    if cur_size >= needed_size:
                        ok = True
                        break
                    
            start_file_idx += 1
            
            if ok:
                break
        
        if not ok:
            logging.warning("All docs in the data have been used, no data can be further extracted to satisfy your need!")
        
        logging.info('\n------------------------------------------------------------\n')
        logging.info('Extracted Information:')
        logging.info('train file size: {}'.format(os.path.getsize(train_file)))
        logging.info('val file size: {}'.format(os.path.getsize(val_file)))
        logging.info('test file size: {}'.format(os.path.getsize(test_file)))
        logging.info('total numer of extracted docs in this time: {}'.format(num_extracted_line))
        num_used_docs += num_extracted_line
        logging.info('total number of docs that has been used (in order) for the dataset {} : {}'.format(data_path, num_used_docs))
        with open(os.path.join(output_path, "num_used_docs"), "w") as f:
            f.write(str(num_used_docs))
        logging.info("Has written the `num_used_docs` information in: {}".format(os.path.join(output_path, "num_used_docs")))
        
        with open(os.path.join(output_path, 'num_lines_per_text_file'), "w") as f:
            for fpath in d_line:
                f.write("{}\t{}".format(fpath, d_line[fpath]))
                f.write('\n')
        logging.info("Has written the `num_lines_per_text_file` information in: {}".format(os.path.join(output_path, "num_lines_per_text_file")))
        
        logging.info("Finished!".format())
        
        

if __name__ == "__main__":
    stage_idx = 0
    ratio = 0.2
    expected_training_disk_sizes = [60, 3.5, 45.5, 31, 60, 53.8, 10.4, 5.8, 81.8, 8.2, 290, 40, 40, 1171.73, 9.9, 1.4, 0.15, 0.82, 66, 750.00, 121.4, 28.6, 91, 1, 138, 100, 240, 50]

    for i, data_path in enumerate(all_data_paths):
        if not (i >= int(sys.argv[1]) and i < int(sys.argv[2])):
            continue
        name, version = data_path.split('/')
        output_path = concat_all_path([BASE_PATH, name, 'training', str(stage_idx)])
        os.makedirs(output_path)
        
        if stage_idx == 0:
            num_used_docs = 0
        else:
            last_training_record_path = concat_all_path([BASE_PATH, name, 'training', str(stage_idx - 1), "num_used_docs"])
            with open(last_training_record_path) as f:
                num_used_docs = int(f.readline().strip())

            last_training_record_path = concat_all_path([BASE_PATH, name, 'training', str(stage_idx - 1), "num_lines_per_text_file"])
            d_line = {}
            try:
                with open(last_training_record_path) as f:
                    for line in f:
                        fpath, num_line = f.readline().strip().split('\t')
                        d_line[fpath] = num_line
            except Exception as e:
                print(e.message)
        # run
        process_dataset(os.path.join(BASE_PATH, data_path), output_path, num_used_docs, expected_training_disk_sizes[i] * 1024 * 1024 * 1024, ratio, d_line)
        
        