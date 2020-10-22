import sys
import os

input_dir = sys.argv[1]
input_file_list = os.listdir(input_dir)

for input_file in input_file_list:
    try:
        input_path = os.path.join(input_dir, input_file)
        with open(input_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                print(input_file.replace('.log', ''), end='\t')
                print(line)
    except:
        continue
