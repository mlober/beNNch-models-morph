import model, helper
from parameter_space import set_parameters
import sys
import os
import json

p = set_parameters(input_list=sys.argv)

os.mkdir(os.path.join(p['data_path'], p['data_folder_hash']))

### safe parameters to file

parameter_file = os.path.join(p['data_path'], 'parameter_dict.json')
            
with open(parameter_file, 'w') as f:
    json.dump(p, f)

# Create folder for storing simulation output
os.mkdir(os.path.join(p['data_path'],
                        p['data_folder_hash'],
                        'recordings'))

### build model

model = model.Model(p)

model.create()

model.connect()

### simulate model

model.simulate()
