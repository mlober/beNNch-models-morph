import model
import parameter_space
from parameter_space import p
import sys
import os
import json

data_folder_hash = sys.argv[1]

params = {
    'threads_per_task': {threads_per_task},  # total number of threads per process
    'num_processes': {num_processes}, # equal amount of areas and processes
    'num_areas': {num_areas},  # 
    'simtime': {model_time_sim},  # total simulation time in ms
    'presimtime': {model_time_presim},  # simulation time until reaching equilibrium
    'record_spikes': {record_spikes},  # switch to record spikes of excitatory neurons to file
    'rng_seed': {rng_seed},  # random number generator seed
    'min_delay': {min_delay},
    'max_delay': {max_delay},
    'threshold_delay': {threshold_delay},
    'morph': {morph},
    'step_data_keys': '{step_data_keys}'  # metrics to be recorded at each time step
}
step_data_keys = params['step_data_keys'].split(',')
params['nvp'] = params['threads_per_task'] * params['num_processes']

p.update(params)

p = parameter_space.calc_dependend_parameters(p)

p['data_path'] = os.path.join(p['data_path'], data_folder_hash)
try:
    os.mkdir(p['data_path'])
except:
    pass

### safe parameters to file

parameter_file = os.path.join(p['data_path'], 'parameter_dict.json')
with open(parameter_file, 'w') as f:
    json.dump(p, f)

# Create folder for storing simulation output
output_path = os.path.join(p['data_path'],
                        'recordings')
try:
    os.mkdir(output_path)
except:
    pass

### build and simulate model

model = model.Model(p)

model.simulate()
