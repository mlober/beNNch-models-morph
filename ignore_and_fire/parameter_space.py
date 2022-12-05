#import parameters as para
import numpy as np
import copy
import nest


def set_parameters(input_list: list):
    p={}

    p['data_path'] = input_list[1]
    p['data_folder_hash'] = input_list[2]

    p['scale'] = input_list[3]                        # scaling factor of the network size

    p['presimtime'] = input_list[4]                   # total network size = scale*11250 neurons
    p['simtime'] = input_list[5]                     # total simulation time in ms
    p['dt'] = 0.1                           # simulation step
    p['record_spikes'] = True               # switch to record spikes of excitatory
                                            # neurons to file
    p['rng_seed'] = input_list[6]                      # random number generator seed
    p['path_name'] ='.'                     # path where all files will have to be written
    p['log_file'] = 'logfile'               # naming scheme for the log files

    p['min_delay'] = input_list[7]
    p['max_delay'] = input_list[8]
    p['threshold_delay'] = input_list[9]

    p['morph_distribution'] = input_list[10]

    ############# Model parameters ##############################################

    p['model_params'] = {}
    p['model_params']['rate'] = input_list[11]
    p['model_params']['phase'] = nest.random.uniform(0, 1)

    ############# Network parameters ############################################


    p['num_areas'] = input_list[12]
    p['areas_list'] = []
    for i in range(p['num_areas']):
        string = 'area_' + str(i)
        p['areas_list'].append(string)

    p['original'] = {}
    p['original']['withinarea'] = {}
    p['original']['withinarea']['N_total'] = int(80000 * p['scale'])
    p['original']['withinarea']['Nrec'] = int(min(p['original']['withinarea']['N_total'], 1000))         # number of neurons to record spikes from
    p['original']['withinarea']['delay_mean'] = p['threshold_delay']/2.
    p['original']['withinarea']['delay_rel_std'] = 0.5                                        
    p['original']['withinarea']['indegree'] = int(0.05 * p['original']['withinarea']['N_total'])         # total num connections per neuron

    p['original']['interareal'] = {}
    p['original']['interareal']['delay_mean'] = (p['max_delay'] + p['threshold_delay']) / 2.
    p['original']['interareal']['delay_rel_std'] = 0.5
    if p['num_areas'] > 1:
        p['original']['interareal']['indegree'] = int(p['original']['withinarea']['indegree'] * 1/(p['num_areas'] - 1))


    ############# Set all parameters to default "original"-Parameters ###########

    p['network_params'] = {}
    for area_pre in p['areas_list']:
        p['network_params'][area_pre] = p['original']['withinarea'].copy()
        for area_post in p['areas_list']:
            if area_pre != area_post:
                p['network_params'][area_pre][area_post] = p['original']['interareal'].copy()

    ############# Adjust parameters of specific areas ###########################

    # example:
    # p['network_params']['area_30']['N_total'] = 10.


    ############# Update dependend parameters ###################################

    if p['num_areas'] <= 1:
        update_dependend_parameters(p, p['network_params'][p['areas_list'][0]])
    else:
        for area_pre in p['areas_list']:
            for area_post in p['areas_list']:
                if area_pre != area_post:
                    p['network_params'][area_pre], p['network_params'][area_pre][area_post] = update_dependend_parameters(p, p['network_params'][area_pre], p['network_params'][area_pre][area_post])

    return p

############# Function to call when deviating from default parameters #######

def update_dependend_parameters(p, input_dict_withinarea, input_dict_interareal=np.nan):
    input_dict_withinarea['indegree'] = int(0.05 * input_dict_withinarea['N_total'])
    input_dict_withinarea['Nrec'] = int(min(input_dict_withinarea['N_total'], 1000)) 
    if p['num_areas'] > 1:
        input_dict_interareal['indegree'] = int(input_dict_withinarea['indegree'] * 1/(p['num_areas'] - 1))
    return input_dict_withinarea, input_dict_interareal


