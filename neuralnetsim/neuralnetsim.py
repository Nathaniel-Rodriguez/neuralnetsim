import networkx as nx
import nest
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import utilities

class NeuralNetSimDataContainer(object):

    def __init__(self):
        pass

class NeuralNetSim(object):
    """
    Things to implement: input signal into specific node list
    THings to implement: parameter distributions - pynest offers built in 
    distributions that let you do this (see help(nest.Connect) 
    and help(nest.Create), etc. -- this program already lets you do this by
    setting the appropriate parameters for connection_pars and neuron_pars
    """

    def __init__(self, network, **kwargs):
        """
        Network must be a weighted networkx Digraph.
        inhib_node_list must be a list of networkx node IDs
        synapse and neuron parameters dictionaries must have key=nx node ID
        and value=dict() of parameters

        external_noise_generator: ["noise_generator", "poisson_generator", ...]
        noise_generator = [{'dt', 'std', 'mean'}, ...]
        poisson_generator = [{'rate'}, {"start": 100.0, "stop": 150.0}...]
        external_noise_connection_pars:[{'model', 'weight', 'delay'}]

        neuron_noise_dict: keys are nx nodes, value is a dictionary with
        noise parameters with keys: generator, generator_pars, connection_pars

        signal_generators: ["dc_generator", "spike_generator", ...]
        signal_generator_pars: [ {'amplitude', 'start', 'stop'}, {'spike_times', 'spike_weights'}, ...]
        signal_connection_pars: [ {'model', 'weight', 'delay'}]
        """

        self.network = network # The network is not changed by NNS in any way.

        valid_keys = set(['neuron_type', 'ex_neuron_parameters', 'seed',
            'inhib_neuron_parameters', 'synapse_type',
            'IE_synapse_parameters', 'EE_synapse_parameters',
            'II_synapse_parameters', 'EI_synapse_parameters',
            'inhib_node_list', 'dt', 'inhib_fraction', 'initial_Vm_range',
            'ensure_neg_inhib', 'signal_connection_pars',
            'external_noise_parameters', 'weight_scale', 'external_noise_generators',
            'parameter_noise', 'use_communities', 'external_noise_generator',
            'external_noise_connection_pars', 'neuron_noise_dict',
            'signal_generators', 'signal_generator_pars', 'num_trials', 
            'sim_time', 'inhib_weight_scale', 'weight_key', 'community_key',
            'synapse_parameter_dict', 'neuron_parameter_dict', 'signal_nodes',
            'signal_connection_ratio', 'signal_community', 'volt_detectors',
            'use_network_weights', 'verbosity'])

        property_defaults = { 'neuron_type': 'iaf_neuron', 
            'seed': 1,
            'verbosity': 30,
            'ex_neuron_parameters': None,
            'inhib_neuron_parameters': None,
            'synapse_type': 'static_synapse',
            'IE_synapse_parameters': None,
            'EE_synapse_parameters': None,
            'II_synapse_parameters': None,
            'EI_synapse_parameters': None,
            'inhib_node_list': [],
            'dt': 0.1,
            'use_network_weights':True,
            'inhib_fraction': 0.0,
            'initial_Vm_range': None,
            'ensure_neg_inhib': True,
            'neuron_noise_dict': None,
            'external_noise_generators': [],
            'external_noise_parameters': [],
            'external_noise_connection_pars': [],
            'weight_scale': 1.0,
            'parameter_noise': False,
            'weight_key': 'weight',
            'signal_nodes': None,
            'signal_connection_ratio': 0.0,
            'signal_generators': [],
            'signal_generator_pars': [],
            'signal_connection_pars': [],
            'use_communities': False,
            'signal_community': 1,
            'num_trials': 1,
            'sim_time': 1000.0,
            'inhib_weight_scale': 0.0, 
            'inhib_fraction': 0.0,
            'community_key': 'community',
            'synapse_parameter_dict': None,
            'neuron_parameter_dict': None,
            'volt_detectors': False }

        for key in kwargs.keys():
            if key not in valid_keys:
                raise KeyError(key + " not a valid key")

        for key, default in property_defaults.iteritems():
            setattr(self, key, kwargs.get(key, default))

        # weights
        if self.synapse_parameter_dict != None:
            if self.use_network_weights:
                for edge in self.network.edges_iter():
                    self.synapse_parameter_dict[edge[0]][edge[1]].update( {'weight': self.weight_scale * \
                        self.network[edge[0]][edge[1]][self.weight_key]})

        # Results
        self.SpikeDetectorOutput_byTrial = []
        self.VoltageDetectorOutput_byTrial = []

        # Set nest output
        nest.set_verbosity(self.verbosity)

    def SetModelParameters(self):
        """
        """

        # Default synapse and neuron parameters (any parameters given in
        # a list will be changed from these defaults - this way global
        # synapse and neuron parameters can be set through default, while
        # local ones can be set via the list)
        if self.ex_neuron_parameters == None:
            self.ex_neuron_parameters = self.RemoveReadOnlyKeys(nest.GetDefaults(self.neuron_type))
        if self.inhib_neuron_parameters == None:
            self.inhib_neuron_parameters = self.RemoveReadOnlyKeys(nest.GetDefaults(self.neuron_type))
        if self.IE_synapse_parameters == None:
            self.IE_synapse_parameters = self.RemoveReadOnlyKeys(nest.GetDefaults(self.synapse_type))
        if self.EI_synapse_parameters == None:
            self.EI_synapse_parameters = self.RemoveReadOnlyKeys(nest.GetDefaults(self.synapse_type))
        if self.II_synapse_parameters == None:
            self.II_synapse_parameters = self.RemoveReadOnlyKeys(nest.GetDefaults(self.synapse_type))
        if self.EE_synapse_parameters == None:
            self.EE_synapse_parameters = self.RemoveReadOnlyKeys(nest.GetDefaults(self.synapse_type))

        # Internal parameters
        self.N = len(self.network)
        if len(self.inhib_node_list) > 0:
            self.N_I = len(self.inhib_node_list)
        else:
            self.N_I = int(self.N * self.inhib_fraction)
        self.N_E = self.N - self.N_I

        # Set default neuron params (will be changed on a neuron by neuron basis
        # if list of parameters is given or if inhibitory neurons exist)
        nest.CopyModel(self.neuron_type, "exh_neuron", self.ex_neuron_parameters)
        nest.CopyModel(self.neuron_type, "inh_neuron", self.inhib_neuron_parameters)
        # Set defaults for synapses ('''')
        nest.CopyModel(self.synapse_type, "EE_synapse", self.EE_synapse_parameters)
        nest.CopyModel(self.synapse_type, "IE_synapse", self.IE_synapse_parameters)
        nest.CopyModel(self.synapse_type, "EI_synapse", self.EI_synapse_parameters)
        nest.CopyModel(self.synapse_type, "II_synapse", self.II_synapse_parameters)

    def RemoveReadOnlyKeys(self, dictionary):

        read_only_list = ["t_spike", "thread", "thread_local_id", "vp",
            "available", "capacity", "type_id", "elementsize", "instantiations",
            "min_delay", "max_delay", "sizeof", "synapsemodel", "num_connections"]

        for key in read_only_list:
            if key in dictionary:
                dictionary.pop(key, None)

        return dictionary

    def GetCommunities(self):

        return set([ self.network.node[node][self.community_key] 
            for node in self.network.nodes_iter() ])

    def GetCommunityNodeDict(self, set=False):
        """
        Returns a dictionary with communities as keys and list of (nx) nodes as
        values. This gives a list of nodes for each community
        """

        if set:
            community_dict = {}
            for node in self.network.nodes_iter():
                community = self.network.node[node][self.community_key]
                if community in community_dict:
                    community_dict[community].append(node)
                else:
                    community_dict[community] = [node]

            return { com: set(node_list) for com, node_list in community_dict.items() }
            
        else:
            community_dict = {}
            for node in self.network.nodes_iter():
                community = self.network.node[node][self.community_key]
                if community in community_dict:
                    community_dict[community].append(node)
                else:
                    community_dict[community] = [node]

        return community_dict

    def GetCommunityNodes(self, community):
        """
        Returns a list of nodes for a given community
        """

        return [ node for node in self.network.nodes_iter() 
            if community == self.network.node[node][self.community_key]]

    def ConvertNodeToNest(self, sequence_nodes):
        """
        Returns a list of nx nodes converted to nest gIDs
        *requires nest neurons to have already been constructed
        """

        return [ self.NodeToNest[node] for node in sequence_nodes ] 

    def ConvertNestToNode(self, sequence_neurons):
        """
        Returns a list of nest gIDS to nx nodes
        *requires nest neurons to have already been constructed
        """

        return [ self.NestToNode[neuron] for neuron in sequence_neurons ]

    def GenerateIDtranslator(self, nest_ordered=True):
        """
        Requires self.neurons_ex, self.neurons_in, self.in_node_set, 
        and self.ex_node_set to be defined.

        nest_ordered assumes that neurons are created according to nests
        ordering, which is exhititory neurons first and inhibitory neurons
        second

        if nest_ordered==False, then it is assumed neurons are created 
        in the order of the network. So the first node in the network
        will correspond to the first nest ID. Additionally, because
        the network is used it doesn't require a parameter setting
        for each neuron if neuron_parameter_dict is given.
        """

        self.NodeToNest = {}
        self.NestToNode = {}
        counter = 1

        if nest_ordered:
            if self.neuron_parameter_dict == None:
                for node in list(self.ex_node_set):
                    self.NodeToNest[node] = counter 
                    self.NestToNode[counter] = node 
                    counter +=1
                for node in list(self.in_node_set):
                    self.NodeToNest[node] = counter 
                    self.NestToNode[counter] = node 
                    counter += 1
            else:
                for node in self.neuron_parameter_dict.keys():
                    if node in self.ex_node_set:
                        self.NodeToNest[node] = counter
                        self.NestToNode[counter] = node
                        counter += 1

                for node in self.neuron_parameter_dict.keys():
                    if node in self.in_node_set:
                        self.NodeToNest[node] = counter
                        self.NestToNode[counter] = node 
                        counter += 1

        else:
            for node in self.network.nodes_iter():
                self.NodeToNest[node] = counter 
                self.NestToNode[counter] = node 
                counter += 1

    def ConstructNeuralNetwork(self):
        """
        Builds nest nodes and connections
        """

        self.SetModelParameters()

        if len(self.inhib_node_list) == 0:
            self.in_node_set = set(np.random.choice(self.network.nodes(), size=self.N_I, replace=False))
        else:
            self.in_node_set = set(self.inhib_node_list)

        self.ex_node_set = set([ node for node in self.network.nodes_iter() if node not in self.in_node_set])
        self.all_node_set = self.ex_node_set.union(self.in_node_set)

        self.ConstructNeurons()
        self.ConstructConnections()
        if self.signal_generators != []:
            self.ConstructSignalElements()
        if self.external_noise_generators != []:
            self.ConstructNoiseElements()
        self.ConstructDetectors()

    def ConstructNeurons(self):
        """
        Makes nest neurons for the network
        """

        if self.neuron_parameter_dict == None:
            if self.N_E > 0:
                self.neurons_ex = nest.Create("exh_neuron", self.N_E)
            else:
                self.neurons_ex = ()

            if self.N_I > 0:
                self.neurons_in = nest.Create("inh_neuron", self.N_I)
            else:
                self.neurons_in = ()
        else:
            if self.N_E > 0:
                self.neurons_ex = nest.Create("exh_neuron", self.N_E, 
                    [ params for node, params in self.neuron_parameter_dict.iteritems() 
                    if node in self.ex_node_set ] )
            else:
                self.neurons_ex = ()

            if self.N_I > 0:
                self.neurons_in = nest.Create("inh_neuron", self.N_I, 
                    [ params for node, params in self.neuron_parameter_dict.iteritems()
                    if node in self.in_node_set ])
            else:
                self.neurons_in = ()

        self.all_neurons = self.neurons_ex + self.neurons_in
        self.GenerateIDtranslator()

    def ConstructConnections(self):
        """
        Makes nest synapses for the network (requires nest nodes to connect)
        """

        for edge in self.network.edges_iter():

            source_neuron = [self.NodeToNest[edge[0]]]
            target_neuron = [self.NodeToNest[edge[1]]]

            if self.synapse_parameter_dict == None:
                if (edge[0] in self.ex_node_set) and (edge[1] in self.ex_node_set):
                    nest.Connect(source_neuron, target_neuron, \
                        syn_spec={'model':"EE_synapse", 
                        'weight': self.weight_scale * self.network[edge[0]][edge[1]][self.weight_key]})
                elif (edge[0] in self.ex_node_set) and (edge[1] in self.in_node_set):
                    nest.Connect(source_neuron, target_neuron, syn_spec={'model':"EI_synapse", 
                        'weight': self.weight_scale * self.network[edge[0]][edge[1]][self.weight_key]})
                elif (edge[0] in self.in_node_set) and (edge[1] in self.ex_node_set):
                    nest.Connect(source_neuron, target_neuron, syn_spec={'model':"IE_synapse", 
                        'weight': self.weight_scale * self.network[edge[0]][edge[1]][self.weight_key]})
                elif (edge[0] in self.in_node_set) and (edge[1] in self.in_node_set):
                    nest.Connect(source_neuron, target_neuron, syn_spec={'model': "II_synapse",
                        'weight': self.weight_scale * self.network[edge[0]][edge[1]][self.weight_key]})
            else:
                nest.Connect(source_neuron, target_neuron, syn_spec=self.synapse_parameter_dict[edge[0]][edge[1]])

    def ConstructSignalElements(self):
        """
        Supports arguments for any kind of nest signal device
        """

        for i, generator in enumerate(self.signal_generators):
            self.signal_node = nest.Create(generator, params=self.signal_generator_pars[i])

            if self.signal_nodes == None:
                num_inputs = int(self.N * self.signal_connection_ratio)
                if num_inputs < 1:
                    num_inputs = 1

                if self.use_communities:
                    community_neurons = set([ self.NodeToNest[node] for node in self.GetCommunityNodes(self.signal_community) ])
                    self.input_neurons = list(community_neurons.intersection(set(self.neurons_ex)))
                else:
                    self.input_neurons = list(self.neurons_ex)

                if len(self.input_neurons) < num_inputs:
                    print( "Warning: Only ", str(self.input_neurons),
                        " available. Require ", str(num_inputs), ". Inputs set to max available.")
                    num_inputs = len(self.input_neurons)

                self.input_neurons = list(np.random.choice(self.input_neurons, size=num_inputs, replace=False))
                nest.Connect(self.signal_node, self.input_neurons, conn_spec='all_to_all', syn_spec=self.signal_connection_pars[i])

            else:
                # Convert nodes to nest neuron IDs
                self.input_neurons = [ self.NodeToNest[node] for node in self.signal_nodes ]
                nest.Connect(self.signal_node, self.input_neurons, 'all_to_all', self.signal_connection_pars[i])

    def ConstructNoiseElements(self):
        """
        """

        if self.neuron_noise_dict == None:
            self.noise_nodes = []
            for i, generator in enumerate(self.external_noise_generators):
                self.noise_nodes += nest.Create(generator, params=self.external_noise_parameters[i])
                nest.Connect([self.noise_nodes[-1]], list(self.all_neurons), 
                    "all_to_all", self.external_noise_connection_pars[i])
        else:
            self.noise_nodes = []
            for node in self.neuron_noise_dict.keys():
                self.noise_nodes += nest.Create(self.neuron_noise_dict[node]['generator'], 
                    params=self.neuron_noise_dict[node]['generator_pars'])
                nest.Connect([self.noise_nodes[-1]], [NodeToNest[node]], "all_to_all",
                    syn_spec=self.neuron_noise_dict[node]['connection_pars'])

    def ConstructDetectors(self):
        """
        Creates detectors for both communities individually and for the whole
        network. Both spiking and voltage detectors are created.
        """

        if self.use_communities:
            dictCom_listNodes = self.GetCommunityNodeDict()
            self.ComSpikeDetectors = {}
            self.ComVoltDetectors = {}
            for community, nodes in dictCom_listNodes.iteritems():
                self.ComSpikeDetectors[community] = nest.Create("spike_detector", 
                    params={'to_memory':True, 'to_file':False, 'withtime':True, 
                    'withgid':True})
                com_neurons = self.ConvertNodeToNest(nodes)
                nest.Connect(com_neurons, self.ComSpikeDetectors[community], "all_to_all")

                if self.volt_detectors:
                    self.ComVoltDetectors[community] = nest.Create("multimeter",
                        params={'to_memory':True, 'to_file':False, 'interval': self.dt,
                        'to_accumulator':False, 'withtime':True, 'record_from':['V_m']})

                    nest.Connect(self.ComVoltDetectors[community], com_neurons, "all_to_all")

        self.SpikeDetector = nest.Create("spike_detector", 
                    params={'to_memory':True, 'to_file':False, 'withtime':True, 
                    'withgid':True})
        nest.Connect(self.neurons_ex + self.neurons_in, self.SpikeDetector, "all_to_all")

        if self.volt_detectors:
            self.VoltDetector = nest.Create("multimeter",
                        params={'to_memory':True, 'to_file':False, 'interval': self.dt,
                        'to_accumulator':False, 'withtime':True, 'record_from':['V_m']})

            nest.Connect(self.VoltDetector, self.neurons_ex + self.neurons_in, "all_to_all")

    def RecordTrial(self, trial, wrap=False, buffer=False):
        """
        Records events for the trial and pushes them to the appropriate
        list. Also restructures data so that it can be used by other functions
        more easily.
        """

        if wrap:

            if buffer:
                pass
            else:
                pass
        else:
            spike_detector_output = { 'all': self.RestructureEventDictionary(self.SpikeDetector, "spike") }
            if self.use_communities:
                for com in self.ComSpikeDetectors.keys():
                    spike_detector_output[com] = self.RestructureEventDictionary(self.ComSpikeDetectors[com], "spike")
            self.SpikeDetectorOutput_byTrial.append(spike_detector_output)

            if self.volt_detectors:
                voltage_detector_output = { 'all': self.RestructureEventDictionary(self.VoltDetector, 'voltage')}
                if self.use_communities:
                    for com in self.ComVoltDetectors.keys():
                        voltage_detector_output[com] = self.RestructureEventDictionary(self.ComVoltDetectors[com], "voltage")
                self.VoltageDetectorOutput_byTrial.append(voltage_detector_output)

    def Engage(self, wrap=False, buffer=False, state_reset=False):
        """
        If kernel_reset is true, then all other options are disregarded,
        as resetting the kernel recreates the entire neural network.

        If wrap is true, then buffer and state_reset options will determine if
        some of the network's states are reverted to their initial values
        (except dynamic synapses and perhaps some neurons). If buffer is true,
        then a buffer time must be given as buffer=X ms as a float. During this
        time recordings will be discarded.
        """

        for i in xrange(self.num_trials):
            # Resets all nodes/synapses/detectors to initial states (except noise generators)
            # (except dyn synapses and--newer NEST will support this)
            # nest.SetKernelStatus({'time':0.0}) # resets time because resetnetwork doesn't do this
            # nest.ResetNetwork()

            # Moves this stuff here because reset kernel needs to come before setkernel status
            # to prevent parallel process conflicts... may adjust in future if wrap is needed
            # Additionally, if we want to keep the same inputs from trial to trial
            # we can move the input calculation stuff to a function that runs
            # in __init__ so that it doesn't get recalculated on constructnetwork
            # Construct network should only do the following: copymodels, create and connect
            # all other stuff involving parameter calculations should be done in a different function in __init__
            nest.ResetKernel()
            nest.SetKernelStatus({'rng_seeds': [self.seed + i], 'resolution': self.dt, 
                'overwrite_files': True, 'print_time':False })
            self.ConstructNeuralNetwork()
            nest.Simulate(self.sim_time)
            self.RecordTrial(i, wrap, buffer)

            # if i < self.num_trials-1:
            #     if not wrap:
            #         nest.ResetKernel()
            #         nest.SetKernelStatus({'rng_seeds': [self.seed + i], 'resolution': self.dt, 
            #             'overwrite_files': True, 'print_time':False })
            #         self.ConstructNeuralNetwork()

            #     else:

            #         if buffer:
            #             pass
            #         else:
            #             pass
            #         if state_reset:
            #             nest.ResetNetwork()

    def RestructureEventDictionary(self, detector, detector_type, convertNestToNode=True):
        """
        Event dictionaries for individual detectors recording from many neurons
        agglomerate times, voltages, senders, etc, into the same arrays. This
        function separates the senders (ie. who the detector is logging from)
        and their data into separate arrays. The resulting dictionary has 
        senders as keys and all other data related to that sender is inside.
        """

        event_dict = {}
        if detector_type == 'spike':
            raw_event_dict = nest.GetStatus(detector)[0]['events']

            # Loop through all events and add them to dictionary
            for i in xrange(len(raw_event_dict['senders'])):
                if raw_event_dict['senders'][i] not in event_dict:
                    event_dict[raw_event_dict['senders'][i]] = [raw_event_dict['times'][i]]
                else:
                    event_dict[raw_event_dict['senders'][i]].append(raw_event_dict['times'][i])

            for key in event_dict.keys():
                event_dict[key] = np.array(event_dict[key])

        elif detector_type == 'voltage':

            raw_event_dict = nest.GetStatus(detector)[0]['events']

            # Loop through all events and add them to dictionary
            event_dict = {}
            for i in xrange(len(raw_event_dict['senders'])):
                if raw_event_dict['senders'][i] not in event_dict:
                    event_dict[raw_event_dict['senders'][i]] = [(raw_event_dict['V_m'][i] , raw_event_dict['times'][i])]
                else:
                    event_dict[raw_event_dict['senders'][i]].append((raw_event_dict['V_m'][i] , raw_event_dict['times'][i]))

            for key in event_dict.keys():

                V_m, times = zip(*event_dict[key])
                event_dict[key] = (np.array(V_m), np.array(times))

        if convertNestToNode:
            return self.ConvertEventDictionary(event_dict)
        else:
            return event_dict

    def ConvertEventDictionary(self, event_dict, nest_to_node=True):
        """
        Converts the keys of an event dictionary to nx nodes of the original
        network by creating a new dictionary with new keys
        """

        return { self.NestToNode[neuron] : value for neuron, value in event_dict.iteritems() }

    def CalculateISI(self, spike_times):

        return spike_times[1:] - spike_times[:-1]

    def ReturnResults(self, result_type='SpikeTimes', com=False, avg=True, **kwargs):
        """

        NOTE: IN FUTURE WE WOULD LIKE TO MAKE ALL DATATYPES BE THE SAME, SO COM and
        NON-COM WILL BE THE SAME. SO WHEN COMS DON'T EXIST COM=1

        results_types: SpikeTimes, VoltageTraces, OnceFire, Activity, ISI, PSTH

        Returns a datastructure with the desired result type in a dictionary.
        The dictionary has nx node keys that match the node ids of the original
        network. Values vary depending upon result type, com, and whether it is
        averaged. non-average gets a 2D array, while avg just gets 1D array of
        values. Com gets a dictionary with keys that correspond to the original
        networks communities (with results embedded). SpikeTimes get an array of
        spike times, traces get an array of voltages and an array of times,
        once fire gets a scalar, and activity gets an array of times and an
        array of values. Activity requires an extra argument 'binsize' from 
        which to determine integrate spikes over.

        *assumes result dictionaries have been converted from nest neuron gIDs
        to nx node IDs

        'dt' needed for activity and psth
        """

        if result_type == 'SpikeTimes':
            
            if com:
                dictCom_setNodes = self.GetCommunityNodeDict()
                dictCom_dictNodes_listTrials_arraySpikeTimes = { com : {} for com in self.GetCommunities() }
                for com in dictCom_dictNodes_listTrials_arraySpikeTimes.keys():
                    for node in self.network.nodes_iter():
                        dictCom_dictNodes_listTrials_arraySpikeTimes[com][node] = []
                        for event_dict in self.SpikeDetectorOutput_byTrial:
                            if node in dictCom_setNodes[com]:
                                dictCom_dictNodes_listTrials_arraySpikeTimes[com][node].append(event_dict[com][node])
                            else:
                                dictCom_dictNodes_listTrials_arraySpikeTimes[com][node].append([])
                return dictCom_dictNodes_listTrials_arraySpikeTimes

            else:
                dictNodes_listTrials_arraySpikeTimes = {}
                for node in self.network.nodes_iter():
                    dictNodes_listTrials_arraySpikeTimes[node] = []
                    for event_dict in self.SpikeDetectorOutput_byTrial:
                        if node in event_dict['all']:
                            dictNodes_listTrials_arraySpikeTimes[node].append(event_dict['all'][node])
                        else:
                            dictNodes_listTrials_arraySpikeTimes[node].append([ ])

                return dictNodes_listTrials_arraySpikeTimes

        elif result_type == 'VoltageTraces':
            if com:
                if avg:
                    pass
                else:
                    pass
            else:
                if avg:
                     # All times should be same so just pick first one
                    times = np.copy(self.VoltageDetectorOutput_byTrial[0]['all'][self.network.nodes()[0]][1])
                    dictNodes_arrayVoltage = {}
                    for node in self.network.nodes_iter():
                        arrayTrials_arrayVoltage = np.zeros((self.num_trials, len(times)))
                        for i, event_dict in enumerate(self.VoltageDetectorOutput_byTrial):
                            arrayTrials_arrayVoltage[i][:] = event_dict['all'][node][0]

                        dictNodes_arrayVoltage[node] = np.mean(arrayTrials_arrayVoltage, axis=0)
                    return dictNodes_arrayVoltage, times

                else:
                     # All times should be same so just pick first one
                    times = np.copy(self.VoltageDetectorOutput_byTrial[0]['all'][self.network.nodes()[0]][1])
                    dictNodes_arrayTrials_arrayVoltage = { node: np.zeros((self.num_trials, len(times))) }
                    for node in self.network.nodes_iter():
                        for i, event_dict in enumerate(self.VoltageDetectorOutput_byTrial):
                            if times != None:
                                times = event_dict['all'][node][1]
                            dictNodes_arrayTrials_arrayVoltage[i][:] = event_dict['all'][node][0]

                    return dictNodes_arrayTrials_arrayVoltage, times

        elif result_type == 'OnceFire':
            if com:
                if avg:
                    dictCom_oncefire_activity = { com : 0.0 for com in self.GetCommunities() }
                    for com in dictCom_oncefire_activity.keys():
                        for event_dict in self.SpikeDetectorOutput_byTrial:
                            dictCom_oncefire_activity[com] += len(event_dict[com]) / float(len(self.network))
                        dictCom_oncefire_activity[com] /= len(self.SpikeDetectorOutput_byTrial)
                    return dictCom_oncefire_activity
                else:
                    dictCom_oncefire_activity = { com : [] for com in self.GetCommunities() }
                    for com in dictCom_oncefire_activity.keys():
                        for event_dict in self.SpikeDetectorOutput_byTrial:
                            dictCom_oncefire_activity[com].append(len(event_dict[com]) / float(len(self.network)))
                        dictCom_oncefire_activity[com] = np.array(dictCom_oncefire_activity[com])
                    return dictCom_oncefire_activity
            else:
                if avg:
                    avg_oncefire_activity = 0.0
                    for event_dict in self.SpikeDetectorOutput_byTrial:
                        avg_oncefire_activity += len(event_dict['all']) / float(len(self.network))
                    return avg_oncefire_activity / len(self.SpikeDetectorOutput_byTrial)
                else:
                    oncefire_activity = []
                    for event_dict in self.SpikeDetectorOutput_byTrial:
                        oncefire_activity.append(len(event_dict['all']) / float(len(self.network)))
                    return np.array(oncefire_activity)

        elif result_type == 'Activity': # page 177 of book: 1/deltaT * 1/num_neurons * num_spikes_in_network_in_window(deltaT)
            bins = np.arange(0.0, self.sim_time + kwargs['dt'], kwargs['dt'])
            if com:
                if avg:
                    dictCom_arrayActivities = { com : {} for com in self.GetCommunities() }
                    for com in dictCom_arrayActivities.keys():
                        arrayTrials_arrayActivities = np.zeros((self.num_trials, len(bins)-1))
                        for i, event_dict in enumerate(self.SpikeDetectorOutput_byTrial):
                            spike_freqs_by_nodes = []
                            for node in self.GetCommunityNodes(com):
                                if node in event_dict[com]:
                                    spike_freqs_by_nodes.append(np.histogram(event_dict[com][node], bins=bins)[0])
                                else:
                                    spike_freqs_by_nodes.append(np.histogram([], bins=bins)[0])

                            arrayTrials_arrayActivities[i][:] = \
                                np.mean(spike_freqs_by_nodes, axis=0) / kwargs['dt'] * 1000.0

                        dictCom_arrayActivities[com] = np.mean(arrayTrials_arrayActivities, axis=0)

                    return dictCom_arrayActivities, bins

                else:
                    dictCom_arrayTrials_arrayActivities = { com: np.zeros((self.num_trials, len(bins)-1)) 
                        for com in self.GetCommunities()}
                    for com in dictCom_arrayTrials_arrayActivities.keys():
                        for i, event_dict in enumerate(self.SpikeDetectorOutput_byTrial):
                            spike_freqs_by_nodes = []
                            for node in self.GetCommunityNodes(com):
                                if node in event_dict[com]:
                                    spike_freqs_by_nodes.append(np.histogram(event_dict[com][node], bins=bins)[0])
                                else:
                                    spike_freqs_by_nodes.append(np.histogram([], bins=bins)[0])

                            dictCom_arrayTrials_arrayActivities[com][i][:] = \
                                np.mean(spike_freqs_by_nodes, axis=0) / kwargs['dt'] * 1000.0

                    return dictCom_arrayTrials_arrayActivities, bins
            else:
                
                if avg:
                    arrayTrials_arrayActivities = np.zeros((self.num_trials, len(bins)-1))
                    for i, event_dict in enumerate(self.SpikeDetectorOutput_byTrial):
                        spike_freqs_by_nodes = []
                        for node in self.network.nodes_iter():
                            if node in event_dict['all']:
                                spike_freqs_by_nodes.append(np.histogram(event_dict['all'][node], bins=bins)[0])
                            else:
                                spike_freqs_by_nodes.append(np.histogram([], bins=bins)[0])

                        arrayTrials_arrayActivities[i][:] = np.mean(spike_freqs_by_nodes, axis=0) / kwargs['dt'] * 1000.0

                    return np.mean(arrayTrials_arrayActivities, axis=0), bins
                else:
                    arrayTrials_arrayActivities = np.zeros((self.num_trials, len(bins)-1))
                    for i, event_dict in enumerate(self.SpikeDetectorOutput_byTrial):
                        spike_freqs_by_nodes = []
                        for node in self.network.nodes_iter():
                            if node in event_dict['all']:
                                spike_freqs_by_nodes.append(np.histogram(event_dict['all'][node], bins=bins)[0])
                            else:
                                spike_freqs_by_nodes.append(np.histogram([], bins=bins)[0])

                        arrayTrials_arrayActivities[i][:] = np.mean(spike_freqs_by_nodes, axis=0) / kwargs['dt'] * 1000.0

                    return arrayTrials_arrayActivities, bins

        elif result_type == 'ISI':
            dictNodes_arrayISI = {}
            for node in self.network.nodes_iter():
                for event_dict in self.SpikeDetectorOutput_byTrial:
                    if node in event_dict['all']:
                        if node in dictNodes_arrayISI:
                            dictNodes_arrayISI[node] = np.append(dictNodes_arrayISI[node], 
                                self.CalculateISI(event_dict['all'][node]))
                        else:
                            dictNodes_arrayISI[node] = self.CalculateISI(event_dict['all'][node])

            return dictNodes_arrayISI

        elif result_type == 'PSTH': # page 176 of book: 1/deltaT * 1/num_trials * num_spikes_of_neuron_n_in_window(deltaT)
            bins = np.arange(0.0, self.sim_time + kwargs['dt'], kwargs['dt'])
            if com:
                dictCom_dictNodes_arrayPSTH = { com : {} for com in self.GetCommunities() }
                for com in dictCom_dictNodes_arrayPSTH.keys():
                    for node in self.GetCommunityNodes(com):
                        spike_freqs_by_trial = []
                        for event_dict in self.SpikeDetectorOutput_byTrial:
                            if node in event_dict[com]:
                                spike_freqs_by_trial.append(np.histogram(event_dict[com][node], bins=bins)[0])
                            else:
                                spike_freqs_by_trial.append(np.histogram([], bins=bins)[0])

                        dictCom_dictNodes_arrayPSTH[com][node] = np.mean(spike_freqs_by_trial, axis=0) / kwargs['dt'] * 1000.0
                
                return dictCom_dictNodes_arrayPSTH, bins     

            else:
                dictNodes_arrayPSTH = {}
                for node in self.network.nodes_iter():
                    spike_freqs_by_trial = []
                    for event_dict in self.SpikeDetectorOutput_byTrial:
                        if node in event_dict['all']:
                            spike_freqs_by_trial.append(np.histogram(event_dict['all'][node], bins=bins)[0])
                        else:
                            spike_freqs_by_trial.append(np.histogram([], bins=bins)[0])

                    dictNodes_arrayPSTH[node] = np.mean(spike_freqs_by_trial, axis=0) / kwargs['dt'] * 1000.0

                return dictNodes_arrayPSTH, bins

    def SaveDataToFile(self, prefix):
        """
        Saves spike events and voltages to file
        """

        exclude_nest_keys = set(['ex_neuron_parameters', 'inhib_neuron_parameters',
            'IE_synapse_parameters', 'EI_synapse_parameters', 'II_synapse_parameters',
            'EE_synapse_parameters'])
        utilities.save_object({key : nns.__dict__[key] for key in nns.__dict__.keys() \
            if key not in exclude_nest_keys }, prefix + "_NNS_results.pyobj")

    def PlotVoltageTrace(self, prefix, nodes='all'):
        """
        """

        dictNodes_arrayVoltage, times = self.ReturnResults('VoltageTraces', avg=True)

        if nodes == 'all':
            nodes = dictNodes_arrayVoltage.keys()

        for node in nodes:
            plt.clf()
            plt.plot(times, dictNodes_arrayVoltage[node], ls='-', marker='None')
            plt.xlabel('time (ms)')
            plt.ylabel('potential (mV)')
            plt.savefig(prefix + "_" + str(node) + "_voltage-trace.png", dpi=300)
            plt.clf()
            plt.close()

    def PlotSpiketrain(self, prefix, trials='all', window=None):

        if trials == 'all':
            trials = range(self.num_trials)

        if self.use_communities:

            for j, trial_num in enumerate(trials):
                dictCom_dictNodes_listTrials_arraySpikeTimes = self.ReturnResults('SpikeTimes', com=True)
                communities = dictCom_dictNodes_listTrials_arraySpikeTimes.keys()
                plot_posy = 0
                
                # Loop through each community and plot to order nodes by community
                plt.clf()
                # fig, axes = plt.subplots(figsize=(5,5), dpi=300)
                for i, community in enumerate(communities):
                    # Plot ts for each node
                    for node, spike_times in dictCom_dictNodes_listTrials_arraySpikeTimes[community].iteritems():
                        try:
                            if window != None:
                                points_to_plot = [ time for time in spike_times[j] if time > window[0] and time < window[1] ]
                            else:
                                points_to_plot = spike_times[j]
                        except IndexError:
                            plot_posy += 1
                        
                        plt.plot(points_to_plot, plot_posy * np.ones(len(points_to_plot)),
                            linestyle='None', marker='o', color=utilities.generate_color(i), 
                            markersize=2.0, markeredgewidth=0.0)
                        plot_posy += 1

                plt.ylim(0, plot_posy)
                if window != None:
                    plt.xlim(window[0], window[1])
                plt.xlabel('time (ms)')
                plt.savefig(prefix + '_' + str(trial_num) + '_spike_train.png', dpi=300)
                plt.close()
                plt.clf()

        else:

            for i, trial_num in enumerate(trials):
                dictNodes_listTrials_arraySpikeTimes = self.ReturnResults('SpikeTimes', com=False)
                nodes = dictNodes_listTrials_arraySpikeTimes.keys()
                if len(nodes) == 0:
                    continue
                plt.clf()
                # fig, axes = plt.subplots(figsize=(5,5), dpi=300)
                # Assumes node IDs are numbers... which could be dangerous but is usually the case
                for node, spike_times in dictNodes_listTrials_arraySpikeTimes.iteritems():
                    try:
                        if window != None:
                            points_to_plot = [ time for time in spike_times[i] if time > window[0] and time < window[1] ]
                        else:
                            points_to_plot = spike_times[i]
                    except IndexError:
                        continue

                    plt.plot(points_to_plot, node * np.ones(len(points_to_plot)),
                        linestyle='None', marker='o', 
                        color='black', markersize=2.0, markeredgewidth=0.0)

                plt.ylim(min(nodes)-0.25, max(nodes)+0.25)
                if window != None:
                    plt.xlim(window[0], window[1])
                plt.xlabel('time (ms)')
                plt.savefig(prefix + '_' + str(trial_num) + '_spike_train.png', dpi=300)
                plt.close()
                plt.clf()

    def PlotNetworkActivity(self, prefix, dt=5.0):

        if self.use_communities:
            dictCom_arrayActivities, bins = self.ReturnResults('Activity', True, True, dt=dt)
            times = (bins[1:] + bins[:-1])/2.
            plt.clf()
            for com, activity in dictCom_arrayActivities.iteritems():
                plt.plot(times, activity, ls='-', marker='None', color=utilities.generate_color(com))

            plt.xlabel('time (ms)')
            plt.ylabel("Activity")
            plt.savefig(prefix + '_network_activity.png', dpi=300)
            plt.close()
            plt.clf()
        else:
            arrayActivities, bins = self.ReturnResults('Activity', False, True, dt=dt)
            times = (bins[1:] + bins[:-1])/2.
            plt.clf()
            plt.plot(times, arrayActivities, ls='-', marker='None', color='black')
            plt.xlabel('time (ms)')
            plt.ylabel("Activity")
            plt.savefig(prefix + '_network_activity.png', dpi=300)
            plt.close()
            plt.clf()

    def PlotISIdistribution(self, prefix, nodes='all', x_log=False, y_log=False):
        """
        """

        dictNodes_arrayISI = self.ReturnResults('ISI')

        if nodes == 'all':
            nodes = dictNodes_arrayISI.keys()

        for node in nodes:
            utilities.plot_ccdf(prefix + '_' + str(node) + '_ISI_dist', 
                dictNodes_arrayISI[node], "interval (ms)", x_log=x_log, y_log=y_log)

    def PlotPSTH(self, prefix, nodes='all', dt=5.0):
        """
        """

        dictNodes_arrayPSTH, bins = self.ReturnResults('PSTH', dt=dt)

        if nodes == 'all':
            nodes = dictNodes_arrayPSTH.keys()

        for node in nodes:
            plt.clf()
            plt.bar(bins[:-1], dictNodes_arrayPSTH[node], bins[1:] - bins[:-1])
            plt.xlabel('time (ms)')
            plt.ylabel(r'$\rho$')
            plt.savefig(prefix + "_" + str(node) + "_psth.png", dpi=300)
            plt.clf()
            plt.close()

class NeuronFittingSim(NeuralNetSim):
    """
    A child class of NeuralNetSim that focuses on simulating circuits of a network
    given some input spike trains. Input nodes must be designated which are members
    of the network but will be created as parrots and recieve the provided spike trains.
    The parrots are then connected to the rest of the nodes as defined by the graph.
    Both parrots (the original input signal spike train) and all other nodes are recorded.
    Noise is only input into the remaining nodes of the network. They
    are not connected to the input nodes (parrots). 

    *Signals are not currently supported - they will get input into parrots [may add signal function if I need it]
    *Delay between the spike and the parrot is set automatically to the simulation resolution.
    """

    def __init__(self, network, input_nodes, input_spike_trains, **kwargs):
        super(NeuronFittingSim, self).__init__(network, **kwargs)
        self.input_nodes = input_nodes
        self.input_spike_trains = input_spike_trains
        self.signal_generators = [True] # signal must be provided in terms of input nodes and spike trains

    def ConstructNeurons(self):
        """
        Makes all input nodes parrot_neurons
        """
        
        self.parrots = ()
        self.neurons_ex = ()
        self.neurons_in = ()
        for node in self.network.nodes_iter():
            if node in self.input_nodes:
                parrot = nest.Create("parrot_neuron")
                self.parrots += parrot
                if node in self.ex_node_set:
                    self.neurons_ex += parrot
                elif node in self.in_node_set:
                    self.neurons_in += parrot
            else:
                if node in self.ex_node_set:
                    neuron = nest.Create("exh_neuron", 1, [ self.neuron_parameter_dict[node] ])
                    self.neurons_ex += neuron 
                elif node in self.in_node_set:
                    neuron = nest.Create("inh_neuron", 1, [ self.neuron_parameter_dict[node] ])
                    self.neurons_in += neuron

        self.all_neurons = self.neurons_ex + self.neurons_in
        self.GenerateIDtranslator(nest_ordered=False)

    def ConstructNoiseElements(self):
        """
        Create noise elements but only attach them to non-parrot neurons
        """

        if self.neuron_noise_dict == None:
            self.reciever_set = set(self.all_neurons).difference(set(self.parrots))
            self.noise_nodes = []
            for i, generator in enumerate(self.external_noise_generators):
                self.noise_nodes += nest.Create(generator, params=self.external_noise_parameters[i])
                nest.Connect([self.noise_nodes[-1]], list(self.reciever_set), 
                    "all_to_all", self.external_noise_connection_pars[i])
        else:
            self.noise_nodes = []
            for node in self.neuron_noise_dict.keys():
                self.noise_nodes += nest.Create(self.neuron_noise_dict[node]['generator'], 
                    params=self.neuron_noise_dict[node]['generator_pars'])
                nest.Connect([self.noise_nodes[-1]], [NodeToNest[node]], "all_to_all",
                    syn_spec=self.neuron_noise_dict[node]['connection_pars'])

    def ConstructSignalElements(self):
        """
        Creates spike generators and connects them to the parrots

        assumes parrots created in network order and thereby line
        up with input list of spike-times
        """

        for i, parrot in enumerate(self.parrots):
            spike_generator = nest.Create("spike_generator", 1, {'allow_offgrid_spikes':True, 
                'spike_times': self.input_spike_trains[i]})
            nest.Connect(spike_generator, [parrot], "all_to_all", \
                syn_spec={"model": "static_synapse", "weight":1.0, "delay":0.1})

def avalanche_information(spike_train_data, sim_time, num_trials, window=2.0, use_communities=False):
    """
    Takes spike result data from NNS and calculates the avalanche distribution
    Three things are calculated for an avalanche: the number of activations that
    occur over the whole duration of an avalanche, the duration of the
    avalanche, and the inter-event interval (IEI).

    Avalanches are determined by a consecutive series of neuron activations
    that occur within a window of time (in ms).

    If communities are active, then avalanches are calculated independently as 
    well for each community, irrespective of the other.
    """

    if use_communities == False:
        #dictNodes_listTrials_arraySpiketimes
        bins = np.arange(0.0, sim_time + window, window)
        dictAvalancheData = {}
        dictAvalancheData['bins'] = bins
        combined_avalanche_sizes = []
        combined_avalanche_duration = []
        combined_avalanche_IEI = []
        combined_branching_values = []
        for trial in xrange(num_trials):
            binned_spike_frequency = bin_trial_spike_frequency(trial, spike_train_data, bins)
            avalanche_sizes, avalanche_duration, avalanche_IEI = find_avalanches(binned_spike_frequency)
            combined_avalanche_sizes.append(avalanche_sizes)
            combined_avalanche_duration.append(avalanche_duration)
            combined_avalanche_IEI.append(avalanche_IEI)
            combined_branching_values.append(branching_values(binned_spike_frequency))

        dictAvalancheData['branching_parameter'] = np.mean(np.concatenate(combined_branching_values))
        dictAvalancheData['avalanche_duration'] = np.concatenate(combined_avalanche_duration) * window
        dictAvalancheData['avalanche_sizes'] = np.concatenate(combined_avalanche_sizes)
        dictAvalancheData['avalanche_IEI'] = np.concatenate(combined_avalanche_IEI) * window
    
    if use_communities:
        bins = np.arange(0.0, sim_time + window, window)
        dictAvalancheData = {}
        dictAvalancheData['bins'] = bins
        dictAvalancheData['community_data'] = {}
        #dictCom_dictNodes_listTrials_arraySpiketimes
        for com in spike_train_data.keys():
            dictAvalancheData['community_data'][com] = {}
            combined_avalanche_sizes = []
            combined_avalanche_duration = []
            combined_avalanche_IEI = []
            combined_branching_values = []
            for trial in xrange(num_trials):
                binned_spike_frequency = bin_trial_spike_frequency(trial, spike_train_data[com], bins)
                avalanche_sizes, avalanche_duration, avalanche_IEI = find_avalanches(binned_spike_frequency)
                combined_avalanche_sizes.append(avalanche_sizes)
                combined_avalanche_duration.append(avalanche_duration)
                combined_avalanche_IEI.append(avalanche_IEI)
                combined_branching_values.append(branching_values(binned_spike_frequency))

            dictAvalancheData['community_data'][com]['branching_parameter'] = np.mean(np.concatenate(combined_branching_values))
            dictAvalancheData['community_data'][com]['avalanche_duration'] = np.concatenate(combined_avalanche_duration) * window
            dictAvalancheData['community_data'][com]['avalanche_sizes'] = np.concatenate(combined_avalanche_sizes)
            dictAvalancheData['community_data'][com]['avalanche_IEI'] = np.concatenate(combined_avalanche_IEI) * window

    return dictAvalancheData

def find_avalanches(binned_spike_frequency):
    """
    determines whether there are avalanches forming. Can not deal with multiple
    simultaneous avalanches in the network

    avalanche is defined by a consecutive sequence of bins with non-zero spike
    counts.
    """

    avalanche_sizes = []
    avalanche_IEI = []
    avalanche_duration = []
    is_avalanche = False
    av_counter = 0
    duration_counter = 0
    IEI_counter = 0
    for count in binned_spike_frequency:
        if not is_avalanche and count == 0:
            IEI_counter += 1
        elif not is_avalanche and count > 0:
            is_avalanche = True
            av_counter += count
            duration_counter += 1
            avalanche_IEI.append(IEI_counter)
            IEI_counter = 0
        elif is_avalanche and count > 0:
            av_counter += count 
            duration_counter += 1
        elif is_avalanche and count == 0:
            is_avalanche = False
            avalanche_sizes.append(av_counter)
            av_counter = 0
            avalanche_duration.append(duration_counter)
            duration_counter = 0
            IEI_counter += 1

    return avalanche_sizes, avalanche_duration, avalanche_IEI

def bin_trial_spike_frequency(trial, dictNodes_listTrials_arraySpikeTimes, bins):
    """
    """

    freq_time_series = np.zeros(len(bins) - 1)
    for node in dictNodes_listTrials_arraySpikeTimes.keys():
        freq_time_series += np.histogram(dictNodes_listTrials_arraySpikeTimes[node][trial], bins)[0]
    
    return freq_time_series

def branching_values(binned_spike_frequency):

    branching_values = binned_spike_frequency[1:] / (binned_spike_frequency[:-1] * 1.0)
    return branching_values[np.isfinite(branching_values)]

def calc_branching_parameter(binned_spike_frequency):
    """
    branching parameter is considered as the expected value of the number of
    descendants of a process divided by the number of ascedants.

    this can be calculated by taking the average of f_t+1 / f_t for all t.
    Excepting inactive bins, which are not allowed to be ascedants.
    """

    branching_values = binned_spike_frequency[1:] / (binned_spike_frequency[:-1] * 1.0)
    return np.mean(branching_values[np.isfinite(branching_values)])

def PlotPSTH(self, prefix, array_spike_times, sim_time, dt=5.0, window=None, start_time=0.0):
    """
    """

    bins = np.arange(start_time, sim_time + dt, dt)

    if len(array_spike_times.shape) == 1:
        arrayPSTH = np.histogram(array_spike_times, bins=bins)[0] / dt * 1000.0
    elif len(array_spike_times.shape) == 2:
        arrayPSTH = np.mean([ np.histogram(trial, bins=bins)[0] \
            for trial in array_spike_times ], axis=0) / dt * 1000.0

    plt.clf()
    plt.bar(bins[:-1], arrayPSTH, bins[1:] - bins[:-1])
    if window != None:
        plt.xlim(window[0], window[1])
    plt.xlabel('time (ms)')
    plt.ylabel(r'$\rho$')
    plt.savefig(prefix + "_psth.png", dpi=300)
    plt.clf()
    plt.close()

if __name__ == '__main__':
    pass
    # # Couple neuron testing
    # network = nx.from_numpy_matrix(np.matrix([[0,1.0],[0,0]]))
    # nns = NeuralNetSim(network, neuron_type='iaf_neuron', 
    #     external_noise_generators=['noise_generator', 'poisson_generator'],
    #     external_noise_parameters=[{'std':10.0, 'dt':0.1, 'mean':0.0}, {'rate':200.0}], 
    #     external_noise_connection_pars=[{'weight':1.0, 'delay': 0.1, 'model':"static_synapse"}, 
    #     {'weight':400.0, 'delay': 0.1, 'model':"static_synapse"}], sim_time=5000.0,
    #     volt_detectors=True, signal_nodes=[0,1], signal_generators=['spike_generator'],
    #     signal_generator_pars=[{'spike_times':[1.0,50.0], 'spike_weights':[2000.0, 2000.0]}],
    #     signal_connection_pars=[{'model':'static_synapse', 'weight':1.0, 'delay':0.1}],
    #     synapse_type='tsodyks_synapse',
    #     EE_synapse_parameters={"U":0.67, "u":0.67, 'x':1.0, "tau_rec":450.0, "tau_fac":0.0, "weight":1000.}, num_trials=3)
    # nns.Engage()
    # nns.PlotVoltageTrace('test')
    # nns.PlotSpiketrain('test')
    # nns.PlotPSTH('test', dt=50.0)
    # nns.PlotISIdistribution('test')
    # nns.PlotNetworkActivity('test', dt=20)
    # nns.SaveDataToFile('test')

    # # Community testing/ many neuron testing
    # import two_community_block_model
    # network = two_community_block_model.generate_weighted_two_community_graph(50, 0.1, 7, EE_W=1000.0)
    # nns = NeuralNetSim(network, sim_time=100.0, neuron_type='iaf_psc_exp', volt_detectors=True, 
    #     signal_connection_ratio=0.16,
    #     signal_generators=['spike_generator'],
    #     signal_generator_pars=[{'spike_times':[1.0], 'spike_weights':[30000.0]}],
    #     signal_connection_pars=[{'model':'static_synapse', 'weight':1.0, 'delay':0.1}],
    #     synapse_type='tsodyks_synapse',
    #     EE_synapse_parameters={"U":0.67, "u":0.67, 'x':1.0, "tau_rec":450.0, "tau_fac":0.0, "weight":40000.},
    #     num_trials=3, use_communities=True, signal_community=1)
    # nns.Engage()
    # # nns.PlotVoltageTrace('test')
    # nns.PlotSpiketrain('test')
    # # nns.PlotPSTH('test', dt=10.0)
    # # nns.PlotISIdistribution('test')
    # nns.PlotNetworkActivity('test', dt=10)

    # # Neuron fitting class test
    # network = utilities.load_object('13010800_network_full.pyobj')
    # subgraph_nodes = [1]
    # subgraph_nodes += network.predecessors(1)
    # select_subgraph = network.subgraph(subgraph_nodes)
    # to_remove = []
    # for edge in select_subgraph.edges_iter():
    #     if edge[0] == 1:
    #         to_remove.append((edge[0], edge[1]))
    #     elif edge[1] != 1:
    #         to_remove.append((edge[0], edge[1]))
    # select_subgraph.remove_edges_from(to_remove)

    # synapse_parameter_dict = { edge[0] : { edge[1] : {'model':'tsodyks2_synapse', 
    #     'U':0.5, 'u':0.5, 'x': 1.0, 'tau_rec': 600.0, 'tau_fac': 35.} }\
    #     for edge in select_subgraph.edges_iter() }
    # neuron_noise_dict = {'generator': 'noise_generator', 
    #     'generator_pars': {'mean': 200.0, 'std': 250.0, 'dt': 1.0}, 
    #     'connection_pars': {'delay':0.1, 'weight': 1.0}}
    # neuron_parameter_dict = { 1 : {'C_m': 281.0, 't_ref': 0.0, 'V_reset': -60.0,
    #     'a': 4.0, 'b': 80.5, 'Delta_T': 2.0, 'tau_w': 144.0, 'V_th': -50.4, 'V_peak': 0.0,
    #     'E_ex': 0.0, 'E_in': -85.0, 'tau_syn_ex': 0.2, 'tau_syn_in': 2.0}}
    # input_nodes = select_subgraph.nodes()
    # input_nodes.remove(1)
    # neuron_spike_trains = [ np.sort(np.random.uniform(0.0, 1000.0, 50)) for node in input_nodes ]
    # nfs = NeuronFittingSim(select_subgraph, input_nodes, neuron_spike_trains, 
    #     sim_time=1000.0, neuron_type='aeif_cond_exp', volt_detectors=False, seed=1,
    #     use_communities=False, dt=0.1, weight_scale=1.0, num_trials=1,
    #     neuron_noise_dict=neuron_noise_dict, synapse_parameter_dict=synapse_parameter_dict, 
    #     neuron_parameter_dict=neuron_parameter_dict)
    # nfs.Engage()
    # nfs.PlotSpiketrain('test')

    # nodes = input_nodes
    # plt.clf()
    # # fig, axes = plt.subplots(figsize=(5,5), dpi=300)
    # # Assumes node IDs are numbers... which could be dangerous but is usually the case
    # for node, spike_times in enumerate(neuron_spike_trains):
    #     try:
    #         points_to_plot = spike_times
    #     except IndexError:
    #         continue

    #     plt.plot(points_to_plot, input_nodes[node] * np.ones(len(points_to_plot)),
    #         linestyle='None', marker='o', 
    #         color='black', markersize=2.0, markeredgewidth=0.0)

    # plt.ylim(min(nodes)-0.25, max(nodes)+0.25)
    # plt.xlabel('time (ms)')
    # plt.savefig('datatest_spike_train.png', dpi=300)
    # plt.close()
    # plt.clf()