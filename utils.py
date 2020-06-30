import pandas as pd
from openpyxl import load_workbook
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import sys 

class GraphUtils():
    ''' provide a set of utility fucntions to help visualize and load networkx graphs.
    The class follows a singliton design pattern
    '''
    
    @staticmethod
    def load_graph(file_path):
        '''
        load a file containing the structure of a graph as an excel sheet into a pandas dataframe.

        Args:
            file_path: String
        '''
        assert file_path.endswith('.xlsx')
        
        network_structure = None
        try:
            network_structure = pd.read_excel(file_path, sheet_name='network_structure')
            network_structure.fillna('', inplace=True)
            GraphUtils._validate_data_frame(network_structure)
        except OSError as err:
            print("OS error: {0}".format(err))
            raise
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
        return network_structure
    
    @staticmethod
    def _validate_data_frame(df):
        '''validate the data frame has the expected columns.
        Args:
            df: pandas data frame
        Throws:
            AssertionError if the the column names do not match the expected list (type, parent_node, node)
        '''
        assert sum(df.columns==['type', 'parent_node', 'node'])==3


    @staticmethod
    def visualise_network(network_structure):
        ''' a wrapper to run networkx visualisation tool
        Args:
            network_structure: a pandas data frame of the network structure. The columns expected are:
                - type: the name of the structure
                - parent_node: a comma seperated list of the parent nodes
                - node: the name of the node 
        '''
        GraphUtils._validate_data_frame(network_structure)

        graphs = GraphUtils._get_nx_graphs(network_structure)
        types = list(network_structure['type'].unique()) 
        for graph,unique_type in zip(graphs,types):
            nx.draw(graph, with_labels=True)
            plt.title("Bayesian network structure for nodes of type " + unique_type)
            plt.show()
            print("\n")
    

    @staticmethod
    def _get_nx_graphs(network_structure):
        """
        transform the graphs into a networksx format to faciliate other operations
        """
        graphs = []
        for unique_type in list(network_structure['type'].unique()):
            unique_nodes = \
                list(network_structure[network_structure['type'] == \
                                            unique_type]['node'].unique())

            graph = nx.DiGraph()
            graph.add_nodes_from(unique_nodes)

            for node in unique_nodes:
                parents = \
                    network_structure[(network_structure['type'] == unique_type) & \
                                           (network_structure['node'] == node)]['parent_node'].iloc[0].split(",")

                if parents == ['']:
                    pass
                elif len(parents) == 1:
                    graph.add_edge(parents[0], node)
                else:
                    for parent in parents:
                        graph.add_edge(parent.strip(), node)
            
            graphs.append(graph)
        
        return graphs


class ExpUtils:
    @staticmethod
    def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
        '''
        Prepare an input function to be used in training TF models
        Args:
            - data_df: a data frame of the data
            - label_df: a data frame containing the numerical labels
            - num_epochs: number of epochs
            - shuffle: True if the data/labels are to be shuffled
            - batch_size: the size of the batch to be used during training 
        Returns:
            Function
        '''
        def input_function():
            if label_df is None:
                #print(data_df.shape)
                ds = tf.data.Dataset.from_tensor_slices(dict(data_df))
            else:
                ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
            if shuffle:
                ds = ds.shuffle(1000)
            ds = ds.batch(batch_size).repeat(num_epochs)
            return ds
        return input_function

    @staticmethod
    def get_class_probs(input_fn,est):
        pred_dicts = list(est.predict(input_fn))
        probs = np.zeros((len(pred_dicts),pred_dicts[0]['probabilities'].shape[0]))
        for i,pred in enumerate(pred_dicts):
            probs[i,:] = pred['probabilities']
        return probs

    @staticmethod
    def conditional_prob(input_fn,est):
    #calculate the conditional probability of the output of the estimator (est) given the input data passed by (input_fn)
        pred_dicts = list(est.predict(input_fn))
        probs = np.array([pred['probabilities'] for pred in pred_dicts])
        return probs



#print(GraphUtils.load_graph(r"toy_example_structure.xlsx"))

