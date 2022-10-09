import argparse
import os


parser = argparse.ArgumentParser(description='Multi-Hop Knowledge Graph Reasoning with Reward Shaping')

# Experiment control
parser.add_argument('--process_data', action='store_true',
                    help='process knowledge graph (default: False)')
parser.add_argument('--train', action='store_true',
                    help='run path selection set_policy training (default: False)')
parser.add_argument('--inference', action='store_true',
                    help='run knowledge graph inference (default: False)')
parser.add_argument('--search_random_seed', action='store_true',
                    help='run experiments with multiple random initializations and compute the result statistics '
                         '(default: False)')
parser.add_argument('--eval', action='store_true',
                    help='compute evaluation metrics (default: False)')
parser.add_argument('--eval_by_relation_type', action='store_true',
                    help='compute evaluation metrics for to-M and to-1 relations separately (default: False)')
parser.add_argument('--eval_by_seen_queries', action='store_true',
                    help='compute evaluation metrics for seen queries and unseen queries separately (default: False)')
parser.add_argument('--run_ablation_studies', action='store_true',
                    help='run ablation studies')
parser.add_argument('--run_analysis', action='store_true',
                    help='run algorithm analysis and print intermediate results (default: False)')
parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'),
                    help='directory where the knowledge graph data is stored (default: None)')
parser.add_argument('--model_root_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model'),
                    help='root directory where the model parameters are stored (default: None)')
parser.add_argument('--model_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model'),
                    help='directory where the model parameters are stored (default: None)')
parser.add_argument('--gpu', type=int, default=0,
                    help='gpu device (default: 0)')
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='path to a pretrained checkpoint')
parser.add_argument('--seed', type=int, default=543, metavar='S',
                    help='random seed (default: 543)')

# Data
parser.add_argument('--test', action='store_true',
                    help='perform inference on the test set (default: False)')
parser.add_argument('--group_examples_by_query', action='store_true',
                    help='group examples by topic entity + query relation (default: False)')

# Network Architecture
parser.add_argument('--model', type=str, default='point',
                    help='knowledge graph QA model (default: point)')
parser.add_argument('--entity_dim', type=int, default=200, metavar='E',
                    help='entity embedding dimension (default: 200)')
parser.add_argument('--relation_dim', type=int, default=200, metavar='R',
                    help='relation embedding dimension (default: 200)')
parser.add_argument('--use_action_space_bucketing', action='store_true',
                    help='bucket adjacency list by outgoing degree to avoid memory blow-up (default: False)')
parser.add_argument('--bucket_interval', type=int, default=10,
                    help='adjacency list bucket size (default: 32)')
parser.add_argument('--type_only', action='store_true',
                    help='use denote knowledge graph node by entity types only (default: False)')
parser.add_argument('--relation_only', action='store_true',
                    help='search with relation information only, ignoring entity representation (default: False)')
parser.add_argument('--relation_only_in_path', action='store_true',
                    help='include intermediate entities in path (default: False)')

# Knowledge Graph
parser.add_argument('--num_graph_convolution_layers', type=int, default=0,
                    help='number of graph convolution layers to use (default: 0, no GC is used)')
parser.add_argument('--graph_convolution_rank', type=int, default=10,
                    help='number of ranks ')
parser.add_argument('--add_reverse_relations', type=bool, default=True,
                    help='add reverse relations to KB (default: True)')
parser.add_argument('--add_reversed_training_edges', action='store_true',
                    help='add reversed edges to extend training set (default: False)')
parser.add_argument('--train_entire_graph', type=bool, default=False,
                    help='add all edges in the graph to extend training set (default: False)')
parser.add_argument('--emb_dropout_rate', type=float, default=0.3,
                    help='Knowledge graph embedding dropout rate (default: 0.3)')
parser.add_argument('--zero_entity_initialization', type=bool, default=False,
                    help='Initialize all entities to zero (default: False)')
parser.add_argument('--uniform_entity_initialization', type=bool, default=False,
                    help='Initialize all entities with the same random embedding (default: False)')

# Optimization
parser.add_argument('--num_epochs', type=int, default=200,
                    help='maximum number of pass over the entire training set (default: 20)')
parser.add_argument('--num_wait_epochs', type=int, default=5,
                    help='number of epochs to wait before stopping training if dev set performance drops')
parser.add_argument('--num_peek_epochs', type=int, default=2,
                    help='number of epochs to wait for next dev set result check (default: 2)')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='epoch from which the training should start (default: 0)')
parser.add_argument('--batch_size', type=int, default=256,
                    help='mini-batch size (default: 256)')
parser.add_argument('--train_batch_size', type=int, default=256,
                    help='mini-batch size during training (default: 256)')
parser.add_argument('--dev_batch_size', type=int, default=64,
                    help='mini-batch size during inferece (default: 64)')
parser.add_argument('--margin', type=float, default=0,
                    help='margin used for base MAMES training (default: 0)')
parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--learning_rate_decay', type=float, default=1.0,
                    help='learning rate decay factor for the Adam optimizer (default: 1)')
parser.add_argument('--adam_beta1', type=float, default=0.9,
                    help='Adam: decay rates for the first movement estimate (default: 0.9)')
parser.add_argument('--adam_beta2', type=float, default=0.999,
                    help='Adam: decay rates for the second raw movement estimate (default: 0.999)')
parser.add_argument('--grad_norm', type=float, default=10000,
                    help='norm threshold for gradient clipping (default 10000)')
parser.add_argument('--xavier_initialization', type=bool, default=True,
                    help='Initialize all model parameters using xavier initialization (default: True)')
parser.add_argument('--random_parameters', type=bool, default=False,
                    help='Inference with random parameters (default: False)')

# Fact Network
parser.add_argument('--label_smoothing_epsilon', type=float, default=0.1,
                    help='epsilon used for label smoothing')
parser.add_argument('--hidden_dropout_rate', type=float, default=0.3,
                    help='ConvE hidden layer dropout rate (default: 0.3)')
parser.add_argument('--feat_dropout_rate', type=float, default=0.2,
                    help='ConvE feature dropout rate (default: 0.2)')
parser.add_argument('--emb_2D_d1', type=int, default=10,
                    help='ConvE embedding 2D shape dimension 1 (default: 10)')
parser.add_argument('--emb_2D_d2', type=int, default=20,
                    help='ConvE embedding 2D shape dimension 2 (default: 20)')
parser.add_argument('--num_out_channels', type=int, default=32,
                    help='ConvE number of output channels of the convolution layer (default: 32)')
parser.add_argument('--kernel_size', type=int, default=3,
                    help='ConvE kernel size (default: 3)')
parser.add_argument('--distmult_state_dict_path', type=str, default='',
                    help='Path to the DistMult network state_dict (default: '')')
parser.add_argument('--complex_state_dict_path', type=str, default='',
                    help='Path to the ComplEx network state dict (default: '')')
parser.add_argument('--conve_state_dict_path', type=str, default='',
                    help='Path to the ConvE network state dict (default: '')')
parser.add_argument('--num_emb_hidden_layers', type=int, default=1,
                    help='number of embedding hidden layers')
parser.add_argument('--emb_hidden_size', type=int, default=250,
                    help='embedding hidden size')
parser.add_argument('--num_hidden_layers', type=int, default=1,
                    help='number of hidden layers')
parser.add_argument('--hidden_size', type=int, default=1024,
                    help='hidden size')
parser.add_argument('--use_tau', default=False, action='store_true',
                    help='whether use parameter lambda and tau')
parser.add_argument('--tau', type=float, default=0.9,
                    help='temperature parameter')
parser.add_argument('--bandwidth', type=int, default=300,
                    help='maximum number of outgoing edges to explore at each step (default: 300)')
parser.add_argument('--num_negative_samples', type=int, default=100,
                    help='Number of negative samples to use for embedding-based methods')
parser.add_argument('--beam_size', type=int, default=128,
                    help='size of beam used in beam search inference (default: 100)')
parser.add_argument('--save_beam_search_paths', action='store_true', default=False,
                    help='save the decoded path into a CSV file (default: False)')

# Graph Completion
parser.add_argument('--theta', type=float, default=0.2,
                    help='Threshold for sifting high-confidence facts (default: 0.2)')

# Separate Experiments
parser.add_argument('--export_to_embedding_projector', action='store_true',
                    help='export model embeddings to the Tensorflow Embedding Projector format (default: False)')
parser.add_argument('--export_reward_shaping_parameters', action='store_true',
                    help='export KG embeddings and fact network parameters for reward shaping models (default: False)')
parser.add_argument('--compute_fact_scores', action='store_true',
                    help='[Debugging Option] compute embedding based model scores (default: False)')
parser.add_argument('--export_fuzzy_facts', action='store_true',
                    help='export the facts recovered by embedding based method (default: False)')
parser.add_argument('--export_error_cases', action='store_true',
                    help='export the error cases of a model')
parser.add_argument('--compute_map', action='store_true',
                    help='compute the Mean Average Precision evaluation metrics (default: False)')

# Hyperparameter Search
parser.add_argument('--tune', type=str, default='',
                    help='Specify the hyperparameters to tune during the search, separated by commas (default: None)')
parser.add_argument('--grid_search', action='store_true',
                    help='Conduct grid search of hyperparameters')

args = parser.parse_args()
