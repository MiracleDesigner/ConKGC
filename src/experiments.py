#!/usr/bin/env python3

import copy
import itertools
import numpy as np
import os, sys
import random
import torch

from src.parse_args import args

import src.data_utils as data_utils
import src.eval
from src.knowledge_graph import KnowledgeGraph
from src.emb.fact_network import ConKGC
from src.emb.emb import EmbeddingBasedMethod

torch.cuda.set_device(args.gpu)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def process_data():
    data_dir = args.data_dir
    raw_kb_path = os.path.join(data_dir, 'raw.kb')
    train_path = data_utils.get_train_path(args)
    dev_path = os.path.join(data_dir, 'dev.triples')
    test_path = os.path.join(data_dir, 'test.triples')
    data_utils.prepare_kb_envrioment(raw_kb_path, train_path, dev_path, test_path, args.test, args.add_reverse_relations)

def initialize_model_directory(args, random_seed=None):
    # add model parameter info to model directory
    model_root_dir = args.model_root_dir
    dataset = os.path.basename(os.path.normpath(args.data_dir))

    reverse_edge_tag = '-RV' if args.add_reversed_training_edges else ''
    entire_graph_tag = '-EG' if args.train_entire_graph else ''
    if args.xavier_initialization:
        initialization_tag = '-xavier'
    elif args.uniform_entity_initialization:
        initialization_tag = '-uniform'
    else:
        initialization_tag = ''

    # Hyperparameter signature
    if args.model == 'conkgc':
        hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            args.entity_dim,
            args.relation_dim,
            args.batch_size,
            args.dev_batch_size,
            args.learning_rate,
            args.num_emb_hidden_layers,
            args.emb_hidden_size,
            args.num_hidden_layers,
            args.hidden_size
        )
        if args.use_tau:
            hyperparam_sig += '-{}'.format(args.tau)
            # hyperparam_sig += '-{}-{}'.format(args.lamb, args.tau)
    else:
        raise NotImplementedError

    model_sub_dir = '{}-{}{}{}{}-{}'.format(
        dataset,
        args.model,
        reverse_edge_tag,
        entire_graph_tag,
        initialization_tag,
        hyperparam_sig
    )
    if args.model == 'set':
        model_sub_dir += '-{}'.format(args.beam_size)
        model_sub_dir += '-{}'.format(args.num_paths_per_entity)
    if args.relation_only:
        model_sub_dir += '-ro'
    elif args.relation_only_in_path:
        model_sub_dir += '-rpo'
    elif args.type_only:
        model_sub_dir += '-to'

    if args.test:
        model_sub_dir += '-test'

    model_dir = os.path.join(model_root_dir, model_sub_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print('Model directory created: {}'.format(model_dir))
    else:
        print('Model directory exists: {}'.format(model_dir))

    args.model_dir = model_dir

def construct_model(args):
    """
    Construct NN graph.
    """
    kg = KnowledgeGraph(args)
    if args.model == 'conkgc':
        fn = ConKGC(args)
        lf = EmbeddingBasedMethod(args, kg, fn)
    else:
        raise NotImplementedError
    return lf

def train(lf):
    train_path = data_utils.get_train_path(args)
    dev_path = os.path.join(args.data_dir, 'dev.triples')
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    train_data = data_utils.load_triples(
        train_path, entity_index_path, relation_index_path, group_examples_by_query=args.group_examples_by_query,
        add_reverse_relations=args.add_reversed_training_edges)
    if 'NELL' in args.data_dir:
        adj_list_path = os.path.join(args.data_dir, 'adj_list.pkl')
        seen_entities = data_utils.load_seen_entities(adj_list_path, entity_index_path)
    else:
        seen_entities = set()
    dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities)
    if args.checkpoint_path is not None:
        lf.load_checkpoint(args.checkpoint_path)
    lf.run_train(train_data, dev_data)

def inference(lf):
    lf.batch_size = args.dev_batch_size
    lf.eval()
    lf.load_checkpoint(get_checkpoint_path(args))
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    if 'NELL' in args.data_dir:
        adj_list_path = os.path.join(args.data_dir, 'adj_list.pkl')
        seen_entities = data_utils.load_seen_entities(adj_list_path, entity_index_path)
    else:
        seen_entities = set()

    eval_metrics = {
        'dev': {},
        'test': {}
    }

    if args.compute_map:
        relation_sets = [
            'concept:athletehomestadium',
            'concept:athleteplaysforteam',
            'concept:athleteplaysinleague',
            'concept:athleteplayssport',
            'concept:organizationheadquarteredincity',
            'concept:organizationhiredperson',
            'concept:personborninlocation',
            'concept:teamplayssport',
            'concept:worksfor'
        ]
        mps = []
        for r in relation_sets:
            print('* relation: {}'.format(r))
            test_path = os.path.join(args.data_dir, 'tasks', r, 'test.pairs')
            test_data, labels = data_utils.load_triples_with_label(
                test_path, r, entity_index_path, relation_index_path, seen_entities=seen_entities)
            pred_scores = lf.forward(test_data, verbose=False)
            mp = src.eval.link_MAP(test_data, pred_scores, labels, lf.kg.all_objects, verbose=True)
            mps.append(mp)
        map_ = np.mean(mps)
        print('Overall MAP = {}'.format(map_))
        eval_metrics['test']['avg_map'] = map
    elif args.eval_by_relation_type:
        dev_path = os.path.join(args.data_dir, 'dev.triples')
        dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities)
        pred_scores = lf.forward(dev_data, verbose=False)
        to_m_rels, to_1_rels, _ = data_utils.get_relations_by_type(args.data_dir, relation_index_path)
        relation_by_types = (to_m_rels, to_1_rels)
        print('Dev set evaluation by relation type (partial graph)')
        src.eval.hits_and_ranks_by_relation_type(
            dev_data, pred_scores, lf.kg.dev_objects, relation_by_types, verbose=True)
        print('Dev set evaluation by relation type (full graph)')
        src.eval.hits_and_ranks_by_relation_type(
            dev_data, pred_scores, lf.kg.all_objects, relation_by_types, verbose=True)
    elif args.eval_by_seen_queries:
        dev_path = os.path.join(args.data_dir, 'dev.triples')
        dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities)
        pred_scores = lf.forward(dev_data, verbose=False)
        seen_queries = data_utils.get_seen_queries(args.data_dir, entity_index_path, relation_index_path)
        print('Dev set evaluation by seen queries (partial graph)')
        src.eval.hits_and_ranks_by_seen_queries(
            dev_data, pred_scores, lf.kg.dev_objects, seen_queries, verbose=True)
        print('Dev set evaluation by seen queries (full graph)')
        src.eval.hits_and_ranks_by_seen_queries(
            dev_data, pred_scores, lf.kg.all_objects, seen_queries, verbose=True)
    else:
        dev_path = os.path.join(args.data_dir, 'dev.triples')
        test_path = os.path.join(args.data_dir, 'test.triples')
        dev_data = data_utils.load_triples(
            dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
        test_data = data_utils.load_triples(
            test_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
        print('Dev set performance:')
        pred_scores = lf.forward(dev_data, verbose=args.save_beam_search_paths)
        dev_metrics = src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.dev_objects, verbose=True)
        eval_metrics['dev'] = {}
        eval_metrics['dev']['hits_at_1'] = dev_metrics[0]
        eval_metrics['dev']['hits_at_3'] = dev_metrics[1]
        eval_metrics['dev']['hits_at_5'] = dev_metrics[2]
        eval_metrics['dev']['hits_at_10'] = dev_metrics[3]
        eval_metrics['dev']['mrr'] = dev_metrics[4]
        src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.all_objects, verbose=True)
        print('Test set performance:')
        pred_scores = lf.forward(test_data, verbose=False)
        test_metrics = src.eval.hits_and_ranks(test_data, pred_scores, lf.kg.all_objects, verbose=True)
        eval_metrics['test']['hits_at_1'] = test_metrics[0]
        eval_metrics['test']['hits_at_3'] = test_metrics[1]
        eval_metrics['test']['hits_at_5'] = test_metrics[2]
        eval_metrics['test']['hits_at_10'] = test_metrics[3]
        eval_metrics['test']['mrr'] = test_metrics[4]

    return eval_metrics

def export_to_embedding_projector(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.export_to_embedding_projector()

def export_reward_shaping_parameters(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.export_reward_shaping_parameters()

def export_fuzzy_facts(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.export_fuzzy_facts()

def export_error_cases(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.batch_size = args.dev_batch_size
    lf.eval()
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    dev_path = os.path.join(args.data_dir, 'dev.triples')
    dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path)
    lf.load_checkpoint(get_checkpoint_path(args))
    print('Dev set performance:')
    pred_scores = lf.forward(dev_data, verbose=False)
    src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.dev_objects, verbose=True)
    src.eval.export_error_cases(dev_data, pred_scores, lf.kg.dev_objects, os.path.join(lf.model_dir, 'error_cases.pkl'))

def get_checkpoint_path(args):
    if not args.checkpoint_path:
        return os.path.join(args.model_dir, 'model_best.tar')
    else:
        return args.checkpoint_path

def load_configs(config_path):
    with open(config_path) as f:
        print('loading configuration file {}'.format(config_path))
        for line in f:
            if not '=' in line:
                continue
            arg_name, arg_value = line.strip().split('=')
            if arg_value.startswith('"') and arg_value.endswith('"'):
                arg_value = arg_value[1:-1]
            if hasattr(args, arg_name):
                print('{} = {}'.format(arg_name, arg_value))
                arg_value2 = getattr(args, arg_name)
                if type(arg_value2) is str:
                    setattr(args, arg_name, arg_value)
                elif type(arg_value2) is bool:
                    if arg_value == 'True':
                        setattr(args, arg_name, True)
                    elif arg_value == 'False':
                        setattr(args, arg_name, False)
                    else:
                        raise ValueError('Unrecognized boolean value description: {}'.format(arg_value))
                elif type(arg_value2) is int:
                    setattr(args, arg_name, int(arg_value))
                elif type(arg_value2) is float:
                    setattr(args, arg_name, float(arg_value))
                else:
                    raise ValueError('Unrecognized attribute type: {}: {}'.format(arg_name, type(arg_value2)))
            else:
                raise ValueError('Unrecognized argument: {}'.format(arg_name))
    return args

def run_experiment(args):

    if args.test:
        if 'NELL' in args.data_dir:
            dataset = os.path.basename(args.data_dir)
            args.distmult_state_dict_path = data_utils.change_to_test_model_path(dataset, args.distmult_state_dict_path)
            args.complex_state_dict_path = data_utils.change_to_test_model_path(dataset, args.complex_state_dict_path)
            args.conve_state_dict_path = data_utils.change_to_test_model_path(dataset, args.conve_state_dict_path)
        args.data_dir += '.test'

    if args.process_data:

        # Process knowledge graph data

        process_data()
    else:
        with torch.set_grad_enabled(args.train):
            initialize_model_directory(args)
            lf = construct_model(args)
            lf.cuda()

            if args.train:
                train(lf)
            elif args.inference:
                inference(lf)
            elif args.export_to_embedding_projector:
                export_to_embedding_projector(lf)
            elif args.export_reward_shaping_parameters:
                export_reward_shaping_parameters(lf)
            elif args.export_fuzzy_facts:
                export_fuzzy_facts(lf)
            elif args.export_error_cases:
                export_error_cases(lf)

if __name__ == '__main__':
    run_experiment(args)









