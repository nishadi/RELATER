# RELATER - constRaints and attributE values, reLationships, Ambiguity, and
#           refinemenT for Entity Resolution
# An unsupervised graph-based entity resolution framework that is focused on
# resolving the challenges associated with resolving complex entities. We
# propose a global method to propagate merging decisions by propagating
# attribute values and constraints to capture dynamically changing attribute
# values and different relationships, a method for leveraging ambiguity in
# the ER process, an adaptive method of incorporating relationship structure,
# and a dynamic refinement step to improve entity clusters by unmerging
# likely wrong links. RELATER can be employed to resolve records of both
# basic and complex entities.
#
#
# Nishadi Kirielle, Peter Christen, and Thilina Ranbaduge
#
# Contact: nishadi.kirielle@anu.edu.au
#
# School of Computing, The Australian National University, Canberra, ACT, 2600
# -----------------------------------------------------------------------------
#
# Copyright 2021 Australian National University and others.
# All Rights reserved.
#
# -----------------------------------------------------------------------------
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
#
# =============================================================================
import logging
import os
import pickle
import time
from collections import defaultdict

from common import ambiguity, constants as c, sim, settings, util, \
  attributes_meta, stats, hyperparams
from data import bib_data_loader
from er.bib_graph import TYPE_META, BIB_GRAPH


# ---------------------------------------------------------------------------
# PRE-REQUISITES and GRAPH GENERATION
# ---------------------------------------------------------------------------

def append_frequencies(records_input_file, role_type, attribute_index_list,
                       attribute_name, output_file):
  record_dict = pickle.load(open(records_input_file, 'rb'))
  ambiguity.cluster_attribute_combinations(record_dict.values(), role_type,
                                           attribute_index_list,
                                           attribute_name)
  record_list_wf = ambiguity.append_clustered_attr_f(record_dict.values(),
                                                     role_type,
                                                     attribute_index_list,
                                                     attribute_name)
  pickle.dump(record_list_wf, open(output_file, 'wb'))


def generate_atomic_nodes(data_sets, file_name, sim_func_dict):
  atomic_node_dict = {}

  pair_sim = sim.get_pair_sim_no_cache

  record_list_wf1 = pickle.load(open(file_name.format(data_sets[0]), 'rb'))
  record_list_wf2 = pickle.load(open(file_name.format(data_sets[1]), 'rb'))

  for i_attribute, sim_function in sim_func_dict.iteritems():
    atomic_node_dict[i_attribute] = {}

    value_list1 = list({p[i_attribute] for p in record_list_wf1})
    assert '' not in value_list1
    if None in value_list1:
      value_list1.remove(None)
    value_list1.sort()

    value_list2 = list({p[i_attribute] for p in record_list_wf2})
    assert '' not in value_list2
    if None in value_list2:
      value_list2.remove(None)
    value_list2.sort()

    logging.info('1 ---- {} : {}'.format(i_attribute, len(value_list1)))
    logging.info('2 ---- {} : {}'.format(i_attribute, len(value_list2)))

    for i in xrange(len(value_list1)):
      for j in xrange(len(value_list2)):
        key = (value_list1[i], value_list2[j])
        if key in atomic_node_dict[i_attribute] or \
            (value_list2[j], value_list1[i]) in atomic_node_dict[i_attribute]:
          continue

        sim_val = pair_sim(key, sim_function)

        if sim_val >= 0.8:
          atomic_node_dict[i_attribute][key] = sim_val

  pickle.dump(atomic_node_dict, open(settings.atomic_node_file, 'wb'))
  logging.info('Successfully generated atomic nodes')


def merge_datasets(filename1, dataset1_abbreviation, filename2,
                   dataset2_abbreviation, ref_id_index, output_filename):
  record_list1_wf = pickle.load(open(filename1, 'rb'))
  record_list2_wf = pickle.load(open(filename2, 'rb'))

  merged_record_dict = dict()
  for record in record_list1_wf:
    record[0] = '%s-%s' % (dataset1_abbreviation, record[0])
    record[ref_id_index] = map(lambda i: '%s-%s' % (dataset1_abbreviation, i),
                               record[ref_id_index])
    merged_record_dict[record[0]] = record
  for record in record_list2_wf:
    record[0] = '%s-%s' % (dataset2_abbreviation, record[0])
    record[ref_id_index] = map(lambda i: '%s-%s' % (dataset2_abbreviation, i),
                               record[ref_id_index])
    merged_record_dict[record[0]] = record
  pickle.dump(merged_record_dict, open(output_filename, 'wb'))


def pre_requisites(data_sets, dataset1_abbreviation, dataset2_abbreviation):
  """
  Method to run pre-requisites for graph generation

  :param data_sets: Data sets
  :param dataset1_abbreviation: Data set 1 abbreviation
  :param dataset2_abbreviation: Data set 2 abbreviation
  :return:
  """
  # Temporary file locations
  tmp_dir = settings.output_home_directory + 'tmp/'
  if not os.path.isdir(tmp_dir):
    os.makedirs(tmp_dir)
  author_file = tmp_dir + 'authors-{}.gpickle'
  pubs_file = tmp_dir + 'publications-{}.gpickle'
  authors_wf_file = tmp_dir + '/authors-{}-A.gpickle'
  pub_wf_file = tmp_dir + '/publications-{}-P.gpickle'

  # Generate record dictionaries
  bib_data_loader.enumerate_records(author_file, pubs_file)

  for ds in data_sets:
    # Authors: generating clusters and appending frequencies
    append_frequencies(author_file.format(ds), 'A',
                       [c.BB_I_FNAME, c.BB_I_SNAME, c.BB_I_MNAME],
                       'fname-sname-mname', authors_wf_file.format(ds))

    # Publications generating clusters and appending frequencies
    append_frequencies(pubs_file.format(ds), 'P', [c.BB_I_PUBNAME],
                       'pubname', pub_wf_file.format(ds))

  generate_atomic_nodes(data_sets, authors_wf_file,
                        attributes_meta.authors_sim_func)

  # Merge authors in two data sets
  filename1 = authors_wf_file.format(data_sets[0])
  filename2 = authors_wf_file.format(data_sets[1])
  merge_datasets(filename1, dataset1_abbreviation, filename2,
                 dataset2_abbreviation, c.BB_I_COAUTHOR_LIST,
                 settings.authors_file)

  # Merge publications in two data sets
  filename1 = pub_wf_file.format(data_sets[0])
  filename2 = pub_wf_file.format(data_sets[1])
  merge_datasets(filename1, dataset1_abbreviation, filename2,
                 dataset2_abbreviation, c.BB_I_AUTHOR_ID_LIST,
                 settings.pub_file)


def generate_graph(atomic_t):
  """
  Method to generate graph.

  `pre_requisites()` should be completed before running this.
  :param atomic_t: Atmoic node threshold
  :return:
  """
  # Record types
  au_attr = attributes_meta.def_authors(atomic_t)
  pub_attr = attributes_meta.def_pubs(atomic_t)
  type_meta_dict = {
    'A': TYPE_META('A', au_attr, settings.authors_file),
    'P': TYPE_META('P', pub_attr, settings.pub_file)
  }

  graph = BIB_GRAPH(type_meta_dict, dataset1_abbreviation,
                    dataset2_abbreviation)

  atomic_node_dict = pickle.load(open(settings.atomic_node_file, 'rb'))
  graph.generate_graph(atomic_node_dict)

  graph.serialize_graph(settings.graph_file)

  return graph


# ---------------------------------------------------------------------------
# Ground truth analysis
# ---------------------------------------------------------------------------

def gt_analyse_initial_graph(graph, link):
  """
  Analyse the ground truth precision and recall values for the initial graph.

  :param graph: Graph
  :param link: Link
  :return:
  """
  gt_links = graph.gt_links_dict

  # Get the links existing in the graph
  graph_links = set()
  for node_id, node in graph.G.nodes(data=True):
    if node[c.TYPE1] == c.N_RELTV and node[c.TYPE2] == link:
      p1 = graph.type_meta_dict['P'].record_dict[node[c.R1]]
      p2 = graph.type_meta_dict['P'].record_dict[node[c.R2]]

      key = (p1[c.I_ID], p2[c.I_ID])
      graph_links.add(key)

  # Count tp, fp, and fn
  true_positive_dict = defaultdict(int)
  false_positive_dict = defaultdict(int)
  false_negative_dict = defaultdict(int)

  for processed_link in graph_links:
    if processed_link in gt_links[link]:
      true_positive_dict[link] += 1
    else:
      false_positive_dict[link] += 1

  for gt_link in gt_links[link]:
    if gt_link not in graph_links:
      false_negative_dict[link] += 1

  # Dictionary to hold the precision and recall of each link type
  precision_dict = {}
  recall_dict = {}

  # Precision calculation
  precision_divisor = \
    float(true_positive_dict[link] + false_positive_dict[link])
  if precision_divisor == 0:
    precision_dict[link] = 0
  else:
    precision_dict[link] = round(true_positive_dict[link] * 100.0 / \
                                 precision_divisor, 2)

  # Recall calculation
  recall_divisor = \
    float(true_positive_dict[link] + false_negative_dict[link])
  if recall_divisor == 0:
    recall_dict[link] = 0
  else:
    recall_dict[link] = round(true_positive_dict[link] * 100.0 / \
                              recall_divisor, 2)

  stats_dict = {
    'pre': precision_dict,
    're': recall_dict,
    'fp': false_positive_dict,
    'fn': false_negative_dict,
    'tp': true_positive_dict
  }
  stats.persist_gt_analysis(gt_links, defaultdict(list), stats_dict,
                            '_initial-graph', ['P-P'], -1)


# ---------------------------------------------------------------------------
# Link
# ---------------------------------------------------------------------------

def link(dataset1_abbreviation, dataset2_abbreviation, atomic_t, merge_t):

  graph = get_graph(dataset1_abbreviation, dataset2_abbreviation, au_attr,
                    pub_attr)

  # gt_analyse_initial_graph(graph, 'P-P')

  start_time = time.time()
  graph.bootstrap([hyperparams.bootstrap_t])
  t1 = round(time.time() - start_time, 2)
  logging.info('Bootstrapping time {}'.format(t1))
  # graph.validate_ground_truth('rem-sin-bootstrapped-%s' % hyperparams.scenario,
  #                             t1)

  settings.enable_analysis = False
  start_time = time.time()
  graph.link(merge_t)
  t2 = round(time.time() - start_time, 2)
  logging.info('Linking time {}'.format(t2))
  # graph.validate_ground_truth('extrapubyear-extraauthoryear-exacty-withsin0.95%s' % hyperparams.scenario, t1 + t2)
  graph.validate_ground_truth('%s' % hyperparams.scenario, t1 + t2)


def get_graph(dataset1_abbreviation, dataset2_abbreviation, au_attr,
              pub_attr):
  type_meta_dict = {
    'A': TYPE_META('A', au_attr, settings.authors_file),
    'P': TYPE_META('P', pub_attr, settings.pub_file)
  }

  graph = BIB_GRAPH(type_meta_dict, dataset1_abbreviation,
                    dataset2_abbreviation)
  graph.deserialize_graph(settings.graph_file)
  return graph


if __name__ == '__main__':
  data_sets = ['A', 'B']
  dataset1_abbreviation = 'DA'
  dataset2_abbreviation = 'DB'

  log_file_name = util.get_logfilename(settings.output_home_directory + 'link')
  logging.basicConfig(filename=log_file_name, level=logging.INFO, filemode='w')

  # PRE-REQUISITES
  pre_requisites(data_sets, dataset1_abbreviation, dataset2_abbreviation)

  # GENERATE GRAPH
  graph = generate_graph(hyperparams.atomic_t)
  util.persist_graph_gen_times('def_authors/def_pubs', stats.a_time,
                               stats.r_time)

  # LINK
  if c.data_set == 'dblp-acm1':
    au_attr = attributes_meta.def_authors(hyperparams.atomic_t)
    pub_attr = attributes_meta.def_pubs(hyperparams.atomic_t)
  elif c.data_set == 'dblp-scholar1':
    au_attr = attributes_meta.def_authors_scholar(hyperparams.atomic_t)
    pub_attr = attributes_meta.def_pubs_scholar(hyperparams.atomic_t)
  else:
    raise Exception('Unsupported data set')

  link(dataset1_abbreviation, dataset2_abbreviation, hyperparams.atomic_t,
       hyperparams.merge_t)

