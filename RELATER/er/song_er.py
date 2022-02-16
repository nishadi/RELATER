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
import pickle
import time
from collections import defaultdict

from common import settings, util, constants as c, hyperparams, ambiguity, sim, \
  attributes_meta, stats

from data import songs_data_loader

# ---------------------------------------------------------------------------
# PRE-REQUISITES and GRAPH GENERATION
# ---------------------------------------------------------------------------
from er.song_graph import SONG_GRAPH


def append_frequencies(records_input_file, role_type, attribute_index_list,
                       attribute_name, output_file):
  logging.info('Started appending frequencies!')
  record_dict = pickle.load(open(records_input_file, 'rb'))

  ambiguity.cluster_attribute_combinations(record_dict.values(), role_type,
                                           attribute_index_list,
                                           attribute_name)

  # For calculating frequencies for two different data sets,
  # we use the two data sets separately
  record_list1 = [r for rid, r in record_dict.iteritems() if
                  str(rid).startswith('10')]
  record_list2 = [r for rid, r in record_dict.iteritems() if
                  str(rid).startswith('20')]
  assert len(record_dict) == len(record_list1) + len(record_list2)

  record_list1_wf = ambiguity.append_clustered_attr_f(record_list1, role_type,
                                                      attribute_index_list,
                                                      attribute_name)
  record_list2_wf = ambiguity.append_clustered_attr_f(record_list2, role_type,
                                                      attribute_index_list,
                                                      attribute_name)

  record_list_wf = record_list1_wf + record_list2_wf
  pickle.dump(record_list_wf, open(output_file, 'wb'))
  logging.info('Finished appending frequencies!')


def generate_atomic_nodes(songs_wf_file, sim_func_dict):
  atomic_node_dict = {}

  pair_sim = sim.get_pair_sim_no_cache

  songs_list = pickle.load(open(songs_wf_file, 'rb'))

  record_list_wf1 = [r for r in songs_list if str(r[c.I_ID]).startswith('10')]
  record_list_wf2 = [r for r in songs_list if str(r[c.I_ID]).startswith('20')]

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


def pre_requisites():
  songs_dict_file = settings.output_home_directory + 'songs.gpickle'
  songs_wf_file = settings.songs_file

  # Generate record dictionaries
  songs_data_loader.enumerate_records(hyperparams.data_set, songs_dict_file)

  # Append frequencies
  append_frequencies(songs_dict_file, 'S',
                     [c.SG_I_ARTIST, c.SG_I_TITLE, c.SG_I_ALBUM],
                     'artist-title-album', songs_wf_file)

  # Generate atomic nodes
  generate_atomic_nodes(songs_wf_file, attributes_meta.songs_sim_func)


def generate_graph(atomic_t, output_directory):
  """
  Function to generate the graph.

  :param atomic_t: Atomic threshold
  :param output_directory: Output directory of the graph
  """
  logging.info('Atomic threshold {}'.format(atomic_t))

  atomic_node_dict = pickle.load(open(settings.atomic_node_file, 'rb'))
  graph = SONG_GRAPH(settings.attribute_init_function, atomic_t)
  graph.generate_graph(atomic_node_dict)
  graph.serialize_graph(
    '{}graph-{}.gpickle'.format(output_directory, str(hyperparams.atomic_t)))

  logging.info('Graph serialized successfully')

  return graph


# ---------------------------------------------------------------------------
# Link
# ---------------------------------------------------------------------------

def link(atomic_t, merge_t, sim_attr_init_func):
  """
  Function to link a civil graph

  :param atomic_t: Atomic threshold
  :param merge_t: Merging threshold
  :param sim_attr_init_func: Attribute similarity initialization function
  """
  graph = SONG_GRAPH(sim_attr_init_func, atomic_t)
  graph.deserialize_graph(settings.graph_file)

  gt_analyse_initial_graph(graph)
  logging.info('Bootstrapping ----')
  start_time = time.time()
  graph.bootstrap([hyperparams.bootstrap_t])
  t1 = round(time.time() - start_time, 2)
  graph.validate_ground_truth('bootstrapped-%s' % hyperparams.scenario, t1)
  logging.info('Bootstrapping time {}'.format(t1))

  logging.info('Linking ----')
  start_time = time.time()
  graph.link(merge_t)
  t2 = round(time.time() - start_time, 2)
  graph.validate_ground_truth('%s' % hyperparams.scenario, t1 + t2)
  logging.info('Linking time {}'.format(t2))

  logging.info('Successfully completed!')


# ---------------------------------------------------------------------------
# Ground truth analysis
# ---------------------------------------------------------------------------
def gt_analyse_initial_graph(graph):
  """
  Function to analyse the ground truth precision and recall values for the
  initial graph.

  :param graph: Graph
  """
  gt_links = graph.gt_links_dict

  links_dict = defaultdict(set)
  for node_id, node in graph.G.nodes(data=True):
    if node[c.TYPE1] == c.N_RELTV:
      s1 = graph.record_dict[node[c.R1]]
      s2 = graph.record_dict[node[c.R2]]
      sorted_key = tuple(sorted({s1[c.I_ID], s2[c.I_ID]}))
      links_dict[node[c.TYPE2]].add(sorted_key)

  # Count tp, fp, and fn
  true_positive_dict = defaultdict(int)
  false_positive_dict = defaultdict(int)
  false_negative_dict = defaultdict(int)

  false_positive_link_dict = defaultdict(list)
  false_negative_link_dict = defaultdict(list)

  for link in ['S-S']:
    for processed_link in links_dict[link]:
      if processed_link in gt_links[link]:
        true_positive_dict[link] += 1
      else:
        false_positive_dict[link] += 1
        false_positive_link_dict[link].append(processed_link)
    for gt_link in gt_links[link]:
      if gt_link not in links_dict[link]:
        false_negative_dict[link] += 1
        false_negative_link_dict[link].append(gt_link)

  # Dictionary to hold the precision and recall of each link type
  precision_dict = {}
  recall_dict = {}

  for link in ['S-S']:

    # Precision is tp / (tp + fp)
    precision_divisor = \
      float(true_positive_dict[link] + false_positive_dict[link])
    if precision_divisor == 0:
      precision_dict[link] = 0
    else:
      precision_dict[link] = round(true_positive_dict[link] * 100.0 / \
                                   precision_divisor, 2)

    # Recall is tp / (tp + fn)
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
                            '_initial-graph', ['S-S'], -1)

def baselines():
  """
  Function to run baselines of Dep-Graph and Attr-Sim on Civil Graph

  """

  # BASELINE Dep-Graph
  graph = SONG_GRAPH(settings.attribute_init_function, hyperparams.atomic_t)
  graph.deserialize_graph(settings.graph_file)
  start_time = time.time()
  graph.merge_baseline_depgraph(hyperparams.merge_t,
                                attributes_meta.songs_sim_func)
  t1 = round(time.time() - start_time, 2)
  graph.validate_ground_truth('baseline-dg', t1)

  ## BASELINE Attr-Sim
  graph = SONG_GRAPH(settings.attribute_init_function,  hyperparams.atomic_t)
  graph.deserialize_graph(settings.graph_file)
  graph.deserialize_graph(settings.graph_file)
  graph.merge_attr_sim_baseline(['S-S'], hyperparams.merge_t)

if __name__ == '__main__':
  log_file_name = util.get_logfilename(settings.output_home_directory + 'link')
  logging.basicConfig(filename=log_file_name, level=logging.INFO, filemode='w')

  ## PRE-REQUISITES
  pre_requisites()

  ## GENERATE GRAPH
  graph = generate_graph(hyperparams.atomic_t, settings.output_home_directory)
  util.persist_graph_gen_times('def_sg', stats.a_time, stats.r_time)

  # LINK
  stime = time.time()
  link(hyperparams.atomic_t, hyperparams.merge_t,
       settings.attribute_init_function)
  t = time.time() - stime
  logging.info('Graph processing time {}'.format(t))

  # Baselines
  baselines()
