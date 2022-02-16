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
# ---------------------------------------------------------------------------
# PRE-REQUISITES and GRAPH GENERATION
# ---------------------------------------------------------------------------
import logging
import multiprocessing as mp
import pickle
import time
from collections import defaultdict

from data import data_loader, bhic_transform as bt
from febrl import encode
from er.civil_graph import CIVIL_GRAPH
from common import constants as c, settings, stats, ambiguity, sim, util, \
  attributes_meta, hyperparams
from common.enums import Links, Role


# ---------------------------------------------------------------------------
# PRE-REQUISITES and GRAPH GENERATION
# ---------------------------------------------------------------------------

def generate_people_dict():
  """
  Function to generate the people dictionary.

  """
  people_dict = data_loader.enumerate_records()
  output_filename = '{}people-{}.gpickle'.format(settings.output_home_directory,
                                                 c.data_set)
  pickle.dump(people_dict.values(), open(output_filename, 'wb'))


def append_people_dict_frequencies(role_type):
  """
  Generate ambiguity clusters and append frequencies to the people list.

  :param role_type: Type of role considered for unique records for ambiguity
  :return:
  """
  people_list = pickle.load(open('{}people-{}.gpickle'.format(
    settings.output_home_directory, c.data_set), 'rb'))

  if c.data_set == 'bhic':
    ambiguity.cluster_attribute_combinations_wblocking(people_list, role_type,
                                                       (c.I_FNAME, c.I_SNAME,
                                                        c.I_ADDR1),
                                                       'fname-sname')
    people_list_wf = ambiguity.append_clustered_attr_f(people_list, role_type,
                                                       (c.I_FNAME, c.I_SNAME),
                                                       'fname-sname')
  else:
    # ambiguity.cluster_attribute_combinations(people_list, role_type,
    #                                          (c.I_FNAME, c.I_SNAME,
    #                                           c.I_ADDR1),
    #                                          'fname-sname-a1')

    people_list_wf = ambiguity.append_clustered_attr_f(people_list, role_type,
                                                       (c.I_FNAME, c.I_SNAME,
                                                        c.I_ADDR1),
                                                       'fname-sname-a1')
  pickle.dump(people_list_wf, open(settings.people_file, 'wb'))


def generate_atomic_nodes_small_ds(sim_func_dict):
  """
  Function to generate atomic nodes with blocking for small data sets.

  :param sim_func_dict: Attribute similarity initialization function
  """
  people_list = pickle.load(open('{}people-{}.gpickle'.format(
    settings.output_home_directory, c.data_set), 'rb'))

  atomic_node_dict = {}
  for i_attribute, sim_function in sim_func_dict.iteritems():
    atomic_node_dict[i_attribute] = {}

    value_set = {p[i_attribute] for p in people_list}
    assert '' not in value_set
    if None in value_set:
      value_set.remove(None)

    logging.info('---- {} : {}'.format(i_attribute, len(value_set)))
    value_list = list(value_set)
    value_list.sort()

    atomic_node_dict[i_attribute] = __get_node_key_sim_dict__(value_list,
                                                              sim_function)

  pickle.dump(atomic_node_dict, open(settings.atomic_node_file, 'wb'))
  logging.info('Successfully generated atomic nodes')


def generate_atomic_nodes_wblocking(sim_func_dict):
  """
  Function to generate atomic nodes with blocking for large data sets.

  :param sim_func_dict: Attribute similarity initialization function
  """

  people_list = pickle.load(open(settings.people_file, 'rb'))

  atomic_node_dict = defaultdict(dict)
  logging.info('Generating atomic nodes')
  for i_attribute, sim_function in sim_func_dict.iteritems():

    logging.info('Starting attribute %s' % i_attribute)

    # Generate blocks based on the first letter of the attribute value
    value_blocks = defaultdict(set)
    for p in people_list:
      attr = p[i_attribute]
      if attr is None or attr == '':
        continue

      attr = str(attr)
      for attr_part in attr.split(' '):
        key = encode.dmetaphone(attr_part)
        value_blocks[key].add(attr)

    logging.info('Attribute index %s, number of keys %s' % (i_attribute,
                                                            len(
                                                              value_blocks.keys())))

    pool = mp.Pool(settings.parallel_process_count)
    result_objects = [pool.apply_async(__get_node_key_sim_dict__,
                                       args=(value_set, sim_function))
                      for value_set in value_blocks.itervalues()]
    pool.close()
    pool.join()

    for result in result_objects:
      atomic_node_dict[i_attribute].update(result.get())

  pickle.dump(atomic_node_dict, open(settings.atomic_node_file, 'wb'))
  logging.info('Successfully generated atomic nodes')


def __get_node_key_sim_dict__(value_list, sim_function):
  """
  Function to get a dictionary of atomic node similarity.

  :param value_list: List of values
  :param sim_function: Similarity function
  :return: Atomic node similarity dictionary
  """
  atomic_node_dict = dict()
  value_list = list(value_list)
  for i in xrange(len(value_list)):
    for j in xrange(i, len(value_list)):
      key = (value_list[i], value_list[j])

      # Using cache is useless as the values are unique
      # For a larger data set, cache might get the program memory to be out
      sim_val = sim.get_pair_sim_no_cache(key, sim_function)

      if sim_val >= 0.8:
        atomic_node_dict[key] = sim_val
  return atomic_node_dict


def pre_requisites(role_type):
  """
  Function to run pre-requisites prior to the graph generation.

  :param role_type: Type of role considered for unique records for ambiguity
  """
  generate_people_dict()
  logging.info('Initial people dictionary generated')

  append_people_dict_frequencies(role_type)
  logging.info('Frequency clusters generated')

  stime = time.time()
  if c.data_set == 'bhic':
    generate_atomic_nodes_wblocking(attributes_meta.people_sim_func)
  else:
    generate_atomic_nodes_small_ds(attributes_meta.people_sim_func)
    data_loader.retrieve_ground_truth_links()
  t = time.time() - stime

  logging.info('Atomic nodes generated time {}'.format(t))


def generate_graph(atomic_t, output_directory):
  """
  Function to generate the graph.

  :param atomic_t: Atomic threshold
  :param output_directory: Output directory of the graph
  """
  logging.info('Atomic threshold {}'.format(atomic_t))

  atomic_node_dict = pickle.load(open(settings.atomic_node_file, 'rb'))
  graph = CIVIL_GRAPH(settings.attribute_init_function, atomic_t)
  graph.generate_graph(atomic_node_dict)
  graph.serialize_graph(
    '{}graph-{}.gpickle'.format(output_directory, str(hyperparams.atomic_t)))

  logging.info('Graph serialized successfully')

  return graph


def merge_graphs(atomic_t):
  """
  Function to merge different initial graph generated using different rules.

  :param atomic_t: Atomic threshold
  """

  # Populate graph names to be merged
  #
  graph_name = '%sgraph/def{}/graph-%s.gpickle' % (
    settings.output_home_directory, atomic_t)
  graph_list = [
    (attributes_meta.def6_must3, graph_name.format(6)),
    (attributes_meta.def8_must3, graph_name.format(8)),
    (attributes_meta.def9_must3, graph_name.format(9)),
    (attributes_meta.def10_must2_extra2, graph_name.format(10)),
    (attributes_meta.def11_must2_extra2, graph_name.format(11)),
    (attributes_meta.def13_must2_extra2, graph_name.format(13))
  ]

  # Loading the first graph
  #
  graph = CIVIL_GRAPH(graph_list[0][0], atomic_t)
  graph.deserialize_graph(graph_list[0][1])
  # verify.verify_relational_nodes(graph)
  logging.info('Initial graph {}'.format(graph_list[0][1]))

  # Updating the existing graph with the nodes and edges of other graphs
  #
  for init_func, graph_file_name in graph_list[1:]:
    logging.info('Graph {}'.format(graph_file_name))

    tmp_graph = CIVIL_GRAPH(init_func, atomic_t)
    tmp_graph.deserialize_graph(graph_file_name)

    existing_new_id_dict = {}

    __merge_add_atomic_nodes__(graph, tmp_graph, existing_new_id_dict)
    __merge_add_relationship_nodes__(graph, tmp_graph, existing_new_id_dict)
    __merge_add_edges__(graph, tmp_graph, existing_new_id_dict)

    # verify.verify_relational_nodes(graph)

    logging.info('Asserting graph {}'.format(graph_file_name))

  # Serializing the final graph
  #
  graph.serialize_graph(settings.graph_file)
  logging.info('Graph merged and serialized in %s' % settings.graph_file)


def __merge_add_atomic_nodes__(graph, graph2, existing_new_id_dict):
  attributes_list = attributes_meta.people_sim_func.keys()

  intersecting_node_count = 0
  new_node_count = 0

  for attribute in attributes_list:

    existing_atomic_nodes_list = {}
    for key, id in graph.atomic_nodes[attribute].iteritems():
      s_key = tuple(sorted(key))
      existing_atomic_nodes_list[s_key] = id

    # Iterate through the atomic nodes of graph 2
    for atomic_node_key, id in graph2.atomic_nodes[attribute].iteritems():

      atomic_node_key = tuple(sorted(atomic_node_key))

      # If the nodes are already in graph1
      if atomic_node_key in existing_atomic_nodes_list:
        intersecting_node_count += 1
        existing_new_id_dict[id] = existing_atomic_nodes_list[atomic_node_key]
        continue

      new_node_count += 1
      graph.node_id += 1
      node = graph2.G.nodes[id]
      graph.G.add_node(graph.node_id, **node)

      existing_new_id_dict[id] = graph.node_id
      existing_atomic_nodes_list[atomic_node_key] = graph.node_id
      graph.atomic_nodes[attribute][atomic_node_key] = graph.node_id
      graph.atomic_nodes[attribute][
        (atomic_node_key[1], atomic_node_key[0])] = graph.node_id

  logging.info('Atomic nodes in graph_2  {}'.format(len(graph2.atomic_nodes)))
  logging.info('Existing atomic nodes count {}'.format(intersecting_node_count))
  logging.info('New atomic nodes count {}'.format(new_node_count))

  g2_atomic_nodes = len([node_id for node_id, node in graph2.G.nodes(data=True)
                         if node[c.TYPE1] == c.N_ATOMIC])
  assert g2_atomic_nodes == len(existing_new_id_dict)


def __merge_add_relationship_nodes__(graph, graph2, existing_new_id_dict):
  existing_nodes_list = graph.relationship_nodes

  intersecting_node_count = 0
  new_node_count = 0

  # Iterate through the atomic nodes of graph 2
  for node_key, id in graph2.relationship_nodes.iteritems():

    # If the nodes are already in graph1
    if node_key in existing_nodes_list:
      intersecting_node_count += 1
      existing_new_id_dict[id] = existing_nodes_list[node_key]
      continue

    new_node_count += 1
    graph.node_id += 1
    node = graph2.G.nodes[id]
    graph.G.add_node(graph.node_id, **node)

    existing_new_id_dict[id] = graph.node_id
    graph.relationship_nodes[node_key] = graph.node_id
    graph.relationship_nodes[(node_key[1], node_key[0])] = graph.node_id

  logging.info('Relationship nodes in graph_2  {}'.format(len(
    graph2.relationship_nodes)))
  logging.info('Existing relationship nodes count {}'.format(
    intersecting_node_count))
  logging.info('New relationship nodes count {}'.format(new_node_count))

  assert len(graph2.G.nodes) == len(existing_new_id_dict)


def __merge_add_edges__(graph, graph2, existing_new_id_dict):
  logging.info('Initial graph edges count {}'.format(graph.G.number_of_edges()))
  logging.info('Initial graph 2 edges count {}'.format(
    graph2.G.number_of_edges()))

  count_existing_edges = 0
  count_new_edges = 0
  graph1_edges = graph.G.number_of_edges()
  for source, target, edge_data in graph2.G.edges(data=True):

    new_source = existing_new_id_dict[source]
    new_target = existing_new_id_dict[target]

    if graph.G.has_edge(new_source, new_target):

      existing_data = graph.G.get_edge_data(new_source, new_target)
      assert existing_data['t1'] == edge_data['t1']
      assert existing_data['t2'] == edge_data['t2']
      count_existing_edges += 1
    else:
      graph.G.add_edge(new_source, new_target, **edge_data)
      count_new_edges += 1

  logging.info('After graph edges count {}'.format(graph.G.number_of_edges()))
  logging.info('Existing edges count {}'.format(count_existing_edges))
  logging.info('New edges count {}'.format(count_new_edges))

  assert graph.G.number_of_edges() == (graph1_edges +
                                       graph2.G.number_of_edges() - count_existing_edges)


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
      p1 = graph.record_dict[node[c.R1]]
      p2 = graph.record_dict[node[c.R2]]
      gender = util.get_node_gender(p1[c.I_SEX], p2[c.I_SEX])
      if gender == 0:
        raise Exception('Wrong node')
      p1_cert_id = util.append_role(p1[c.I_CERT_ID], p1[c.I_ROLE], p1[c.I_SEX])
      p2_cert_id = util.append_role(p2[c.I_CERT_ID], p2[c.I_ROLE], p2[c.I_SEX])
      sorted_key = tuple(sorted({p1_cert_id, p2_cert_id}))
      links_dict[node[c.TYPE2]].add(sorted_key)

  # Count tp, fp, and fn
  true_positive_dict = defaultdict(int)
  false_positive_dict = defaultdict(int)
  false_negative_dict = defaultdict(int)

  false_positive_link_dict = defaultdict(list)
  false_negative_link_dict = defaultdict(list)

  for link in Links.list():
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

  for link in Links.list():

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
                            '_initial-graph', Links.list(), -1)

  if not settings.enable_analysis:
    return

  logging.info('FALSE NEGATIVES ----')
  records = graph.record_dict

  certid_id_dict = dict()
  for p in graph.record_dict.itervalues():
    cert_id = util.append_role(p[c.I_CERT_ID], p[c.I_ROLE],
                               p[c.I_SEX])
    certid_id_dict[cert_id] = p[c.I_ID]

  for link in ['Bb-Dd']:
    for id_1, id_2 in false_negative_link_dict[link]:
      id_1 = certid_id_dict[id_1]
      id_2 = certid_id_dict[id_2]
      logging.info('{} - {}'.format(id_1, id_2))
      logging.info('\t{}'.format(records[id_1]))
      logging.info('\t{}'.format(records[id_2]))


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
  graph = CIVIL_GRAPH(sim_attr_init_func, atomic_t)
  graph.deserialize_graph(settings.graph_file)
  logging.info('Gamma {}'.format(hyperparams.wa))

  # gt_analyse_initial_graph(graph)
  logging.info('Bootstrapping ----')
  start_time = time.time()
  graph.bootstrap([hyperparams.bootstrap_t])
  graph.refine_gt_measures()
  t1 = round(time.time() - start_time, 2)
  # if not c.data_set.startswith('bhic'):
  #   graph.validate_ground_truth('bootstrapped-%s' % hyperparams.scenario, t1)
  logging.info('Bootstrapping time {}'.format(t1))

  logging.info('Linking ----')
  start_time = time.time()
  graph.link(merge_t)
  graph.refine_gt_measures()
  t2 = round(time.time() - start_time, 2)
  # if not c.data_set.startswith('bhic'):
  #   graph.validate_ground_truth('linked-%s' % hyperparams.scenario, t1 + t2)

  logging.info('Refining ----')
  settings.enable_analysis = False
  start_time = time.time()
  graph.refine(merge_t)
  graph.refine_gt_measures()
  t3 = round(time.time() - start_time, 2)
  if not c.data_set.startswith('bhic'):
    graph.validate_ground_truth('%s' % hyperparams.scenario, t1 + t2 + t3)

  # graph.analyse_entities(Role.list())
  logging.info('Time logs :')
  logging.info('\tBootstrapping : {}'.format(round(t1, 2)))
  logging.info('\tLinking and Refining : {}'.format(round(t2 + t3, 2)))

  logging.info('Atomic enriched {}'.format(stats.atomic_enriched))
  logging.info('Split Entities b {}'.format(stats.split_entity_count_b))
  logging.info('Split Entities d {}'.format(stats.split_entity_count_d))
  logging.info('Removed Edges {}'.format(stats.removed_edge_count))
  logging.info('Added Node count {}'.format(stats.added_node_count))
  logging.info('Added group count {}'.format(stats.added_group_count))
  logging.info('Successfully completed!')



def bhic_data_gen():
  """
  Function to generate CSV files for births, marriages, and deaths of the BHIC
  data set from the existing XML's for a particular period of years

  :return:
  """
  bt.start = 1900
  bt.b_end = 1920
  bt.m_start = 1915
  bt.end = 1935

  logging.info('Births {} - {}'.format(bt.start, bt.b_end))
  logging.info('Marriages {} - {}'.format(bt.m_start, bt.end))
  logging.info('Deaths {} - {}'.format(bt.start, bt.end))

  settings.birth_file = settings.birth_file.format(bt.start)
  settings.marriage_file = settings.marriage_file.format(bt.start)
  settings.death_file = settings.death_file.format(bt.start)

  bt.transform_births()
  bt.transform_marriages()
  bt.transform_deaths()


def baselines():
  """
  Function to run baselines of Dep-Graph and Attr-Sim on Civil Graph

  """

  ## BASELINE Dep-Graph
  graph = CIVIL_GRAPH(settings.attribute_init_function, hyperparams.atomic_t)
  graph.deserialize_graph(settings.graph_file)
  start_time = time.time()
  graph.merge_baseline_depgraph(hyperparams.merge_t,
                                attributes_meta.people_sim_func)
  t1 = round(time.time() - start_time, 2)
  graph.validate_ground_truth('baseline-dg', t1)

  ## BASELINE Attr-Sim
  graph = CIVIL_GRAPH(settings.attribute_init_function, hyperparams.atomic_t)
  graph.deserialize_graph(settings.graph_file)
  graph.merge_attr_sim_baseline(Links.list(), hyperparams.merge_t)


if __name__ == '__main__':

  log_file_name = util.get_logfilename(settings.output_home_directory + 'link')
  logging.basicConfig(filename=log_file_name, level=logging.INFO, filemode='w')

  # PRE-REQUISITES
  attribute_init_function = eval(sys.argv[2])
  stime = time.time()
  bhic_data_gen()
  pre_requisites(settings.role_type)
  t = time.time() - stime
  logging.info('Pre-requisites time {}'.format(t))

  append_people_dict_frequencies(settings.role_type)

  # ## GENERATE GRAPH
  settings.attribute_init_function = attribute_init_function
  graph = generate_graph(atomic_t, output_directory)
  util.persist_graph_gen_times(sys.argv[2], stats.a_time, stats.r_time)
  logging.info('Successfully finished!')

  ## MERGE GRAPHS
  #
  # merge_graphs(hyperparams.atomic_t)

  ## LINK
  #
  stime = time.time()
  link(hyperparams.atomic_t, hyperparams.merge_t,
         settings.attribute_init_function)
  t = time.time() - stime
  logging.info('Graph processing time {}'.format(t))

  ## BASELINES
  # baselines()
