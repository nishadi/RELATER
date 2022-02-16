import logging
import pickle
import sys
import time
from collections import defaultdict

from data import data_loader
from er.hh_graph import HH_GRAPH
from common import constants as c, settings, util, ambiguity, sim, \
  attributes_meta, stats, hyperparams
from common.enums import HH_Links


def generate_people_dict():
  """
  Function to generate the people dictionary.

  """

  people_dict = data_loader.enumerate_hh_records()

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

  ambiguity.cluster_attribute_combinations(people_list, role_type,
                                           (c.HH_I_FNAME, c.HH_I_SNAME,
                                            c.HH_I_STATE),
                                           'fname-sname-state')
  people_list_wf = ambiguity.append_clustered_attr_f(people_list, role_type,
                                                     (c.HH_I_FNAME,
                                                      c.HH_I_SNAME,
                                                      c.HH_I_STATE),
                                                     'fname-sname-state')
  pickle.dump(people_list_wf, open(settings.people_file, 'wb'))


def generate_atomic_nodes(sim_func_dict):
  """
  Function to generate atomic nodes with blocking for small data sets.

  :param sim_func_dict: Attribute similarity initialization function
  """
  people_list = pickle.load(open(settings.people_file, 'rb'))

  atomic_node_dict = {}
  for i_attribute, sim_function in sim_func_dict.iteritems():
    atomic_node_dict[i_attribute] = defaultdict(dict)

    value_set = {p[i_attribute] for p in people_list}
    assert '' not in value_set
    if None in value_set:
      value_set.remove(None)

    logging.info('---- {} : {}'.format(i_attribute, len(value_set)))
    value_list = list(value_set)
    value_list.sort()

    for i in xrange(len(value_list)):
      for j in xrange(i, len(value_list)):
        key = (value_list[i], value_list[j])

        # Using cache is useless as the values are unique
        # For a larger data set, cache might get the program memory to be out
        sim_val = sim.get_pair_sim_no_cache(key, sim_function)

        if sim_val >= 0.8:
          atomic_node_dict[i_attribute][key] = sim_val

  pickle.dump(atomic_node_dict, open(settings.atomic_node_file, 'wb'))
  logging.info('Successfully generated atomic nodes')


def pre_requisites(role_type):
  """
  Function to run pre-requisites prior to the graph generation

  :param role_type: Type of role considered for unique records for ambiguity
  """
  generate_people_dict()
  logging.info('Initial people dictionary generated')

  append_people_dict_frequencies(role_type)
  logging.info('Frequency clusters generated')

  generate_atomic_nodes(attributes_meta.hh_sim_func)
  logging.info('Atomic nodes generated')


def generate_graph(atomic_t, census_year1, census_year2):
  """
  Function to generate graph.

  :param atomic_t: Atomic threshold
  :param census_year1: Census year 1
  :param census_year2: Census year 2
  :return: Generated graph
  """
  atomic_node_dict = pickle.load(open(settings.atomic_node_file, 'rb'))

  graph = HH_GRAPH(settings.attribute_init_function, atomic_t, census_year1,
                   census_year2)
  graph.generate_graph(atomic_node_dict)
  graph.serialize_graph(settings.graph_file)

  return graph


def link(atomic_t, merge_t, census_year1, census_year2):
  """
  Function to link a census graph.
  Graph theory measure based refinement is not employed because only two
  census snapshots are being linked resulting entities having records of at
  most 2.

  :param atomic_t: Atomic threshold
  :param merge_t: Merging threshold
  :param census_year1: Census year 1
  :param census_year2: Census year 2
  """
  attribute_init_function = attributes_meta.def_hh_link
  graph = HH_GRAPH(attribute_init_function, atomic_t, census_year1,
                   census_year2)
  graph.deserialize_graph(settings.graph_file)

  gt_analyse_initial_graph(graph)

  start_time = time.time()
  graph.bootstrap([hyperparams.bootstrap_t])
  t1 = round(time.time() - start_time, 2)
  logging.info('Bootstrapping time {}'.format(t1))
  graph.validate_ground_truth('bootstrapped-%s' % hyperparams.scenario, t1)

  start_time = time.time()
  graph.link(merge_t)
  t2 = round(time.time() - start_time, 2)
  # graph.validate_ground_truth('linked-%s' % hyperparams.scenario, t1 + t2)
  logging.info('Linking time {}'.format(t2))

  settings.enable_analysis = False
  start_time = time.time()
  graph.refine(merge_t)
  t3 = round(time.time() - start_time, 2)
  graph.validate_ground_truth('%s' % hyperparams.scenario, t1 + t2 + t3)
  logging.info('Refining time {}'.format(t3))


def baselines():
  """
  Function to run baselines of Dep-Graph and Attr-Sim on Civil Graph
  """
  ## BASELINE Dep-Graph
  graph = HH_GRAPH(attributes_meta.def_hh_link, hyperparams.atomic_t,
                   census_year1,
                   census_year2)
  graph.deserialize_graph(settings.graph_file)
  start_time = time.time()
  graph.merge_baseline_depgraph(hyperparams.merge_t,
                                attributes_meta.hh_sim_func)
  t1 = round(time.time() - start_time, 2)
  graph.validate_ground_truth('baseline-dg', t1)

  ## BASELINE Attr-Sim
  graph = HH_GRAPH(attributes_meta.def_hh_link, hyperparams.atomic_t,
                   census_year1,
                   census_year2)
  graph.deserialize_graph(settings.graph_file)
  graph.merge_attr_sim_baseline(HH_Links.list(), hyperparams.merge_t)


# ---------------------------------------------------------------------------
# Ground truth analysis
# ---------------------------------------------------------------------------

def gt_analyse_initial_graph(graph):
  """
  Analyse the ground truth precision and recall values for the initial graph.

  :param graph: Graph
  """
  #
  links_dict = defaultdict(set)
  for node_id, node in graph.G.nodes(data=True):
    if node[c.TYPE1] == c.N_RELTV:
      p1 = graph.record_dict[node[c.R1]]
      p2 = graph.record_dict[node[c.R2]]
      links_dict[node[c.TYPE2]].add((p1[c.I_ID], p2[c.I_ID]))

  # Count tp, fp, and fn
  true_positive_dict = defaultdict(int)
  false_positive_dict = defaultdict(int)
  false_negative_dict = defaultdict(int)

  false_positive_link_dict = defaultdict(list)
  false_negative_link_dict = defaultdict(list)

  for link in HH_Links.list():
    for processed_link in links_dict[link]:
      if processed_link in graph.gt_links_dict[link]:
        true_positive_dict[link] += 1
      else:
        false_positive_dict[link] += 1
        false_positive_link_dict[link].append(processed_link)
    for gt_link in graph.gt_links_dict[link]:
      if gt_link not in links_dict[link]:
        false_negative_dict[link] += 1
        false_negative_link_dict[link].append(gt_link)

  # Dictionary to hold the precision and recall of each link type
  precision_dict = {}
  recall_dict = {}

  for link in HH_Links.list():

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
  stats.persist_gt_analysis(graph.gt_links_dict, defaultdict(list), stats_dict,
                            '_initial-graph', HH_Links.list(), -1)

  if not settings.enable_analysis:
    return

  logging.info('FALSE NEGATIVES ----')
  records = graph.record_dict

  for link in HH_Links.list():
    for id_1, id_2 in false_negative_link_dict[link]:

      logging.info('{} - {}'.format(id_1, id_2))
      for id in [id_1, id_2]:
        p = records[id]
        logging.info(p)
        if p[c.HH_I_MOTHER] is not None:
          logging.info('\t m {}'.format(records[p[c.HH_I_MOTHER]]))
        if p[c.HH_I_FATHER] is not None:
          logging.info('\t f {}'.format(records[p[c.HH_I_FATHER]]))
        if p[c.HH_I_CHILDREN] is not None:
          for c_id in p[c.HH_I_CHILDREN]:
            logging.info('\t c {}'.format(records[c_id]))


if __name__ == '__main__':
  # settings.role_type = 'F'
  # c.data_set = 'ipums'

  # atomic_t = float(sys.argv[1])
  # merge_t = float(sys.argv[2])
  #
  # settings.__init_settings__(merge_t, atomic_t, role_type)
  # settings.scenario = 'rel-dac'

  census_year1 = 1870
  census_year2 = 1880

  log_file_name = util.get_logfilename(settings.output_home_directory + 'link')
  logging.basicConfig(filename=log_file_name, level=logging.INFO, filemode='w')

  ## PRE-REQUISITES
  pre_requisites(settings.role_type)
  append_people_dict_frequencies(settings.role_type)

  ## GENERATE GRAPH
  graph = generate_graph(hyperparams.atomic_t, census_year1, census_year2)
  util.persist_graph_gen_times('def_hh', stats.a_time, stats.r_time)

  ## LINK
  link(hyperparams.atomic_t, hyperparams.merge_t, census_year1, census_year2)

  ## BASELINES
  # baselines()
