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
import csv
import logging
import os
import pickle
import time
from collections import defaultdict, deque
import networkx as nx
import common.constants as c
from common import sim, stats, settings, hyperparams
from common.enums import State


class BASE_GRAPH():

  def __init__(self, record_dict=None):

    # Create a new networkx graph that is used to hold nodes and edges for
    # processing.
    self.G = nx.DiGraph()

    # Dictionary holding the people obtained by merging the initial records.
    # Person ID acts as the key in the dictionary.
    self.entity_dict = {}

    # Dict to hold the keys of atomic nodes such as fname1-fname2. This is used
    # to keep track of already available atomic node keys to improve the
    # performance. Keys are sorted in alphabetical order.
    self.atomic_nodes = defaultdict(dict)

    # Dict to hold the keys of relationship nodes to node_id mapping.
    self.relationship_nodes = {}

    # Set to hold the keys of relationship nodes such as person_id1-person_id2.
    # This is used to keep track of already available relationship node keys to
    # improve the performance. Keys are sorted in alphabetical order.
    # If we use the dictionary in `self.relationship_nodes`, graph generation
    # takes unacceptable time.
    self.relationship_node_key_set = set()

    # ID used for indexing the nodes.
    self.node_id = 0

    # ID used for indexing the entities.
    self.entity_id = 0

    # Dictionary holding the records enumerated from the original certificates.
    # ID acts as the key in the dictionary.
    self.record_dict = record_dict

    self.group_dict = None

    self.unique_record_count = 0

    self.gt_links_dict = None

    # Ground truth analysis variables
    self.linkage_reason_dict = defaultdict(dict)

    # Create necessary directories
    er_dir = '{}er/'.format(settings.output_home_directory)
    if not os.path.isdir(er_dir):
      os.makedirs(er_dir)

    self.attributes = None

  # ---------------------------------------------------------------------------
  # Serialization and deserialization
  # ---------------------------------------------------------------------------
  def serialize_graph(self, graph_filename):
    """
    Method to serialize graph.
    :param graph_filename: Graph file name
    :param people_filename: People file name
    """
    stime = time.time()
    nx.write_gpickle(self.G, graph_filename)
    t = time.time() - stime
    logging.info('Graph serialize time {}'.format(t))

  def deserialize_graph(self, graph_file):
    """
    Method to deserialize graph.
    """
    stime = time.time()
    graph = nx.read_gpickle(graph_file)
    t = time.time() - stime
    logging.info('Graph node count {}'.format(graph.number_of_nodes()))
    logging.info('Graph edge count {}'.format(graph.number_of_edges()))
    logging.info('Graph deserialize time {}'.format(t))

    self.G = graph

    # Setting the current node ID
    self.node_id = max(list(self.G.nodes))

    atomic_node_count = 0
    relationship_node_count = 0
    for node_id, node in self.G.nodes(data=True):
      key1 = (node[c.R1], node[c.R2])
      key2 = (node[c.R2], node[c.R1])

      if node[c.TYPE1] == c.N_ATOMIC:

        atomic_node_count += 1
        self.atomic_nodes[node[c.TYPE2]][key1] = node_id
        self.atomic_nodes[node[c.TYPE2]][key2] = node_id

      elif node[c.TYPE1] == c.N_RELTV:
        relationship_node_count += 1
        self.relationship_nodes[key1] = node_id
        self.relationship_nodes[key2] = node_id

    logging.info('Atomic nodes %s' % atomic_node_count)
    logging.info('Relationship nodes %s' % relationship_node_count)

  # ---------------------------------------------------------------------------
  # Graph generation
  # ---------------------------------------------------------------------------

  def generate_graph(self, atomic_node_dict):
    pass

  def __gen_relationship_nodes__(self):
    pass

  def __add_atomic_nodes__(self, atomic_node_dict, attributes):
    """
    Function to generate atomic nodes.

    :param atomic_node_dict:
    :param attributes:
    """

    logging.info('Atomic node addition\n Starting node id %s' % self.node_id)
    node_id = self.node_id

    for attributes_cat in [attributes.must_attr, attributes.core_attr,
                           attributes.extra_attr]:

      for i_attribute in attributes_cat.attributes:

        self.atomic_nodes[i_attribute] = {}

        for key, atomic_sim in atomic_node_dict[i_attribute].iteritems():
          if atomic_sim >= attributes_cat.min_sim:
            node_id += 1
            self.G.add_node(node_id, r1=key[0], r2=key[1], t1=c.N_ATOMIC,
                            t2=i_attribute, s=atomic_sim, st=State.MERGED)

            # Add both keys
            self.atomic_nodes[i_attribute][key] = node_id
            self.atomic_nodes[i_attribute][(key[1], key[0])] = node_id
        logging.info('Attribute %s ends with node id %s' % (i_attribute,
                                                            node_id))

    self.node_id = node_id

  def __add_single_node__(self, p1, p2, relationship_type, attributes):
    """
    Function to add a single node to the graph including the
    edges to atomic nodes.

    :param p1: Person 1 or ID of Person 1
    :param p2: person 2 or ID of Person 2
    :param relation(41, 42) in self.relationship_node_key_setship_type: Type of node
    :return: Node ID or None if the node is already added or node does not
    has sufficient atomic nodes (based on settings.atomic_node_count_t) to be
    considered as a valid relationship node.
    """

    # If person IDs are provided, persons are retrieved from the dictionary.
    if type(p1) == int:
      p1 = self.record_dict[p1]
      p2 = self.record_dict[p2]

    I_ID = c.I_ID
    N_RELTV = 'R'
    E_REAL = 'R'

    # Check if the node is a duplicate node
    node_key = (p2[I_ID], p1[I_ID])

    # Retrieve and validate if the incoming atomic node count is
    # greater than the threshold values of `self.attributes`
    incoming_atomic_nodes = [
      (attr_index, self.atomic_nodes[attr_index][(p1[attr_index],
                                                  p2[attr_index])])
      for attr_index in attributes.attr_index_list
      if (p1[attr_index], p2[attr_index]) in self.atomic_nodes[attr_index]]

    must_count = 0
    core_count = 0
    extra_count = 0
    for attr_index, _ in incoming_atomic_nodes:
      if attr_index in attributes.core_attr.attributes:
        core_count += 1
      elif attr_index in attributes.extra_attr.attributes:
        extra_count += 1
      elif attr_index in attributes.must_attr.attributes:
        must_count += 1

    relationship_node_exists = \
      must_count >= attributes.must_attr.least_count and \
      core_count >= attributes.core_attr.least_count and \
      extra_count >= attributes.extra_attr.least_count

    if relationship_node_exists:

      # This check is expensive in time
      if node_key in self.relationship_node_key_set:
        return None

      self.node_id += 1

      self.relationship_node_key_set.add(node_key)
      self.relationship_node_key_set.add((node_key[1], node_key[0]))

      self.G.add_node(self.node_id, r1=p1[I_ID], r2=p2[I_ID], t1=N_RELTV,
                      t2=relationship_type, s=0, st=State.ACTIVE)

      # Add incoming edges for the atomic nodes
      for r_node_type, r_node_id in incoming_atomic_nodes:
        self.G.add_edge(r_node_id, self.node_id, t1=E_REAL, t2=r_node_type)
      return self.node_id

    return None

  # ---------------------------------------------------------------------------
  # Graph processing
  # ---------------------------------------------------------------------------
  def __enumerate_graph_groups_with_avg_similarity__(self):

    # Optimizations
    nodes = self.G.nodes

    group_dict = defaultdict(list)
    for cc in nx.strongly_connected_components(self.G):

      if len(cc) == 1 and self.G.nodes[next(iter(cc))][c.TYPE1] == c.N_ATOMIC:
        continue
      sim_list = [nodes[node_id][c.SIM_ATOMIC] for node_id in cc]
      cc_avg_sim = sum(sim_list) * 1.0 / len(sim_list)
      group_dict[len(cc)].append((cc, cc_avg_sim))

      # freq_list = [nodes[node_id][c.SIM_ATOMIC] for node_id in cc]
      # cc_avg_freq = sum(freq_list) * 1.0 / len(freq_list)
      # group_dict[len(cc)].append((cc, cc_avg_sim, cc_avg_freq))

    logging.info('Connected components summary')
    for group_size, cc_list in group_dict.iteritems():
      logging.info('\tcc size %s : number of cc\'s %s' % (group_size,
                                                          len(cc_list)))
    self.group_dict = group_dict

  def __enrich_atomic_attribute_node__(self, node_id, highest_sim_pair,
                                       sim_attr_i, existing_attr_node_id_dict):
    # Retrieve the highest similar node ID if it already exists
    highest_sim_atomic_node_id = self.atomic_nodes[sim_attr_i].get(
      highest_sim_pair, None)

    if highest_sim_atomic_node_id is None:
      return False

    # If we already have an atomic node for this attribute,
    if sim_attr_i in existing_attr_node_id_dict:

      # Check if the highest similar pair is the same as the existing atomic
      # node. If they are the same, we do not need to change anything.
      if existing_attr_node_id_dict[sim_attr_i] == highest_sim_atomic_node_id:
        return False

      # If they are different, remove the existing edge.
      self.G.remove_edge(existing_attr_node_id_dict[sim_attr_i], node_id)

    # Add new edge between the highest similar atomic node and this node.
    self.G.add_edge(highest_sim_atomic_node_id, node_id, t1=c.E_REAL,
                    t2=sim_attr_i)
    return True

  def __enrich_node__(self, node_id, node, attr_sim_func_dict):
    atomic_enriched = False

    # Get the person records associated with the node
    p1 = self.record_dict[node[c.R1]]
    p2 = self.record_dict[node[c.R2]]

    # If none of the persons are merged, it is not needed to enrich the
    # node with the merged information
    #
    if p1[c.I_ENTITY_ID] is None and p2[c.I_ENTITY_ID] is None:
      return p1, p2

    # For atomic nodes, create a dictionary
    #   attribute type --> node id
    # For relationship nodes, create a dictionary
    #   relationship type --> node id
    #
    attr_node_id_dict = {}
    relationship_node_id_dict = {}

    for edge in self.G.in_edges(node_id, data=True):
      edge_data = edge[2]

      if edge_data[c.TYPE1] == c.E_REAL:
        attr_node_id_dict[edge_data[c.TYPE2]] = edge[0]

      # Filter the real valued edges
      else:
        relationship_node_id = edge[0]
        relationship_node_id_dict[edge_data[c.TYPE2]] = relationship_node_id

    # If both of the persons are merged
    #
    if p1[c.I_ENTITY_ID] is not None and p2[c.I_ENTITY_ID] is not None:

      entity1 = self.entity_dict[p1[c.I_ENTITY_ID]]
      entity2 = self.entity_dict[p2[c.I_ENTITY_ID]]
      o1, o2 = entity1, entity2

      # If both entities are the same, these two person records are already
      # merged.
      if p1[c.I_ENTITY_ID] == p2[c.I_ENTITY_ID] or hyperparams.scenario.startswith('noprop'):
        return o1, o2

      # Enrich ATOMIC nodes
      #
      # Iterate through the attributes to check if each atomic node
      # corresponding to each attribute value is the optimal combination. If
      # not, remove the existing edge with the atomic node and add a
      # new edge between the highest similar atomic attribute pair.
      for sim_attr_i, sim_attr_func in attr_sim_func_dict.iteritems():
        highest_sim_pair = ()
        highest_sim = 0

        # Get all the attribute values existing for the attribute for
        # entity 1 and entity 2
        if sim_attr_i in entity1 and sim_attr_i in entity2:
          e1_value_set = entity1[sim_attr_i]
          e2_value_set = entity2[sim_attr_i]

          # Find the highest similarity pair
          for e1_value in e1_value_set:
            for e2_value in e2_value_set:
              sim_val = sim.get_pair_sim((e1_value, e2_value), sim_attr_func)
              if sim_val > highest_sim:
                highest_sim = sim_val
                highest_sim_pair = (e1_value, e2_value)
              if highest_sim == 1.0:
                break

          enriched = self.__enrich_atomic_attribute_node__(node_id,
                                                           highest_sim_pair,
                                                           sim_attr_i,
                                                           attr_node_id_dict)
          atomic_enriched = atomic_enriched or enriched

      # Update the atomic similarity value of the node
      if atomic_enriched:
        self.calculate_sim_atomic(node_id, node, p1, p2)

    else:
      if p1[c.I_ENTITY_ID] is None:
        entity = self.entity_dict[p2[c.I_ENTITY_ID]]
        person = p1
      else:
        entity = self.entity_dict[p1[c.I_ENTITY_ID]]
        person = p2

      o1, o2 = entity, person
      if hyperparams.scenario.startswith('noprop'):
        return o1, o2

      # Iterate through the attributes to check if each atomic node
      # corresponding to each attribute value is the optimal combination. If
      # not, remove the existing edge with the atomic node and add a
      # new edge between the highest similar atomic attribute pair.
      for sim_attr_i, sim_attr_func in attr_sim_func_dict.iteritems():
        if person[sim_attr_i] is None:
          continue

        highest_sim_pair = ()
        highest_sim = 0

        # Get all the attribute values existing for the attribute for
        # entity
        entity_value_set = entity[sim_attr_i]

        if person[sim_attr_i] is not None:
          # Find the highest similarity pair
          for e_value in entity_value_set:
            sim_val = sim.get_pair_sim((e_value, person[sim_attr_i]),
                                       sim_attr_func)
            if sim_val > highest_sim:
              highest_sim = sim_val
              highest_sim_pair = (e_value, person[sim_attr_i])
            if highest_sim == 1.0:
              break

          enriched = self.__enrich_atomic_attribute_node__(node_id,
                                                           highest_sim_pair,
                                                           sim_attr_i,
                                                           attr_node_id_dict)
          atomic_enriched = atomic_enriched or enriched

      # Update the atomic similarity value of the node
      if atomic_enriched:
        self.calculate_sim_atomic(node_id, node, p1, p2)

    if atomic_enriched:
      stats.atomic_enriched += 1
      logging.debug('--- Atomic enriched {}'.format(node[c.SIM_ATOMIC]))
    else:
      logging.debug('--- NOT Atomic enriched')

    return o1, o2

  def bootstrap(self, bootstrap_threshold_list):
    pass

  def link_gdg(self, merge_t, attr_sim_func_dict):

    logging.debug('MERGING STEP')

    # Optimizations
    nodes = self.G.nodes
    relationship_nodes = self.relationship_nodes
    record_dict, entity_dict = self.record_dict, self.entity_dict
    inactive, nonmerge = State.INACTIVE, State.NONMERGE

    enrich_node = self.__enrich_node__
    __update_reason__ = self.__update_reason__

    # Initializing the nodes queue
    node_id_set_queue = deque()
    if hyperparams.scenario.startswith('norel'):
      # Initializing the nodes queue for ablation study without relationship structure
      calculate_sim_atomic = self.calculate_sim_atomic
      for node_id, node in self.G.nodes(data=True):
        if node[c.TYPE1] == c.N_ATOMIC:
          continue
        calculate_sim_atomic(node_id, node)
        node_id_set_queue.append({node_id})
    else:
      for group_size in sorted(self.group_dict.iterkeys(), reverse=True):

        # In songs data sets with basic entities, such as 'msd' and 'mb',
        # we only have singletons
        if group_size == 1 and c.data_set not in ['msd', 'mb']:
          for node_id_set, sim in self.group_dict[1]:
            for node_id in node_id_set:
              node = nodes[node_id]
              self.linkage_reason_dict[node[c.TYPE2]][
                (node[c.R1], node[c.R2])] = 'SIN'
        else:
          logging.debug('Merging groups of {}'.format(group_size))
          for node_id_set, sim in sorted(self.group_dict[group_size],
                                         key=lambda a: a[1], reverse=True):
            unmerged_nodes = list(filter(lambda node_id: nodes[node_id][c.STATE]
                                                         != 'M', node_id_set))
            if len(unmerged_nodes) == 0:
              continue
            node_id_set_queue.append(node_id_set)

    logging.info('Node id set queue length {}'.format(len(node_id_set_queue)))
    i = 0
    # Processing the nodes queue
    while node_id_set_queue:

      i += 1
      if i % 10000 == 0:
        logging.info('\tIterating at {}'.format(i))

      # Get the node at the front of the queue
      init_node_id_set = node_id_set_queue.popleft()

      if c.data_set in ['msd', 'mb']:
        node_id = list(init_node_id_set)[0]
        node = nodes[node_id]
        node[c.STATE] = inactive
        if self.__is_valid_node_merge__({node_id}) and node[
          c.SIM_ATOMIC] >= merge_t:
          self.__merge_nodes__({node_id})
        continue

      # Single nodes can get added while adding non-existing pairs.
      # Therefore this needs to be checked.
      if len(init_node_id_set) == 1:
        continue

      node_id_sim_list = list()

      # Enrich the set of nodes and add them to a list with tuples of
      # (node_id, sim)
      for node_id in init_node_id_set:
        node = nodes[node_id]
        assert node[c.SIM_ATOMIC] != 0

        o1, o2 = enrich_node(node_id, node, attr_sim_func_dict)
        if self.__is_valid_entity_role_merge__(o1, o2,
                                               init_node_id_set,
                                               record_dict,
                                               relationship_nodes,
                                               nodes, merge_t,
                                               node_id_set_queue) and \
            self.__is_valid_node_merge__({node_id}):
          node[c.STATE] = inactive
          node_id_sim_list.append((node_id, node[c.SIM_ATOMIC]))
        else:
          node[c.STATE] = nonmerge

      # Sort the node list based on the similarity
      sorted_sim_list = sorted(node_id_sim_list, key=lambda x: x[1],
                               reverse=True)

      # Try to merge them until the group reduced to a pair,
      # by removing the node with min sim in each iteration.
      last_n = len(sorted_sim_list) - 1 if hyperparams.scenario.startswith(
        'norel') else 1 # Ablation No REL
      for n in xrange(len(sorted_sim_list), last_n, -1):
        sorted_node_id_sim = sorted_sim_list[0:n]
        node_id_set = map(lambda x: x[0], sorted_node_id_sim)
        sim_list = map(lambda x: x[1], sorted_node_id_sim)
        sim = sum(sim_list) * 1.0 / len(sim_list)

        if sim >= merge_t:
          self.__merge_nodes__(node_id_set)
          if init_node_id_set in stats.added_node_id_set:
            stats.added_merged_node_id_count += 1
          break

        # # Analysis
        # if n == 2:
        #   __update_reason__(node_id_set, nodes, 'LSIM')
        # else:
        #   ls_node_id = sorted_sim_list[(n - 1)][0]
        #   ls_node = nodes[ls_node_id]
        #   self.linkage_reason_dict[ls_node[c.TYPE2]][
        #     (ls_node[c.R1], ls_node[c.R2])] = 'LSIM'

    logging.info('Enriched node count %s' % (stats.atomic_enriched))
    logging.info('Added neighbour count %s' % (stats.added_neighbors_count))
    logging.info('Activated count %s' % (stats.activated_neighbors_count))

    print 'Merge %s added node count %s' % (
      merge_t, stats.added_neighbors_count)
    if len(stats.added_group_size) > 0:
      print 'Merge %s added node group avg %s' % (merge_t, (
          sum(stats.added_group_size) * 1.0 / len(stats.added_group_size)))
      print 'Merge %s added node group max %s' % (
        merge_t, max(stats.added_group_size))
      print 'Merge %s added node group merged count %s' % (
        merge_t, stats.added_merged_node_id_count)
      p = stats.added_merged_node_id_count * 100.0 / len(
        stats.added_node_id_set)
      print 'Merge %s added node group merged percentage %s%%' % (
        merge_t, p)

  def __is_valid_node_merge__(self, node_id_set):
    pass

  def __is_valid_entity_role_merge__(self, o1, o2, node_id_set, people_dict,
                                     relationship_nodes, nodes, merge_t,
                                     node_id_set_queue=None):
    '''


    :param o1: Person record or entity if the person record is already
    assigned to an entity
    :param o2: Person record or entity if the person record is already
    assigned to an entity
    :param node_id_set:
    :param people_dict:
    :param relationship_nodes:
    :param nodes:
    :param merge_t:
    :param node_id_set_queue:
    :return:
    '''
    pass

  def __merge_nodes__(self, node_id_set):
    pass

  # ---------------------------------------------------------------------------
  # Groundtruth processing and analysis
  # ---------------------------------------------------------------------------
  def validate_ground_truth(self, graph_scenario, t):
    pass

  def calculate_persist_measures(self, processed_links_dict, links_list,
                                 graph_scenario, t, records=None, authors=None):
    # Count tp, fp, and fn
    true_positive_dict = defaultdict(int)
    false_positive_dict = defaultdict(int)
    false_negative_dict = defaultdict(int)

    false_positive_link_dict = defaultdict(list)
    false_negative_link_dict = defaultdict(list)

    for link in links_list:
      for processed_link in processed_links_dict[link]:
        if processed_link in self.gt_links_dict[link]:
          true_positive_dict[link] += 1
        else:
          false_positive_dict[link] += 1
          false_positive_link_dict[link].append(processed_link)
      for gt_link in self.gt_links_dict[link]:
        if gt_link not in processed_links_dict[link]:
          false_negative_dict[link] += 1
          false_negative_link_dict[link].append(gt_link)

    # Dictionary to hold the precision and recall of each link type
    precision_dict = {}
    recall_dict = {}
    for link in links_list:
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
    stats.persist_gt_analysis(self.gt_links_dict, processed_links_dict,
                              stats_dict, graph_scenario, links_list, t)

    if not settings.enable_analysis:
      return

    self.analyse_fp_fn(links_list, false_negative_link_dict,
                       false_positive_link_dict, records, authors)

  def analyse_fp_fn(self, links_list, false_negative_link_dict,
                    false_positive_link_dict, records, authors=None):
    pass

  def analyse_entities(self, role_list):

    entity_role_count_stats = defaultdict(list)
    entity_role_pid_set = defaultdict(set)
    singletons = defaultdict(set)
    singletons_per_decade = defaultdict(lambda: defaultdict(set))

    for entity in self.entity_dict.itervalues():
      for role in role_list:
        # Appending length of each role
        entity_role_count_stats[role].append(len(entity[c.ROLES][role]))

        # Appending PIDs
        for pid, _, _ in entity[c.ROLES][role]:
          entity_role_pid_set[role].add(pid)

    summary_dict = defaultdict(dict)

    logging.info('ENTITY count summary stats:')
    for role, count_list in entity_role_count_stats.iteritems():
      logging.info('\t{} : min {}'.format(role, min(count_list)))
      logging.info('\t{} : max {}'.format(role, max(count_list)))
      avg_count = sum(count_list) * 1.0 / len(count_list)
      logging.info('\t{} : avg {}'.format(role, avg_count))

      summary_dict[role] = {
        'dataset': c.data_set,
        'role': role,
        'entity-avg': avg_count,
        'entity-max': max(count_list)
      }

    # Counting singletons
    for pid, p in self.record_dict.iteritems():
      role = p[c.I_ROLE]
      if pid in entity_role_pid_set[role]:
        pass
      else:
        singletons[role].add(pid)

    # Getting role counts
    role_record_count = defaultdict(int)
    for role in role_list:
      p_list = filter(lambda p: p[c.I_ROLE] == role,
                      self.record_dict.itervalues())
      role_record_count[role] = len(p_list)

    logging.info('SINGLETON summary stats:')
    for role in role_list:
      if role_record_count[role] > 0:
        percentage = len(singletons[role]) * 100.0 / role_record_count[role]
        logging.info('\t{} - {} - ({})'.format(role, len(singletons[role]),
                                               percentage))
        summary_dict[role]['singleton'] = percentage

    filename = settings.output_home_directory + 'summary.csv'
    file_exists = os.path.isfile(filename)

    with open(filename, 'a') as results_file:
      heading_list = ['dataset', 'role', 'entity-avg', 'entity-max',
                      'singleton']
      writer = csv.DictWriter(results_file, fieldnames=heading_list)
      if not file_exists:
        writer.writeheader()

      for role, role_summary_dict in summary_dict.iteritems():
        writer.writerow(role_summary_dict)

    entity_pkl_file = settings.output_home_directory + 'entities.pickle'
    pickle.dump(self.entity_dict, open(entity_pkl_file, 'w'))

    merged_people_pkl_file = settings.output_home_directory + 'merged-people.pickle'
    pickle.dump(self.record_dict, open(merged_people_pkl_file, 'w'))

  # ---------------------------------------------------------------------------
  # Baselines
  # ---------------------------------------------------------------------------
  def merge_baseline_depgraph(self, merge_threshold, attr_sim_func_dict):

    merged_node_count = 0
    calculate_sim = self.__depgraph_calculate_sim__
    calculate_sim_atomic = self.__depgraph_calculate_attr_sim__
    active = State.ACTIVE
    inactive = State.INACTIVE
    merge_nodes = self.__merge_nodes__
    enrich_node = self.__enrich_node__
    is_valid_merge = self.__is_valid_node_merge__

    # Queue up the nodes
    relationship_nodes_list = filter(lambda x: x[1][c.TYPE1] == c.N_RELTV,
                                     self.G.nodes(data=True))
    for node_id, node in relationship_nodes_list:
      calculate_sim_atomic(node_id, node)

    sorted_relationship_node_list = sorted(relationship_nodes_list,
                                           key=lambda x: x[1][c.SIM_ATTR],
                                           reverse=True)
    nodes_queue = deque(sorted_relationship_node_list)

    while nodes_queue:

      node_id, node = nodes_queue.popleft()

      if node[c.STATE] == active:

        if c.SIM_ATOMIC not in node.iterkeys():
          calculate_sim_atomic(node_id, node)

        enrich_node(node_id, node, attr_sim_func_dict)
        calculate_sim(node_id, node)

        if node[c.SIM] >= merge_threshold:

          if is_valid_merge([node_id]):
            logging.debug('--- Is valid merge')
            merge_nodes([node_id])
            merged_node_count += 1

            # Activate neighbors
            self.__activate_neighbors__(node_id, nodes_queue)

          else:
            # todo :
            node[c.STATE] = State.NONMERGE
            logging.debug('--- NOT Is valid merge')
        else:
          node[c.STATE] = inactive
          logging.debug('--- is below threshold')

    print merged_node_count

  def __activate_neighbors__(self, node_id, nodes_queue):
    """
    Function to activate the relationship nodes exis
    :param node_id:
    :param nodes_queue:
    :return:
    """

    # Iterate through the edges
    for edge in list(self.G.in_edges(node_id, data=True)):
      edge_data = edge[2]

      # Filter the boolean valued strong neighbor edges
      # Activate and append to the front of the queue
      if edge_data[c.TYPE1] == c.E_BOOL_S:
        neighbor_node = self.G.nodes[edge[0]]
        # Only inactive nodes are activated. Non-merge nodes are not activated
        if neighbor_node[c.STATE] == State.INACTIVE:
          stats.activated_neighbors_count += 1
          neighbor_node[c.STATE] = State.ACTIVE
          nodes_queue.appendleft((edge[0], neighbor_node))

      # Filter the boolean valued weak neighbor edges
      # Activate and append to the end of the queue
      if edge_data[c.TYPE1] == c.E_BOOL_W:
        neighbor_node = self.G.nodes[edge[0]]
        if neighbor_node[c.STATE] == State.INACTIVE:
          stats.activated_neighbors_count += 1
          neighbor_node[c.STATE] = State.ACTIVE
          nodes_queue.append((edge[0], neighbor_node))

  def merge_attr_sim_baseline(self, links_list, merging_t):

    atomic_average = sim.calculate_weighted_atomic_str_sim
    processed_link_dict = defaultdict(set)

    TYPE1, TYPE2, SIM, N_ATOMIC = c.TYPE1, c.TYPE2, c.SIM, c.N_ATOMIC
    R1, R2 = c.R1, c.R2

    start_time = time.time()
    for node_id, node in self.G.nodes(data=True):

      if node[TYPE1] == N_ATOMIC:
        continue

      atomic_sim_dict = {
        self.G.nodes[neighbor_id][TYPE2]: self.G.nodes[neighbor_id][SIM]
        for neighbor_id in self.G.predecessors(node_id)
        if self.G.nodes[neighbor_id][TYPE1] == N_ATOMIC}
      atomic_sim = atomic_average(atomic_sim_dict, self.attributes)

      if atomic_sim >= merging_t:
        sorted_key = tuple(sorted({node[R1], node[R2]}))
        processed_link_dict[node[TYPE2]].add(sorted_key)

    t = round(time.time() - start_time, 2)
    self.calculate_persist_measures(processed_link_dict, links_list,
                                    'baseline-attr-sim', t)

  # ---------------------------------------------------------------------------
  # Graph processing Utils
  # ---------------------------------------------------------------------------

  def __update_reason__(self, node_id_set, nodes, reason):

    # Updating reasons
    for tmp_node_id in node_id_set:
      tmp_node = nodes[tmp_node_id]
      self.linkage_reason_dict[tmp_node[c.TYPE2]][
        (tmp_node[c.R1], tmp_node[c.R2])] = reason

  def __get_group_avg_similarity__(self, cc):
    sim_list = list()
    for node_id in cc:
      node = self.G.nodes[node_id]
      sim_list.append(node[c.SIM_ATOMIC])
    return sum(sim_list) * 1.0 / len(sim_list)

  def __get_existing_rel_node_id__(self, id1, id2):
    if (id1, id2) in self.relationship_nodes:
      return self.relationship_nodes[(id1, id2)]
    if (id2, id1) in self.relationship_nodes:
      return self.relationship_nodes[(id2, id1)]
    return None

  def calculate_sim_atomic(self, node_id, node, p1=None, p2=None):
    """
    Method to calculate the atomic similarity of relationship nodes.
    """

    if p1 is None:
      p1 = self.record_dict[node[c.R1]]
    if p2 is None:
      p2 = self.record_dict[node[c.R2]]

    # For all atomic neighbors, get type to similarity dictionary
    nodes = self.G.nodes
    N_ATOMIC = c.N_ATOMIC
    SIM, TYPE1, TYPE2 = c.SIM, c.TYPE1, c.TYPE2

    atomic_sim_dict = {}
    for neighbor_id in self.G.predecessors(node_id):
      if nodes[neighbor_id][TYPE1] == N_ATOMIC:
        neighbor_node = nodes[neighbor_id]
        atomic_sim_dict[neighbor_node[TYPE2]] = neighbor_node[SIM]

    sim.atomic_amb_weighted_average(node, atomic_sim_dict, p1, p2,
                                    self.unique_record_count, self.attributes)

  def __depgraph_calculate_sim__(self, node_id, node):
    """
    Method to calculate the atomic similarity of relationship nodes.
    """

    beta = 0.1
    gamma = 0.05

    bool_strong_neighbor_count = 0
    bool_strong_neighbor_merged_count = 0
    bool_weak_neighbor_count = 0
    bool_weak_neighbor_merged_count = 0

    # Iterate node edges
    for edge in self.G.in_edges(node_id, data=True):
      edge_data = edge[2]

      if edge_data[c.TYPE1] == c.E_BOOL_S:
        rel_node = self.G.nodes[edge[0]]
        bool_strong_neighbor_count += 1
        if rel_node[c.STATE] == State.MERGED:
          bool_strong_neighbor_merged_count += 1
      elif edge_data[c.TYPE1] == c.E_BOOL_W:
        rel_node = self.G.nodes[edge[0]]
        bool_weak_neighbor_count += 1
        if rel_node[c.STATE] == State.MERGED:
          bool_weak_neighbor_merged_count += 1

    atomic_sim = node[c.SIM_ATTR]
    node[c.SIM] = atomic_sim * 0.90 + \
                  bool_strong_neighbor_merged_count * beta + \
                  bool_weak_neighbor_merged_count * gamma

  def __depgraph_calculate_attr_sim__(self, node_id, node, p1=None, p2=None):
    """
    Method to calculate the atomic similarity of relationship nodes.
    """

    # For all atomic neighbors, get type to similarity dictionary
    N_ATOMIC = c.N_ATOMIC
    SIM, TYPE1, TYPE2 = c.SIM, c.TYPE1, c.TYPE2

    atomic_sim_dict = {}
    for neighbor_id in self.G.predecessors(node_id):
      if self.G.nodes[neighbor_id][TYPE1] == N_ATOMIC:
        neighbor_node = self.G.nodes[neighbor_id]
        atomic_sim_dict[neighbor_node[TYPE2]] = neighbor_node[SIM]

    # node[c.SIM_ATTR] = sim.atomic_average(atomic_sim_dict)
    node[c.SIM_ATTR] = sim.calculate_weighted_atomic_str_sim(atomic_sim_dict,
                                                             self.attributes)
    node[c.SIM_AMB] = 0.0

  def get_node_people(self, node):
    p1 = self.record_dict[node[c.R1]]
    p2 = self.record_dict[node[c.R2]]
    return p1, p2
