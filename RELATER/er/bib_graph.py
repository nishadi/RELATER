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
from collections import deque, defaultdict, Counter

from data import model, bib_data_loader
from er import analysis
from er.base_graph import BASE_GRAPH
from common.enums import Bib_Edge_Type, State
from common import constants as c, stats, sim, attributes_meta, util, settings, \
  hyperparams


class TYPE_META:

  def __init__(self, role, attributes, record_filename):
    self.role = role
    self.attributes = attributes
    self.record_dict = pickle.load(open(record_filename, 'rb'))
    self.record_count = len(self.record_dict)


class BIB_GRAPH(BASE_GRAPH):

  def __init__(self, type_meta_dict, dataset1_abbreviation,
               dataset2_abbreviation):
    BASE_GRAPH.__init__(self)

    # Dictionary having a mapping of each entity type to entity meta
    # information. Entity meta information should be of the `TYPE_META` class.
    self.type_meta_dict = type_meta_dict

    self.dataset1_abbreviation = dataset1_abbreviation
    self.dataset2_abbreviation = dataset2_abbreviation

    # todo:remove
    self.gt_links_dict = bib_data_loader.enumerate_ground_truthlinks(
      dataset1_abbreviation, dataset2_abbreviation)

  def generate_graph(self, atomic_node_dict):
    """
    Function to generate the graph.
    """

    # Generate atomic nodes based on authors
    stime = time.time()
    self.__add_atomic_nodes__(atomic_node_dict,
                              self.type_meta_dict['A'].attributes)
    stats.a_time = time.time() - stime

    # Generate relationship nodes
    srtime = time.time()
    self.__gen_relationship_nodes__()
    stats.r_time = time.time() - srtime

  def __gen_relationship_nodes__(self):

    record_dict1 = self.type_meta_dict['P'].record_dict
    record_list1 = filter(lambda a: a[0].startswith(self.dataset1_abbreviation),
                          record_dict1.values())
    record_list2 = filter(lambda a: a[0].startswith(self.dataset2_abbreviation),
                          record_dict1.values())

    p_attributes = self.type_meta_dict['P'].attributes
    a_attributes = self.type_meta_dict['A'].attributes

    E_BOOL_W = c.E_BOOL_W
    for r1 in record_list1:
      for r2 in record_list2:

        # Add primary node, a publication pair
        primary_node_id = self.__add_single_node__(r1, r2, 'P-P', p_attributes)

        # If the primary node exists, add secondary nodes, author pairs
        if primary_node_id is not None:

          rec_dict = self.type_meta_dict['A'].record_dict

          ds1_ref_ids = r1[c.BB_I_AUTHOR_ID_LIST]
          ds2_ref_ids = r2[c.BB_I_AUTHOR_ID_LIST]

          ref_node_id_list = list()
          for ref_id1 in ds1_ref_ids:
            for ref_id2 in ds2_ref_ids:

              rec1 = rec_dict[ref_id1]
              rec2 = rec_dict[ref_id2]
              ref_node_id = self.__add_single_node__(rec1, rec2, 'A-A',
                                                     a_attributes)
              if ref_node_id is not None:
                ref_node_id_list.append(ref_node_id)

          for i in xrange(len(ref_node_id_list)):

            # Add relationships with primary node
            self.G.add_edge(ref_node_id_list[i], primary_node_id, t1=E_BOOL_W,
                            t2=Bib_Edge_Type.PUBLICATION)
            self.G.add_edge(primary_node_id, ref_node_id_list[i], t1=E_BOOL_W,
                            t2=Bib_Edge_Type.AUTHOR)

            # Add relationships among the secondary nodes
            for j in xrange(i + 1, len(ref_node_id_list)):
              self.G.add_edge(ref_node_id_list[i], ref_node_id_list[j],
                              t1=E_BOOL_W, t2=Bib_Edge_Type.COAUTHOR)
              self.G.add_edge(ref_node_id_list[j], ref_node_id_list[i],
                              t1=E_BOOL_W, t2=Bib_Edge_Type.COAUTHOR)

  def __add_single_node__(self, p1, p2, relationship_type, attributes):
    """
    Function to add a single node to the graph including the
    edges to atomic nodes.

    :param p1: Person 1 or ID of Person 1
    :param p2: person 2 or ID of Person 2
    :param relationship_type: Type of node
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
    incoming_atomic_nodes = list()
    for attr_index in attributes.attr_index_list:

      # To account for shortened names in authors
      key1, key2 = p1[attr_index], p2[attr_index]
      # if key1 is not None and key2 is not None:
      #   if len(key1) == 2 and key1.endswith('.'):
      #     key2 = key2[0] + '.'
      #   if len(key2) == 2 and key2.endswith('.'):
      #     key1 = key1[0] + '.'
      if key1 is not None and key2 is not None:
        if type(key1) == str and len(key1) == 1:
          key2 = key2[0]
        if type(key2) == str and len(key2) == 1:
          key1 = key1[0]

      if (key1, key2) in self.atomic_nodes[attr_index]:
        incoming_atomic_nodes.append(
          (attr_index,
           self.atomic_nodes[attr_index][(key1, key2)]))

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
  # Deserialize
  # ---------------------------------------------------------------------------
  def deserialize_graph(self, graph_file):
    BASE_GRAPH.deserialize_graph(self, graph_file)

    # Generating relationship nodes based on node type
    self.relationship_nodes = defaultdict(dict)
    for node_id, node in self.G.nodes(data=True):
      key1 = (node[c.R1], node[c.R2])
      key2 = (node[c.R2], node[c.R1])

      if node[c.TYPE1] == c.N_RELTV:
        node_type = node[c.TYPE2].split('-')[0]
        self.relationship_nodes[node_type][key1] = node_id
        self.relationship_nodes[node_type][key2] = node_id

  # ---------------------------------------------------------------------------
  # Graph processing
  # ---------------------------------------------------------------------------
  def bootstrap(self, bootstrap_threshold_list):

    # Calculate node atomic similarity
    calculate_sim_atomic = self.calculate_sim_atomic
    for node_id, node in self.G.nodes(data=True):
      if node[c.TYPE1] == c.N_ATOMIC:
        continue
      calculate_sim_atomic(node_id, node)

    # Enumerate graph groups
    self.__enumerate_graph_groups_with_avg_similarity__()

    # # Bootstrapping the graph
    for bootstrap_threshold in bootstrap_threshold_list:
      self.link(bootstrap_threshold)

  def calculate_sim_atomic(self, node_id, node, p1=None, p2=None):
    """
    Method to calculate the atomic similarity of relationship nodes.
    """

    type_meta = self.type_meta_dict[node[c.TYPE2].split('-')[0]]
    if p1 is None:
      p1 = type_meta.record_dict[node[c.R1]]
    if p2 is None:
      p2 = type_meta.record_dict[node[c.R2]]

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
                                    type_meta.record_count,
                                    type_meta.attributes)

  def link_gdg(self, merge_threshold, attr_sim_func_dict):

    logging.debug('MERGING STEP')

    # Optimizations
    nodes = self.G.nodes
    relationship_nodes = self.relationship_nodes
    people_dict = self.record_dict
    inactive, nonmerge = State.INACTIVE, State.NONMERGE

    enrich_node = self.__enrich_node__
    is_valid_entity_role_merge = self.__is_valid_entity_role_merge__
    is_valid_node_merge = self.__is_valid_node_merge__
    merge_nodes = self.__merge_nodes__
    __update_reason__ = self.__update_reason__

    # Initializing the nodes queue
    node_id_set_queue = deque()
    for group_size in sorted(self.group_dict.iterkeys(), reverse=True):

      if group_size == 1:  # todo later : poor fellows
        # for node_id_set, sim, _ in self.group_dict[1]:
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

    # Processing the nodes queue
    while node_id_set_queue:

      # Get the node at the front of the queue
      init_node_id_set = node_id_set_queue.popleft()

      # Single nodes can be added in adding non-existing pairs.
      # Therefore this needs to be checked.
      if len(init_node_id_set) == 1:
        continue

      node_id_sim_list = list()

      # Enrich the set of nodes and add them to a list with tuples of
      # (node_id, sim)
      for node_id in init_node_id_set:
        node = nodes[node_id]
        assert node[c.SIM_ATOMIC] != 0

        if node[c.TYPE2] == 'P-P':
          attr_sim_func_dict = attributes_meta.pubs_sim_func
        elif node[c.TYPE2] == 'A-A':
          attr_sim_func_dict = attributes_meta.authors_sim_func
        else:
          raise Exception('Wrong node type')

        o1, o2 = enrich_node(node_id, node, attr_sim_func_dict)

        if is_valid_entity_role_merge(o1, o2, init_node_id_set, people_dict,
                                      relationship_nodes, nodes,
                                      merge_threshold, node_id_set_queue) and \
            is_valid_node_merge({node_id}):
          node[c.STATE] = inactive
          node_id_sim_list.append((node_id, node[c.SIM_ATOMIC]))
        else:
          node[c.STATE] = nonmerge

      if len(node_id_sim_list) == 0:
        continue

      # Sort the node list based on the similarity
      sorted_sim_list = sorted(node_id_sim_list, key=lambda x: x[1],
                               reverse=True)

      # Try to merge them until the group reduced to a pair,
      # by removing the node with min sim in each iteration.
      for n in xrange(len(sorted_sim_list), 1, -1):
        # for n in [len(sorted_sim_list)]:
        sorted_node_id_sim = sorted_sim_list[0:n]
        node_id_set = map(lambda x: x[0], sorted_node_id_sim)

        sim_list = map(lambda x: x[1], sorted_node_id_sim)
        sim = sum(sim_list) * 1.0 / len(sim_list)

        if sim >= merge_threshold:
          if is_valid_node_merge(node_id_set):
            merge_nodes(node_id_set)
            if init_node_id_set in stats.added_node_id_set:
              stats.added_merged_node_id_count += 1
            break

        # # Analysis
        # if n == 2:
        #   pass
        #   # __update_reason__(node_id_set, nodes, 'LSIM')
        # else:
        #   ls_node_id = sorted_sim_list[(n - 1)][0]
        #   ls_node = nodes[ls_node_id]
        #   self.linkage_reason_dict[ls_node[c.TYPE2]][
        #     (ls_node[c.R1], ls_node[c.R2])] = 'LSIM'
        #   ls_node[c.STATE] = State.INACTIVE

    logging.info('Enriched node count %s' % (stats.atomic_enriched))
    logging.info('Added neighbour count %s' % (stats.added_neighbors_count))
    logging.info('Activated count %s' % (stats.activated_neighbors_count))

    print 'Merge %s added node count %s' % (
      merge_threshold, stats.added_neighbors_count)
    if len(stats.added_group_size) > 0:
      print 'Merge %s added node group avg %s' % (merge_threshold, (
          sum(stats.added_group_size) * 1.0 / len(stats.added_group_size)))
      print 'Merge %s added node group max %s' % (
        merge_threshold, max(stats.added_group_size))
      print 'Merge %s added node group merged count %s' % (
        merge_threshold, stats.added_merged_node_id_count)
      p = stats.added_merged_node_id_count * 100.0 / len(
        stats.added_node_id_set)
      print 'Merge %s added node group merged percentage %s%%' % (
        merge_threshold, p)

  def merge_singletons(self, merge_threshold):

    logging.debug('MERGING SINGLETONS STEP')
    is_valid_node_merge = self.__is_valid_node_merge__
    is_valid_entity_role_merge = self.__is_valid_entity_role_merge__
    merge_nodes = self.__merge_nodes__

    # Optimizations
    nodes = self.G.nodes
    relationship_nodes = self.relationship_nodes
    people_dict = self.record_dict

    enrich_node = self.__enrich_node__
    __update_reason__ = self.__update_reason__

    # Initializing the nodes queue
    node_id_set_queue = deque()
    for node_id_set, sim in self.group_dict[1]:
      if sim >= merge_threshold:
        node_id_set_queue.append(node_id_set)

    # Processing the nodes queue
    while node_id_set_queue:

      # Get the node at the front of the queue
      init_node_id_set = node_id_set_queue.popleft()
      node_id = next(iter(init_node_id_set))
      node = nodes[node_id]

      if node[c.TYPE2] == 'P-P':
        attr_sim_func_dict = attributes_meta.pubs_sim_func
      elif node[c.TYPE2] == 'A-A':
        attr_sim_func_dict = attributes_meta.authors_sim_func
      else:
        raise Exception('Wrong node type')

      o1, o2 = enrich_node(node_id, node, attr_sim_func_dict)

      if is_valid_entity_role_merge(o1, o2, init_node_id_set, people_dict,
                                    relationship_nodes, nodes,
                                    merge_threshold, node_id_set_queue):
        if is_valid_node_merge(init_node_id_set):
          merge_nodes(init_node_id_set)

  def link(self, merge_threshold):
    self.link_gdg(merge_threshold, None)
    sin_t = 0.95 if c.data_set == 'dblp-scholar1' else 0.9
    self.merge_singletons(sin_t)

  def __update_reason__(self, node_id_set, nodes, reason):

    # Updating reasons
    for tmp_node_id in node_id_set:
      tmp_node = nodes[tmp_node_id]
      self.linkage_reason_dict[tmp_node[c.TYPE2]][
        (tmp_node[c.R1], tmp_node[c.R2])] = reason

  def __is_valid_node_merge__(self, node_id_set):

    # DBLP-Scholar1 data set is not 1:1 mapping.
    # One entry from one data set can have multiple matches
    # from the other data set.
    if c.data_set == 'dblp-scholar1':
      nodes = self.G.nodes
      for node_id in node_id_set:
        node = nodes[node_id]
        type = node[c.TYPE2].split('-')[0]

        r1 = self.type_meta_dict[type].record_dict[node[c.R1]]
        r2 = self.type_meta_dict[type].record_dict[node[c.R2]]

        # if node[c.TYPE2] == 'P-P':
        #   author_diff = abs(
        #     len(r1[c.BB_I_AUTHOR_ID_LIST]) - len(r2[c.BB_I_AUTHOR_ID_LIST]))
        #   if author_diff > 2:
        #     return False

        if not util.is_almost_same_years(r1[c.BB_I_YEAR], r2[c.BB_I_YEAR]):
          return False
      return True
    else:
      nodes = self.G.nodes
      for node_id in node_id_set:
        node = nodes[node_id]
        type = node[c.TYPE2].split('-')[0]

        r1 = self.type_meta_dict[type].record_dict[node[c.R1]]
        r2 = self.type_meta_dict[type].record_dict[node[c.R2]]

        if type == 'P':
          if r1[c.I_ENTITY_ID] is not None or r2[c.I_ENTITY_ID] is not None:
            # self.__update_reason__(node_id_set, nodes, 'P')
            return False
      return True

  def __is_valid_entity_role_merge__(self, o1, o2, node_id_set, people_dict,
                                     relationship_nodes, nodes, merge_t,
                                     node_id_set_queue=None):
    # If both o1 and o2 are entities
    if type(o1) == dict and type(o2) == dict:
      for y1 in o1[c.BB_I_YEAR]:
        for y2 in o2[c.BB_I_YEAR]:
          if y1 is not None and y2 is not None and y1 != y2:
            return False
    elif type(o1) == dict:
      for y1 in o1[c.BB_I_YEAR]:
        if y1 is not None and o2[c.BB_I_YEAR] is not None and y1 != o2[
          c.BB_I_YEAR]:
          return False
    elif type(o2) == dict:
      for y2 in o2[c.BB_I_YEAR]:
        if o1[c.BB_I_YEAR] is not None and y2 is not None and o1[
          c.BB_I_YEAR] != y2:
          return False
    return True

  def __merge_nodes__(self, node_id_set):

    for node_id in node_id_set:

      node = self.G.nodes[node_id]
      if c.SIM_REL not in node.iterkeys():
        node[c.SIM_REL] = 0.0
      if node[c.STATE] == State.MERGED:
        continue
      node[c.STATE] = State.MERGED

      node_type = node[c.TYPE2].split('-')[0]
      sim_attr_list = self.type_meta_dict[node_type].attributes.attr_index_list
      record_dict = self.type_meta_dict[node_type].record_dict

      p1 = record_dict[node[c.R1]]
      p2 = record_dict[node[c.R2]]

      # If both records are not merged
      if p1[c.I_ENTITY_ID] is None and p2[c.I_ENTITY_ID] is None:

        self.entity_id += 1

        # Creating an entity and updating the atomic node attributes of the entity
        entity = model.new_entity(self.entity_id, None)
        entity['type'] = node_type
        for i_attr in sim_attr_list:
          entity[i_attr] = util.retrieve_merged(p1[i_attr], p2[i_attr])

        # Aggregate the relationship PID s
        if node_type == 'P':
          entity[c.AUTHOR] = \
            util.retrieve_merged_lists(p1[c.BB_I_AUTHOR_ID_LIST],
                                       p2[c.BB_I_AUTHOR_ID_LIST])
        else:
          entity[c.AUTHOR] = \
            util.retrieve_merged_lists(p1[c.BB_I_COAUTHOR_LIST],
                                       p2[c.BB_I_COAUTHOR_LIST])

        # Each merged person has a dictionary of roles.
        # Each role has a list of tuples where each tuple corresponds to
        # the ID, timestamp and certificate ID corresponding to the person
        # in that role.
        entity[c.ROLES] = defaultdict(set)
        entity[c.ROLES][p1[c.I_ROLE]].add(p1[c.I_ID])
        entity[c.ROLES][p2[c.I_ROLE]].add(p2[c.I_ID])

        # Trace of entity evolution for ANALYSIS
        entity[c.TRACE].append(analysis.get_trace(node, p1, p2, True, 'RM_E'))

        self.entity_dict[self.entity_id] = entity
        p1[c.I_ENTITY_ID] = self.entity_id
        p2[c.I_ENTITY_ID] = self.entity_id

        # Add entity graph
        p_node1 = '%s-%s' % (p1[c.I_ID], p1[c.I_ROLE])
        p_node2 = '%s-%s' % (p2[c.I_ID], p2[c.I_ROLE])
        entity[c.GRAPH].add_edge(p_node1, p_node2, sim=round(node[c.SIM], 2),
                                 amb=round(node[c.SIM_AMB], 2),
                                 attr=round(node[c.SIM_ATTR], 2))

      # If both people are merged
      elif p1[c.I_ENTITY_ID] is not None and p2[c.I_ENTITY_ID] is not None:

        # Already merged
        if p1[c.I_ENTITY_ID] == p2[c.I_ENTITY_ID]:
          # Add edge to entity graph
          p_node1 = '%s-%s' % (p1[c.I_ID], p1[c.I_ROLE])
          p_node2 = '%s-%s' % (p2[c.I_ID], p2[c.I_ROLE])
          entity = self.entity_dict[p1[c.I_ENTITY_ID]]
          entity[c.GRAPH].add_edge(p_node1, p_node2, sim=round(node[c.SIM], 2),
                                   amb=round(node[c.SIM_AMB], 2),
                                   attr=round(node[c.SIM_ATTR], 2))

          self.linkage_reason_dict[node[c.TYPE2]][
            (node[c.R1], node[c.R2])] = 'RM_M'
          continue

        entity1 = self.entity_dict[p1[c.I_ENTITY_ID]]
        entity2 = self.entity_dict[p2[c.I_ENTITY_ID]]

        # Delete entity 2 record
        del self.entity_dict[p2[c.I_ENTITY_ID]]

        # Aggregate the atomic attribute values
        for attribute in sim_attr_list:
          entity1[attribute] = entity1[attribute].union(entity2[attribute])

        # Aggregate the relationship PID s
        if node_type == 'P':
          entity1[c.AUTHOR] = \
            util.retrieve_merged_lists(entity1[c.AUTHOR],
                                       entity2[c.AUTHOR])
        else:
          entity1[c.AUTHOR] = \
            util.retrieve_merged_lists(entity1[c.AUTHOR],
                                       entity2[c.AUTHOR])

        entity1[c.ROLES][node_type] = entity1[c.ROLES][node_type].union(
          entity2[c.ROLES][node_type])
        for id in entity2[c.ROLES][node_type]:
          self.type_meta_dict[node_type].record_dict[id][c.I_ENTITY_ID] = \
            entity1[c.ENTITY_ID]

        # Trace of entity evolution for ANALYSIS
        entity1[c.TRACE].append(analysis.get_trace(node, p1, p2, True, 'RM_EE'))
        entity1[c.TRACE].append(analysis.get_trace(node, p1, p2, True, 'RM_EE'))
        entity1[c.TRACE].append(entity2[c.TRACE])

        # Merge entity graphs
        for edge_node1, edge_node2, edge_data in entity2[c.GRAPH].edges.data():
          entity1[c.GRAPH].add_edge(edge_node1, edge_node2, **edge_data)
        p_node1 = '%s-%s' % (p1[c.I_ID], p1[c.I_ROLE])
        p_node2 = '%s-%s' % (p2[c.I_ID], p2[c.I_ROLE])
        entity1[c.GRAPH].add_edge(p_node1, p_node2, sim=round(node[c.SIM], 2),
                                  amb=round(node[c.SIM_AMB], 2),
                                  attr=round(node[c.SIM_ATTR], 2))

      ## If only one person is merged while the other is not
      else:
        if p1[c.I_ENTITY_ID] is not None:
          entity = self.entity_dict[p1[c.I_ENTITY_ID]]
          non_merged = p2
        else:
          entity = self.entity_dict[p2[c.I_ENTITY_ID]]
          non_merged = p1

        entity[c.SEX] = util.get_node_gender(entity[c.SEX], non_merged[c.I_SEX])
        if entity[c.SEX] == 0:
          raise Exception(' Different genders')

        # Aggregate the atomic attribute values
        for i_attribute in sim_attr_list:
          if non_merged[i_attribute] is not None:
            entity[i_attribute].add(non_merged[i_attribute])

        entity[c.ROLES][non_merged[c.I_ROLE]].add(non_merged[c.I_ID])

        # Aggregate the relationship PID s
        if node_type == 'P':
          entity[c.AUTHOR] = \
            util.retrieve_merged_lists(entity[c.AUTHOR],
                                       non_merged[c.BB_I_AUTHOR_ID_LIST])
        else:
          entity[c.AUTHOR] = \
            util.retrieve_merged_lists(entity[c.AUTHOR],
                                       non_merged[c.BB_I_COAUTHOR_LIST])

        # Trace of entity evolution for ANALYSIS
        entity[c.TRACE].append(analysis.get_trace(node, p1, p2, True, 'RM_EP'))

        p1[c.I_ENTITY_ID] = entity[c.ENTITY_ID]
        p2[c.I_ENTITY_ID] = entity[c.ENTITY_ID]

        # Add edge into entity graph
        p_node1 = '%s-%s' % (p1[c.I_ID], p1[c.I_ROLE])
        p_node2 = '%s-%s' % (p2[c.I_ID], p2[c.I_ROLE])
        entity[c.GRAPH].add_edge(p_node1, p_node2, sim=round(node[c.SIM], 2),
                                 amb=round(node[c.SIM_AMB], 2),
                                 attr=round(node[c.SIM_ATTR], 2))

  def __enrich_node__(self, node_id, node, attr_sim_func_dict):
    atomic_enriched = False

    # Get the person records associated with the node
    node_type = node[c.TYPE2].split('-')[0]
    record_dict = self.type_meta_dict[node_type].record_dict
    p1 = record_dict[node[c.R1]]
    p2 = record_dict[node[c.R2]]

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

      if c.TYPE1 in edge_data and edge_data[c.TYPE1] == c.E_REAL:
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
      if p1[c.I_ENTITY_ID] == p2[c.I_ENTITY_ID]:
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

  # ---------------------------------------------------------------------------
  # Ground truth processing
  # ---------------------------------------------------------------------------
  def validate_ground_truth(self, graph_scenario, t):

    # Enumerate processed links
    processed_links_dict = defaultdict(set)
    for e_id, entity in self.entity_dict.iteritems():

      if entity['type'] == 'P':
        p_list = list(entity[c.ROLES]['P'])
        for i in xrange(len(p_list)):
          for j in xrange(i + 1, len(p_list)):
            key1, key2 = p_list[i], p_list[j]

            # Consider publications from two data sets
            #
            if len({key1.split('-')[0], key2.split('-')[0]}) == 2:
              key = tuple(sorted({key1, key2}))
              processed_links_dict['P-P'].add(key)

    # Calculate and persist precision and recall values
    self.calculate_persist_measures(processed_links_dict, ['P-P'],
                                    graph_scenario, t, self.type_meta_dict[
                                      'P'].record_dict, self.type_meta_dict[
                                      'A'].record_dict)

  def analyse_fp_fn(self, links_list, false_negative_link_dict,
                    false_positive_link_dict, records, authors=None):
    # Analyse false negative reasons
    print 'FALSE NEGATIVES ----'
    reason_dict = self.linkage_reason_dict
    fn_reasons_dict = defaultdict(Counter)
    for link in links_list:
      for id_1, id_2 in false_negative_link_dict[link]:

        if (id_1, id_2) in reason_dict[link]:
          if reason_dict[link][(id_1, id_2)] in ['LSIM', 'SIN']:
            print reason_dict[link][(id_1, id_2)], id_1, id_2
            print records[id_1]
            for a_id in records[id_1][c.BB_I_AUTHOR_ID_LIST]:
              print '\t {} - {}'.format(a_id, authors[a_id][c.BB_I_FULLNAME])
            print records[id_2]
            for a_id in records[id_2][c.BB_I_AUTHOR_ID_LIST]:
              print '\t {} - {}'.format(a_id, authors[a_id][c.BB_I_FULLNAME])
          fn_reasons_dict[link].update([reason_dict[link][(id_1, id_2)]])
        elif (id_2, id_1) in reason_dict[link]:
          if reason_dict[link][(id_2, id_1)] in ['LSIM', 'SIN']:
            print reason_dict[link][(id_2, id_1)], id_1, id_2
            print records[id_1]
            for a_id in records[id_1][c.BB_I_AUTHOR_ID_LIST]:
              print '\t {} - {}'.format(a_id, authors[a_id][c.BB_I_FULLNAME])
            print records[id_2]
            for a_id in records[id_2][c.BB_I_AUTHOR_ID_LIST]:
              print '\t {} - {}'.format(a_id, authors[a_id][c.BB_I_FULLNAME])
          fn_reasons_dict[link].update([reason_dict[link][(id_2, id_1)]])
        else:
          assert (id_1, id_2) not in self.relationship_nodes and \
                 (id_2, id_1) not in self.relationship_nodes

        # Analyse false positive reasons

    print 'FALSE POSITIVES ----'
    for link in links_list:
      for id_1, id_2 in false_positive_link_dict[link]:
        print id_1, id_2
        print records[id_1]
        for a_id in records[id_1][c.BB_I_AUTHOR_ID_LIST]:
          print '\t {} - {}'.format(a_id,
                                    authors[a_id][c.BB_I_FULLNAME])
        print records[id_2]
        for a_id in records[id_2][c.BB_I_AUTHOR_ID_LIST]:
          print '\t {} - {}'.format(a_id,
                                    authors[a_id][c.BB_I_FULLNAME])

    file_name = '{}fn-reasons.csv'.format(settings.results_dir)
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode='a') as f:
      w = csv.writer(f)
      if not file_exists:
        w.writerow(['link', 'scenario', 'total'] + c.LINKAGE_REASONS_LIST)
      scenario = hyperparams.scenario
      for link, reason_counter in fn_reasons_dict.iteritems():
        row = [link, scenario, len(fn_reasons_dict[link])]
        for reason in c.LINKAGE_REASONS_LIST:
          if reason in reason_counter.iterkeys():
            row.append(reason_counter.get(reason))
          else:
            row.append(0)
        w.writerow(row)

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

        if c.SIM_ATTR not in node.iterkeys():
          calculate_sim_atomic(node_id, node)

        attr_sim_func_dict = attributes_meta.authors_sim_func \
          if node[c.TYPE2] == 'A-A' else attributes_meta.pubs_sim_func

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
      neighbor_node = self.G.nodes[edge[0]]
      if neighbor_node[c.STATE] == State.INACTIVE:
        stats.activated_neighbors_count += 1
        neighbor_node[c.STATE] = State.ACTIVE
        nodes_queue.append((edge[0], neighbor_node))

  def __depgraph_calculate_sim__(self, node_id, node):
    """
    Method to calculate the atomic similarity of relationship nodes.
    """
    gamma = 0.05

    bool_weak_neighbor_count = 0
    bool_weak_neighbor_merged_count = 0

    # Iterate node edges
    for edge in self.G.in_edges(node_id, data=True):
      edge_data = edge[2]

      if edge_data[c.TYPE1] == c.E_BOOL_W:
        rel_node = self.G.nodes[edge[0]]
        bool_weak_neighbor_count += 1
        if rel_node[c.STATE] == State.MERGED:
          bool_weak_neighbor_merged_count += 1

    atomic_sim = node[c.SIM_ATTR]
    node[c.SIM] = atomic_sim * 0.90 + \
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

    node_type = node[c.TYPE2].split('-')[0]
    attributes = self.type_meta_dict[node_type].attributes

    # node[c.SIM_ATTR] = sim.atomic_average(atomic_sim_dict)
    node[c.SIM_ATTR] = sim.calculate_weighted_atomic_str_sim(atomic_sim_dict,
                                                             attributes)
    node[c.SIM_AMB] = 0.0

  def merge_attr_sim_baseline(self, links_list, merging_t):

    nodes = self.G.nodes
    atomic_average = sim.calculate_weighted_atomic_str_sim
    # atomic_average = sim.atomic_average
    processed_link_dict = defaultdict(set)

    start_time = time.time()
    for node_id, node in nodes(data=True):

      if node[c.TYPE1] == c.N_ATOMIC or node[c.TYPE2] == 'A-A':
        continue

      atomic_sim_dict = dict()
      for neighbor_id in self.G.predecessors(node_id):
        neighbor_node = nodes[neighbor_id]

        # Retrieve atomic node similarities
        if neighbor_node[c.TYPE1] == c.N_ATOMIC:
          atomic_sim_dict[neighbor_node[c.TYPE2]] = neighbor_node[c.SIM]

        # Retrieve authors full name similarities
        if neighbor_node[c.TYPE2] == 'A-A':
          for edge in self.G.in_edges(neighbor_id, data=True):
            if edge[2][c.TYPE1] == c.E_REAL and edge[2][c.TYPE2] in [
              c.BB_I_SNAME, c.BB_I_FNAME]:
              # c.BB_I_FULLNAME:
              # atomic_sim_dict['A' + str(neighbor_id)] = nodes[edge[0]][c.SIM]
              atomic_sim_dict[edge[2][c.TYPE2]] = nodes[edge[0]][c.SIM]

      attributes = attributes_meta.def_authors(0.9)
      atomic_sim = atomic_average(atomic_sim_dict, attributes)
      # atomic_sim = atomic_average(atomic_sim_dict)

      if atomic_sim >= merging_t:
        sorted_key = tuple(sorted({node[c.R1], node[c.R2]}))
        processed_link_dict[node[c.TYPE2]].add(sorted_key)

    t = round(time.time() - start_time, 2)
    self.calculate_persist_measures(processed_link_dict, links_list,
                                    'baseline-attr-sim', t)
