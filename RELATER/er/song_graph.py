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
import pickle
import time
from collections import defaultdict

from common import settings, constants as c, stats, attributes_meta, util, sim
from common.enums import State
from data import songs_data_loader, model
from er.base_graph import BASE_GRAPH


class SONG_GRAPH(BASE_GRAPH):

  def __init__(self, sim_attr_init_func, atomic_t):
    songs_dict = {s[c.I_ID]: s for s in
                  pickle.load(open(settings.songs_file, 'r'))}
    BASE_GRAPH.__init__(self, songs_dict)

    self.attributes = sim_attr_init_func(atomic_t)
    self.unique_record_count = \
      len(filter(lambda s: str(s[c.I_ID]).startswith('10'),
                 self.record_dict.itervalues()))
    self.pairs_file = settings.data_set_dir + 'pairs/1/s7-n50-S-S.csv'

    self.gt_links_dict = songs_data_loader.retrieve_ground_truth_links(
      self.pairs_file)

    # Used to keep track of the linked songs as all links are singletons
    self.linked_ids = set()

  # ---------------------------------------------------------------------------
  # Graph generation
  # ---------------------------------------------------------------------------

  def generate_graph(self, atomic_node_dict):
    """
    Function to generate the graph.
    """

    # Generate atomic nodes
    stime = time.time()
    self.__add_atomic_nodes__(atomic_node_dict, self.attributes)
    stats.a_time = time.time() - stime

    # Generate relationship nodes
    srtime = time.time()
    self.__gen_relationship_nodes__()
    stats.r_time = time.time() - srtime

  def __gen_relationship_nodes__(self):

    logging.info('Started generating relational nodes!')
    attributes = self.attributes
    for r in csv.DictReader(open(self.pairs_file, 'r')):
      r1_id = int(r['table1.{}'.format(c.SG_ID)])
      r2_id = int(r['table2.{}'.format(c.SG_ID)])
      self.__add_single_node__(r1_id, r2_id, 'S-S', attributes)
    logging.info('Finished generating relational nodes!')

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

    # Bootstrapping the graph
    for bootstrap_threshold in bootstrap_threshold_list:
      self.link(bootstrap_threshold)

  def link(self, merge_threshold):
    self.link_gdg(merge_threshold, attributes_meta.songs_sim_func)

  def __is_valid_node_merge__(self, node_id_set):
    nodes = self.G.nodes
    song_dict = self.record_dict
    update_reason = self.__update_reason__

    is_almost_same_years = util.is_almost_same_years

    # Check if any person involved in the group has an invalid link
    for node_id in node_id_set:
      node = self.G.nodes[node_id]

      if node[c.STATE] == State.MERGED:
        continue

      s1 = song_dict[node[c.R1]]
      s2 = song_dict[node[c.R2]]

      # Validate individual constraints
      #
      # Validate individual singletons
      if node[c.R1] in self.linked_ids or node[c.R2] in self.linked_ids:
        # update_reason(node_id_set, nodes, 'RP_S')
        return False

      # Validate song years
      if s1[c.SG_I_YEAR] is not None and s2[c.SG_I_YEAR] is not None:
        if not is_almost_same_years(s1[c.SG_I_YEAR], s2[c.SG_I_YEAR]):
          # update_reason(node_id_set, nodes, 'RP_BY')
          return False

      # Validate entity based constraints
      # This validation is not conducted since all links are singletons

    return True

  def __is_valid_entity_role_merge__(self, o1, o2, node_id_set, people_dict,
                                     relationship_nodes, nodes, merge_t,
                                     node_id_set_queue=None):
    return True

  def __merge_nodes__(self, node_id_set):

    sim_attr_list = self.attributes.attr_index_list

    for node_id in node_id_set:

      node = self.G.nodes[node_id]
      if c.SIM_REL not in node.iterkeys():
        node[c.SIM_REL] = 0.0
      if node[c.STATE] == State.MERGED:
        continue
      node[c.STATE] = State.MERGED

      s1 = self.record_dict[node[c.R1]]
      s2 = self.record_dict[node[c.R2]]

      # Update singletons
      self.linked_ids.add(node[c.R1])
      self.linked_ids.add(node[c.R2])

      # Since all links are singletons, asserting none of the records are
      # associated with entities
      assert s1[c.I_ENTITY_ID] is None and s2[c.I_ENTITY_ID] is None

      self.entity_id += 1

      # Creating an entity and updating the atomic node attributes of the entity
      entity = model.new_entity(self.entity_id, None)
      for i_attr in sim_attr_list:
        entity[i_attr] = util.retrieve_merged(s1[i_attr], s2[i_attr])

      self.entity_dict[self.entity_id] = entity
      s1[c.I_ENTITY_ID] = self.entity_id
      s2[c.I_ENTITY_ID] = self.entity_id

      # Add entity graph
      p_node1 = '%s-%s' % (s1[c.I_ID], s1[c.I_ROLE])
      p_node2 = '%s-%s' % (s2[c.I_ID], s2[c.I_ROLE])
      entity[c.GRAPH].add_edge(p_node1, p_node2, sim=round(node[c.SIM], 2),
                               amb=round(node[c.SIM_AMB], 2),
                               attr=round(node[c.SIM_ATTR], 2))

      entity[c.ROLES] = defaultdict(set)
      entity[c.ROLES][s1[c.I_ROLE]].add(s1[c.I_ID])
      entity[c.ROLES][s2[c.I_ROLE]].add(s2[c.I_ID])


  # ---------------------------------------------------------------------------
  # Ground truth processing
  # ---------------------------------------------------------------------------
  def validate_ground_truth(self, graph_scenario, t):

    # Enumerate processed links
    processed_links_dict = defaultdict(set)
    for e_id, entity in self.entity_dict.iteritems():

      song_id_list = list(entity[c.ROLES]['S'])

      # Only two songs get merged as an entity
      assert len(song_id_list) == 2
      key = tuple(sorted({song_id_list[0], song_id_list[1]}))
      processed_links_dict['S-S'].add(key)

    # Calculate and persist precision and recall values
    self.calculate_persist_measures(processed_links_dict, ['S-S'],
                                    graph_scenario, t, self.record_dict)


  # ---------------------------------------------------------------------------
  # Baselines
  # ---------------------------------------------------------------------------
  def merge_attr_sim_baseline(self, links_list, merging_t):

    nodes = self.G.nodes
    atomic_average = sim.calculate_weighted_atomic_str_sim
    processed_link_dict = defaultdict(set)

    start_time = time.time()
    for node_id, node in nodes(data=True):

      if node[c.TYPE1] == c.N_ATOMIC:
        continue

      # Retrieve atomic node similarities
      atomic_sim_dict = dict()
      for neighbor_id in self.G.predecessors(node_id):
        neighbor_node = nodes[neighbor_id]
        if neighbor_node[c.TYPE1] == c.N_ATOMIC:
          atomic_sim_dict[neighbor_node[c.TYPE2]] = neighbor_node[c.SIM]
      # Calculate average similarity
      atomic_sim = atomic_average(atomic_sim_dict, self.attributes)

      if atomic_sim >= merging_t:
        sorted_key = tuple(sorted({node[c.R1], node[c.R2]}))
        processed_link_dict[node[c.TYPE2]].add(sorted_key)

    t = round(time.time() - start_time, 2)
    self.calculate_persist_measures(processed_link_dict, links_list,
                                    'baseline-attr-sim', t)