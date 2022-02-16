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
from collections import defaultdict, Counter


from data import model
from er import analysis
from er.base_graph import BASE_GRAPH
from common import constants as c, stats, util, settings, attributes_meta, \
  hyperparams
from common.enums import HH_Edge_Type, State, HH_Role, HH_Links


class HH_GRAPH(BASE_GRAPH):

  def __init__(self, sim_attr_init_func, atomic_t, census_year1,
               census_year2):

    people_dict = self.__get_people_dict__(settings.people_file)

    BASE_GRAPH.__init__(self, people_dict)

    self.attributes = sim_attr_init_func(atomic_t)

    self.census_year1 = census_year1

    self.census_year2 = census_year2

    # Used to keep track of the linked people as all links are singletons
    self.linked_ids = set()

    self.unique_record_count = \
      len(filter(lambda p: p[c.I_ROLE] == settings.role_type,
             self.record_dict.itervalues()))

    self.gt_links_dict = pickle.load(open(settings.gt_file, 'rb'))

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

    logging.info('Graph generated successfully')

  def __gen_relationship_nodes__(self):
    """
    Method to generate relationship nodes
    :return:
    """
    logging.info('Relationship node addition')
    records = self.record_dict.values()

    # Filter people with different roles
    mothers_1 = filter(lambda p: p[c.I_ROLE] == 'M' and
                                 p[c.HH_I_YEAR] == self.census_year1, records)
    mothers_2 = filter(lambda p: p[c.I_ROLE] == 'M' and
                                 p[c.HH_I_YEAR] == self.census_year2, records)

    fathers_1 = filter(lambda p: p[c.I_ROLE] == 'F' and
                                 p[c.HH_I_YEAR] == self.census_year1, records)
    fathers_2 = filter(lambda p: p[c.I_ROLE] == 'F' and
                                 p[c.HH_I_YEAR] == self.census_year2, records)

    girls_1 = filter(lambda p: p[c.I_ROLE] == 'C' and p[c.I_SEX] == 'F' and
                               p[c.HH_I_YEAR] == self.census_year1, records)
    girls_2 = filter(lambda p: p[c.I_ROLE] == 'C' and p[c.I_SEX] == 'F' and
                               p[c.HH_I_YEAR] == self.census_year2, records)

    boys_1 = filter(lambda p: p[c.I_ROLE] == 'C' and p[c.I_SEX] == 'M' and
                              p[c.HH_I_YEAR] == self.census_year1, records)
    boys_2 = filter(lambda p: p[c.I_ROLE] == 'C' and p[c.I_SEX] == 'M' and
                              p[c.HH_I_YEAR] == self.census_year2, records)

    is_valid_node = self.__is_valid_node__
    add_single_node = self.__add_single_node__
    attributes = self.attributes

    # This dictionary keeps track of the nodes belonging to each household pair.
    # To be used while adding edges
    hh_pair_node_id_dict = defaultdict(list)

    # Add nodes
    for list1, list2 in [(mothers_1, mothers_2), (fathers_1, fathers_2),
                         (girls_1, girls_2), (boys_1, boys_2)]:
                         # (boys_1, fathers_2), (girls_1, mothers_2)]:
    # for list1, list2 in [(girls_1, girls_2)]:

      logging.info('Lengths of list 1 and list 2 are %s, %s' % (len(list1),
                                                                len(list2)))

      for i in xrange(len(list1)):
        for j in xrange(len(list2)):
          p1 = list1[i]
          p2 = list2[j]

          if p1[c.I_ID] == p2[c.I_ID]:
            continue

          link = '%s-%s' % (p1[c.I_ROLE], p2[c.I_ROLE])
          if is_valid_node(p1, p2):
            node_id = add_single_node(p1, p2, link, attributes)
            if node_id is not None:
              hh_pair_node_id_dict[(p1[c.HH_I_SERIAL],
                                    p2[c.HH_I_SERIAL])].append((node_id, link))

    # Add edges
    for node_id_link_list in hh_pair_node_id_dict.itervalues():

      # Identify mother, father, and child nodes
      mother_node_id = None
      father_node_id = None
      children_node_ids = list()

      for node_id, link in node_id_link_list:
        if link == 'M-M':
          if mother_node_id is None:
            mother_node_id = node_id
          else:
            raise Exception('Two mothers in a household')
        if link == 'F-F':
          if father_node_id is None:
            father_node_id = node_id
          else:
            raise Exception('Two fathers in a household')
        if link == 'C-C':
          children_node_ids.append(node_id)

      # Father, mother edges
      if mother_node_id is not None and father_node_id is not None:
        self.G.add_edge(father_node_id, mother_node_id, t1=c.E_BOOL_W,
                        t2=HH_Edge_Type.SPOUSE)
        self.G.add_edge(mother_node_id, father_node_id, t1=c.E_BOOL_W,
                        t2=HH_Edge_Type.SPOUSE)

      # Mother, father, child edges
      for child_node_id in children_node_ids:

        if father_node_id is not None:
          self.G.add_edge(child_node_id, father_node_id, t1=c.E_BOOL_S,
                          t2=HH_Edge_Type.CHILDREN)
          self.G.add_edge(father_node_id, child_node_id, t1=c.E_BOOL_W,
                          t2=HH_Edge_Type.FATHER)

        if mother_node_id is not None:
          self.G.add_edge(child_node_id, mother_node_id, t1=c.E_BOOL_W,
                          t2=HH_Edge_Type.CHILDREN)
          self.G.add_edge(mother_node_id, child_node_id, t1=c.E_BOOL_W,
                          t2=HH_Edge_Type.MOTHER)

      # Sibling edges
      for i in xrange(len(children_node_ids)):
        for j in xrange(i + 1, len(children_node_ids)):
          c1 = children_node_ids[i]
          c2 = children_node_ids[j]

          self.G.add_edge(c1, c2, t1=c.E_BOOL_W, t2=HH_Edge_Type.SIBLING)
          self.G.add_edge(c2, c1, t1=c.E_BOOL_W, t2=HH_Edge_Type.SIBLING)

  def __is_valid_node__(self, p1, p2):
    if util.is_almost_same_years(p1[c.HH_I_BYEAR], p2[c.HH_I_BYEAR]):
      return True
    return False

  # ---------------------------------------------------------------------------
  # Serialization
  # ---------------------------------------------------------------------------
  def __get_people_dict__(self, people_file):
    """
    Deserializing people
    :param people_file:
    :return:
    """
    people_list = pickle.load(open(people_file, 'rb'))
    people_dict = {}
    for p in people_list:
      people_dict[p[c.I_ID]] = p

    logging.info('People count {}'.format(len(people_dict)))
    return people_dict

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
    for bootsrap_threshold in bootstrap_threshold_list:
      self.link(bootsrap_threshold)

  def link(self, merge_threshold):
    self.link_gdg(merge_threshold, attributes_meta.hh_sim_func)

  def __is_valid_node_merge__(self, node_id_set):

    if hyperparams.scenario.startswith('noprop'):
      return True

    nodes = self.G.nodes
    people_dict = self.record_dict
    update_reason = self.__update_reason__

    is_almost_same_years = util.is_almost_same_years

    # Check if any person involved in the group has an invalid link
    for node_id in node_id_set:
      node = nodes[node_id]

      if node[c.STATE] == State.MERGED:
        continue

      p1 = people_dict[node[c.R1]]
      p2 = people_dict[node[c.R2]]

      # Validate individual constraints
      #
      # Validate individual singletons
      if node[c.R1] in self.linked_ids or node[c.R2] in self.linked_ids:
        # update_reason(node_id_set, nodes, 'RP_S')
        return False

      # Validate individual gender
      gender = util.get_node_gender(p1[c.I_SEX], p2[c.I_SEX])
      if gender == 0:
        # update_reason(node_id_set, nodes, 'RP_G')
        return False

      # Validate birth years
      if p1[c.HH_I_BYEAR] is not None and p2[c.HH_I_BYEAR] is not None:
        if not is_almost_same_years(p1[c.HH_I_BYEAR], p2[c.HH_I_BYEAR]):
          # update_reason(node_id_set, nodes, 'RP_BY')
          return False

      ### Validate entity based constraints
      ##
      ## If both people are merged
      if p1[c.I_ENTITY_ID] is not None and p2[c.I_ENTITY_ID] is not None:

        if p1[c.I_ENTITY_ID] == p2[c.I_ENTITY_ID]:
          continue

        entity1 = self.entity_dict[p1[c.I_ENTITY_ID]]
        entity2 = self.entity_dict[p2[c.I_ENTITY_ID]]

        # Gender constraint
        gender = util.get_node_gender(entity1[c.SEX], entity2[c.SEX])
        if gender == 0:
          # update_reason(node_id_set, nodes, 'RE2_G')
          # Trace of entity evolution for ANALYSIS
          # entity1[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
          #                                            'RE2_G'))
          # entity2[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
          #                                            'RE2_G'))
          return False

        # Temporal constraints
        for e1_birth_year in entity1[c.HH_I_BYEAR]:

          # Birth years should align
          for e2_birth_year in entity2[c.HH_I_BYEAR]:
            if not util.is_almost_same_years(e1_birth_year, e2_birth_year):
              # update_reason(node_id_set, nodes, 'RE2_BY')
              # Trace of entity evolution for ANALYSIS
              # cause = ' {} {}'.format(e1_birth_year, e2_birth_year)
              # entity1[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
              #                                            'RE2_BY', cause))
              # entity2[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
              #                                            'RE2_BY', cause))
              return False

      ## If single person is merged and the other is not
      elif p1[c.I_ENTITY_ID] is not None or p2[c.I_ENTITY_ID] is not None:

        if p1[c.I_ENTITY_ID] is not None:
          entity = self.entity_dict[p1[c.I_ENTITY_ID]]
          non_merged = p2
        else:
          entity = self.entity_dict[p2[c.I_ENTITY_ID]]
          non_merged = p1

        # Gender constraint
        gender = util.get_node_gender(entity[c.SEX], non_merged[c.I_SEX])
        if gender == 0:
          # update_reason(node_id_set, nodes, 'RE1_G')
          # Trace of entity evolution for ANALYSIS
          # entity[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
          #                                           'RE1_G'))
          return False
        entity[c.SEX] = gender

        non_merged_by = non_merged[c.HH_I_BYEAR]
        for e1_birth_year in entity[c.HH_I_BYEAR]:

          if non_merged[c.I_BIRTH_YEAR] is not None:
            if not util.is_almost_same_years(e1_birth_year, non_merged_by):
              # update_reason(node_id_set, nodes, 'RE1_BY')
              # Trace of entity evolution for ANALYSIS
              # cause = ' {} {}'.format(e1_birth_year, non_merged_by)
              # entity[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
              #                                           'RE1_BY', cause))
              return False

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

      p1 = self.record_dict[node[c.R1]]
      p2 = self.record_dict[node[c.R2]]

      # Update singletons
      self.linked_ids.add(node[c.R1])
      self.linked_ids.add(node[c.R2])

      # If both people are not merged
      if p1[c.I_ENTITY_ID] is None and p2[c.I_ENTITY_ID] is None:

        gender = util.get_node_gender(p1[c.I_SEX], p2[c.I_SEX])
        if gender == 0:
          if hyperparams.scenario.startswith('noprop'):
            return
          raise Exception(' Different genders')

        self.entity_id += 1

        # Creating an entity and updating the atomic node attributes of the entity
        entity = model.new_entity(self.entity_id, gender)
        for i_attr in sim_attr_list:
          entity[i_attr] = util.retrieve_merged(p1[i_attr], p2[i_attr])

        # Aggregate the relationship PID s
        entity[c.MOTHER] = util.retrieve_merged(p1[c.HH_I_MOTHER],
                                                p2[c.HH_I_MOTHER])
        entity[c.FATHER] = util.retrieve_merged(p1[c.HH_I_FATHER],
                                                p2[c.HH_I_FATHER])
        entity[c.CHILDREN] = util.retrieve_merged_lists(p1[c.HH_I_CHILDREN],
                                                        p2[c.HH_I_CHILDREN])
        entity[c.SPOUSE] = util.retrieve_merged(p1[c.HH_I_SPOUSE],
                                                p2[c.HH_I_SPOUSE])

        # Each merged person has a dictionary of roles.
        # Each role has a list of tuples where each tuple corresponds to
        # the ID, timestamp and certificate ID corresponding to the person
        # in that role.
        entity[c.ROLES] = defaultdict(set)
        entity[c.ROLES][p1[c.I_ROLE]].add((p1[c.I_ID], p1[c.HH_I_YEAR]))
        entity[c.ROLES][p2[c.I_ROLE]].add((p2[c.I_ID], p2[c.HH_I_YEAR]))

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

          self.linkage_reason_dict[node[c.TYPE2]][(node[c.R1], node[c.R2])] = 'RM_M'
          continue

        entity1 = self.entity_dict[p1[c.I_ENTITY_ID]]
        entity2 = self.entity_dict[p2[c.I_ENTITY_ID]]

        # Delete entity 2 record
        del self.entity_dict[p2[c.I_ENTITY_ID]]
        entity1[c.SEX] = util.get_node_gender(entity1[c.SEX], entity2[c.SEX])
        if entity1[c.SEX] == 0:
          if hyperparams.scenario.startswith('noprop'):
            return
          raise Exception(' Different genders')

        # Aggregate the atomic attribute values
        for attribute in sim_attr_list:
          entity1[attribute] = entity1[attribute].union(entity2[attribute])

        # Aggregate the relationship PID s
        for relationship in HH_Edge_Type.list():
          entity1[relationship] = entity1[relationship].union(
            entity2[relationship])

        for role in HH_Role.list():
          entity1[c.ROLES][role] = \
            entity1[c.ROLES][role].union(entity2[c.ROLES][role])

          # Update entity_id for all people in entity2
          for record in entity2[c.ROLES][role]:
            pid = record[0]
            self.record_dict[pid][c.I_ENTITY_ID] = entity1[c.ENTITY_ID]

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
        entity[c.ROLES][non_merged[c.I_ROLE]].add((non_merged[c.I_ID],
                                                   non_merged[c.HH_I_YEAR]))

        if non_merged[c.HH_I_MOTHER] is not None:
          entity[c.MOTHER].add(non_merged[c.HH_I_MOTHER])
        if non_merged[c.HH_I_FATHER] is not None:
          entity[c.FATHER].add(non_merged[c.HH_I_FATHER])
        if non_merged[c.HH_I_SPOUSE] is not None:
          entity[c.SPOUSE].add(non_merged[c.HH_I_SPOUSE])
        if non_merged[c.HH_I_CHILDREN] is not None:
          for c_id in non_merged[c.HH_I_CHILDREN]:
            entity[c.CHILDREN].add(c_id)

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

  def refine(self, merge_threshold):

    nodes = self.G.nodes
    is_valid_merge = self.__is_valid_node_merge__
    merge_nodes = self.__merge_nodes__
    enrich_node = self.__enrich_node__

    refined_count = 0
    for id, entity in self.entity_dict.items():

      # Refining based on mothers and fathers
      for role in [c.MOTHER, c.FATHER]:

        person_list = list(entity[role])
        for i in xrange(len(person_list)):
          for j in xrange(i + 1, len(person_list)):

            node_id = self.__get_existing_rel_node_id__(person_list[i],
                                                        person_list[j])
            if node_id is not None:
              node = nodes[node_id]
              if node[c.STATE] != State.MERGED:
                enrich_node(node_id, node, attributes_meta.hh_sim_func)
                if is_valid_merge([node_id]):
                  refined_count += 1
                  merge_nodes([node_id])

      # Refining based on spouses and children
      for role in [c.SPOUSE, c.CHILDREN]:

        person_list = list(entity[role])
        for i in xrange(len(person_list)):
          for j in xrange(i + 1, len(person_list)):

            node_id = self.__get_existing_rel_node_id__(person_list[i],
                                                        person_list[j])
            if node_id is not None:
              node = nodes[node_id]
              if node[c.STATE] != State.MERGED:
                enrich_node(node_id, node, attributes_meta.hh_sim_func)
                if node[c.SIM_ATOMIC] >= merge_threshold:
                  if is_valid_merge([node_id]):
                    refined_count += 1
                    merge_nodes([node_id])

    logging.info('Refined merged count {}'.format(refined_count))

  # ---------------------------------------------------------------------------
  # Ground truth processing
  # ---------------------------------------------------------------------------
  def validate_ground_truth(self, graph_scenario, t):

    # Enumerate processed links
    processed_links_dict = defaultdict(set)
    for e_id, entity in self.entity_dict.iteritems():
      mother_list = list(entity[c.ROLES]['M'])
      father_list = list(entity[c.ROLES]['F'])
      children_list = list(entity[c.ROLES]['C'])

      # Adding homogeneous links
      for list1, link in [(mother_list, 'M-M'), (father_list, 'F-F'),
                          (children_list, 'C-C')]:
        for i in xrange(len(list1)):
          for j in xrange(i + 1, len(list1)):
            id1, yr1 = list1[i]
            id2, yr2 = list1[j]

            key = tuple(sorted({id1, id2}))
            processed_links_dict[link].add(key)

      # Adding different links
      for list1, list2, link in [(children_list, mother_list, 'C-M'),
                                 (children_list, father_list, 'C-F')]:
        for i in xrange(len(list1)):
          for j in xrange(len(list2)):

            id1, yr1 = list1[i]
            id2, yr2 = list2[j]
            key = tuple(sorted({id1, id2}))
            processed_links_dict[link].add(key)

    # Calculate and persist precision and recall values
    self.calculate_persist_measures(processed_links_dict, HH_Links.list(),
                                    graph_scenario, t, self.record_dict)

  def analyse_fp_fn(self, links_list, false_negative_link_dict,
                    false_positive_link_dict, records, secondary_records=None):
    # Analyse false negative reasons
    logging.info('FALSE NEGATIVES ----')
    reason_dict = self.linkage_reason_dict
    fn_reasons_dict = defaultdict(Counter)

    for link in links_list:
      for id_1, id_2 in false_negative_link_dict[link]:

        if (id_1, id_2) in reason_dict[link]:
          if reason_dict[link][(id_1, id_2)] in ['LSIM', 'SIN']:

            logging.info('{} {} - {}'.format(reason_dict[link][(id_1, id_2)],
                                            id_1, id_2))
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
          fn_reasons_dict[link].update([reason_dict[link][(id_1, id_2)]])
        elif (id_2, id_1) in reason_dict[link]:
          if reason_dict[link][(id_2, id_1)] in ['LSIM', 'SIN']:
            logging.info('{} {} - {}'.format(reason_dict[link][(id_2, id_1)],
                         id_1, id_2))
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
        else:
          assert (id_1, id_2) not in self.relationship_nodes and \
                 (id_2, id_1) not in self.relationship_nodes

    logging.info('FALSE POSITIVES ----')
    for link in links_list:
      for id_1, id_2 in false_positive_link_dict[link]:
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


    file_name = '{}fn-reasons.csv'.format(settings.results_dir)
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode='a') as f:
      w = csv.writer(f)
      if not file_exists:
        w.writerow(['link', 'scenario', 'total'] + c.LINKAGE_REASONS_LIST)
      scenario = settings.scenario
      for link, reason_counter in fn_reasons_dict.iteritems():
        row = [link, scenario, len(fn_reasons_dict[link])]
        for reason in c.LINKAGE_REASONS_LIST:
          if reason in reason_counter.iterkeys():
            row.append(reason_counter.get(reason))
          else:
            row.append(0)
        w.writerow(row)
