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
from collections import defaultdict, Counter

import csv
import networkx as nx

from data import model, data_loader
from er import analysis
from febrl import encode
from er.base_graph import BASE_GRAPH
from common import constants as c, stats, util, settings, attributes_meta, sim, \
  hyperparams
from common.enums import Sex, Role, Edge_Type, Links, State


class CIVIL_GRAPH(BASE_GRAPH):

  def __init__(self, sim_attr_init_func, atomic_t):

    people_dict = self.__get_people_dict__(settings.people_file)
    BASE_GRAPH.__init__(self, people_dict)

    self.attributes = sim_attr_init_func(atomic_t)
    self.unique_record_count = \
      len(filter(lambda p: p[c.I_ROLE] == settings.role_type,
                 self.record_dict.itervalues()))

    self.singleton_links_dict = {}
    for r in Links.get_singleton_links():
      self.singleton_links_dict[r] = set()

  # ---------------------------------------------------------------------------
  # Graph generation
  # ---------------------------------------------------------------------------

  def deserialize_graph(self, graph_file):
    BASE_GRAPH.deserialize_graph(self, graph_file)
    self.gt_links_dict = data_loader.retrieve_ground_truth_links(self)

    ## For IoS data set, we need to add geocoding based atomic nodes
    if c.data_set != 'ios':
      return

    # Retrieve geocodes
    geocode_dict = dict()
    for r in csv.DictReader(
        open(settings.output_home_directory + 'geocodes.csv', 'r')):
      if r['geocode'] != '':
        geocode = r['geocode'].replace('[', '').replace(']', '').split(',')
        geocode_dict[r['location']] = (float(geocode[0]), float(geocode[1]))

    # Retrieve geocode similarities dictionary
    geocode_sim_file = settings.output_home_directory + 'geocode-sim-dict.pkl'
    geocode_sim_dict = dict()

    R1, R2 = c.R1, c.R2
    I_ADDR1, N_ATOMIC = c.I_ADDR1, c.N_ATOMIC
    TYPE1, TYPE2 = c.TYPE1, c.TYPE2
    #
    if os.path.isfile(geocode_sim_file):
      geocode_sim_dict = pickle.load(open(geocode_sim_file, 'rb'))
    for node_id, node in self.G.nodes(data=True):
      if node[c.TYPE1] == c.N_RELTV:
        addr1 = self.record_dict[node[R1]][I_ADDR1]
        addr2 = self.record_dict[node[R2]][I_ADDR1]
        if addr1 == addr2:
          continue
        # Get the sim value
        sim_val = None
        if (addr1, addr2) in geocode_sim_dict.keys():
          sim_val = geocode_sim_dict[(addr1, addr2)]
        elif (addr2, addr1) in geocode_sim_dict.keys():
          sim_val = geocode_sim_dict[(addr2, addr1)]
        elif addr1 in geocode_dict and addr2 in geocode_dict:
          d = util.geocode_distance(geocode_dict[addr1], geocode_dict[addr2])
          sim_val = 1.0 - d * 1.0 / settings.geographical_comparison_dmax \
            if d <= settings.geographical_comparison_dmax else 0.0
          geocode_sim_dict[(addr1, addr2)] = sim_val
        # Update
        if sim_val is not None and sim_val >= hyperparams.atomic_t:
          updated = False
          for neighbor_id in self.G.predecessors(node_id):
            neighbor = self.G.nodes[neighbor_id]
            if neighbor[TYPE1] == N_ATOMIC and neighbor[TYPE2] == I_ADDR1:
              neighbor['s'] = sim_val
              updated = True
              break
          if not updated:
            self.node_id += 1
            self.G.add_node(self.node_id, r1=addr1, r2=addr2, t1=N_ATOMIC,
                            t2=I_ADDR1, s=sim_val, st=State.MERGED)
            # Add both keys
            self.atomic_nodes[I_ADDR1][(addr1, addr2)] = self.node_id
            self.atomic_nodes[I_ADDR1][(addr2, addr1)] = self.node_id

    pickle.dump(geocode_sim_dict, open(geocode_sim_file, 'wb'))

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
    if c.data_set == 'bhic':
      self.__gen_relationship_nodes_wblocking__()
    else:
      self.__gen_relationship_nodes__()
    stats.r_time = time.time() - srtime

  def __add_single_node_no_constraints__(self, p1, p2, relationship_type):
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

    # Retrieve and validate if the incoming atomic node count is
    # greater than the threshold value of settings.atomic_node_count_t
    # relationship_node_exists, incoming_atomic_nodes = \
    #   self.__get_incoming_atomic_nodes__(p1, p2)

    node_key = (p1[c.I_ID], p2[c.I_ID])
    if node_key in self.relationship_nodes:
      return None

    incoming_atomic_nodes = [(attr_index, self.atomic_nodes[attr_index][
      (p1[attr_index], p2[attr_index])])
                             for attr_index in self.attributes.attr_index_list
                             if (p1[attr_index], p2[attr_index]) in
                             self.atomic_nodes[attr_index]]

    if len(incoming_atomic_nodes) == 0:
      return None

    self.node_id += 1
    self.relationship_nodes[node_key] = self.node_id
    self.relationship_nodes[(node_key[1], node_key[0])] = self.node_id

    # r1 and r2 are in temporally increasing order.
    # Timestamp of the node is the earliest timestamp of two.
    self.G.add_node(self.node_id, r1=p1[c.I_ID], r2=p2[c.I_ID], t1=c.N_RELTV,
                    t2=relationship_type, s=0, st=State.ACTIVE)

    # Add incoming edges for the atomic nodes
    for r_node_type, r_node_id in incoming_atomic_nodes:
      self.G.add_edge(r_node_id, self.node_id, t1=c.E_REAL, t2=r_node_type)

    return self.node_id

  def __add_two_nodes__(self, p1, p2, types_list):
    """
    Function to add two graph nodes (person and spouse nodes) including the
    edges between relationship nodes and edges from atomic nodes.

    :param p1: Person 1
    :param p2: person 2
    :param types_list: List of types
    """
    person_node_id = self.__add_single_node__(p1, p2, types_list[0],
                                              self.attributes)

    # Add node for spouse
    if p1[c.I_SPOUSE] is not None and p2[c.I_SPOUSE] is not None:
      spouse_node_id = self.__add_single_node__(p1[c.I_SPOUSE],
                                                p2[c.I_SPOUSE],
                                                types_list[1], self.attributes)
      # Add edge between person-person node and spouse-spouse node
      if person_node_id is not None and spouse_node_id is not None:
        self.G.add_edge(person_node_id, spouse_node_id, t1=c.E_BOOL_W,
                        t2=Edge_Type.SPOUSE)
        self.G.add_edge(spouse_node_id, person_node_id, t1=c.E_BOOL_W,
                        t2=Edge_Type.SPOUSE)

  def __add_three_nodes__(self, p1, p2, types_list):
    """
    Function to add three nodes (child, father, mother) including the edges
    between relationship nodes and edges from atomic nodes.

    :param p1: Person 1
    :param p2: person 2
    :param types_list: List of types
    """

    child_node_id = self.__add_single_node__(p1, p2, types_list[0],
                                             self.attributes)

    father_node_id, mother_node_id = None, None

    # Add node for mother
    if p1[c.I_MOTHER] is not None and p2[c.I_MOTHER] is not None:
      mother_node_id = self.__add_single_node__(p1[c.I_MOTHER],
                                                p2[c.I_MOTHER],
                                                types_list[1], self.attributes)
      # Add edge between child-child node and mother-mother node
      if mother_node_id is not None and child_node_id is not None:
        self.G.add_edge(child_node_id, mother_node_id, t1=c.E_BOOL_S,
                        t2=Edge_Type.CHILDREN)
        self.G.add_edge(mother_node_id, child_node_id, t1=c.E_BOOL_W,
                        t2=Edge_Type.MOTHER)

    # Add node for father
    if p1[c.I_FATHER] is not None and p2[c.I_FATHER] is not None:
      father_node_id = self.__add_single_node__(p1[c.I_FATHER],
                                                p2[c.I_FATHER],
                                                types_list[2], self.attributes)
      # Add edge between child-child node and father-father node
      if father_node_id is not None and child_node_id is not None:
        self.G.add_edge(child_node_id, father_node_id, t1=c.E_BOOL_S,
                        t2=Edge_Type.CHILDREN)
        self.G.add_edge(father_node_id, child_node_id, t1=c.E_BOOL_W,
                        t2=Edge_Type.FATHER)

    # Add edge between mother-mother node and father-father node
    if mother_node_id is not None and father_node_id is not None:
      self.G.add_edge(father_node_id, mother_node_id, t1=c.E_BOOL_W,
                      t2=Edge_Type.SPOUSE)
      self.G.add_edge(mother_node_id, father_node_id, t1=c.E_BOOL_W,
                      t2=Edge_Type.SPOUSE)

  def __add_four_nodes__(self, p1, p2, types_list):
    """
    Function to add four nodes basing the child.

    :param p1: Person 1
    :param p2: person 2
    :param types_list: List of types
    """
    child_node_id = self.__add_single_node__(p1, p2, types_list[0],
                                             self.attributes)

    father_node_id, mother_node_id = None, None

    # Add node for mother
    if p1[c.I_MOTHER] is not None and p2[c.I_MOTHER] is not None:
      mother_node_id = self.__add_single_node__(p1[c.I_MOTHER],
                                                p2[c.I_MOTHER],
                                                types_list[1], self.attributes)
      # Add edge between child-child node and mother-mother node
      if mother_node_id is not None and child_node_id is not None:
        self.G.add_edge(child_node_id, mother_node_id, t1=c.E_BOOL_S,
                        t2=Edge_Type.CHILDREN)
        self.G.add_edge(mother_node_id, child_node_id, t1=c.E_BOOL_W,
                        t2=Edge_Type.MOTHER)

    # Add node for father
    if p1[c.I_FATHER] is not None and p2[c.I_FATHER] is not None:
      father_node_id = self.__add_single_node__(p1[c.I_FATHER],
                                                p2[c.I_FATHER],
                                                types_list[2], self.attributes)
      # Add edge between child-child node and father-father node
      if father_node_id is not None and child_node_id is not None:
        self.G.add_edge(child_node_id, father_node_id, t1=c.E_BOOL_S,
                        t2=Edge_Type.CHILDREN)
        self.G.add_edge(father_node_id, child_node_id, t1=c.E_BOOL_W,
                        t2=Edge_Type.FATHER)

    # Add edge between mother-mother node and father-father node
    if mother_node_id is not None and father_node_id is not None:
      self.G.add_edge(father_node_id, mother_node_id, t1=c.E_BOOL_W,
                      t2=Edge_Type.SPOUSE)
      self.G.add_edge(mother_node_id, father_node_id, t1=c.E_BOOL_W,
                      t2=Edge_Type.SPOUSE)

    # Add node for spouse
    if p1[c.I_SPOUSE] is not None and p2[c.I_SPOUSE] is not None:
      node_s_id = self.__add_single_node__(p1[c.I_SPOUSE], p2[c.I_SPOUSE],
                                           types_list[3], self.attributes)
      # Add edge between person-person node and spouse-spouse node
      if node_s_id is not None and child_node_id is not None:
        self.G.add_edge(node_s_id, child_node_id, t1=c.E_BOOL_W,
                        t2=Edge_Type.SPOUSE)
        self.G.add_edge(child_node_id, node_s_id, t1=c.E_BOOL_W,
                        t2=Edge_Type.SPOUSE)

  def __gen_relationship_nodes__(self):
    """
    Function to generate the relationship nodes.
    """

    # People are separated based on gender.
    # A person without a proper gender is assigned to both categories.
    f = list(filter((lambda p: p[c.I_SEX] in [Sex.F, '']),
                    self.record_dict.values()))
    m = list(filter((lambda p: p[c.I_SEX] in [Sex.M, '']),
                    self.record_dict.values()))

    # People are categorized to each role based on gender
    females = {
      'Bb': filter(lambda p: p[c.I_ROLE] == Role.Bb, f),
      'Bp': filter(lambda p: p[c.I_ROLE] == Role.Bp, f),
      'Mm': filter(lambda p: p[c.I_ROLE] == Role.Mm, f),
      'Mbp': filter(lambda p: p[c.I_ROLE] == Role.Mbp, f),
      'Mgp': filter(lambda p: p[c.I_ROLE] == Role.Mgp, f),
      'Dd': filter(lambda p: p[c.I_ROLE] == Role.Dd, f),
      'Ds': filter(lambda p: p[c.I_ROLE] == Role.Ds, f),
      'Dp': filter(lambda p: p[c.I_ROLE] == Role.Dp, f)
    }
    logging.info('Female list sizes')
    for key, f_list in females.iteritems():
      logging.info('\t %s %s' % (key, len(f_list)))

    males = {
      'Bb': filter(lambda p: p[c.I_ROLE] == Role.Bb, m),
      'Bp': filter(lambda p: p[c.I_ROLE] == Role.Bp, m),
      'Mm': filter(lambda p: p[c.I_ROLE] == Role.Mm, m),
      'Mbp': filter(lambda p: p[c.I_ROLE] == Role.Mbp, m),
      'Mgp': filter(lambda p: p[c.I_ROLE] == Role.Mgp, m),
      'Dd': filter(lambda p: p[c.I_ROLE] == Role.Dd, m),
      'Ds': filter(lambda p: p[c.I_ROLE] == Role.Ds, m),
      'Dp': filter(lambda p: p[c.I_ROLE] == Role.Dp, m)
    }
    logging.info('Male list sizes')
    for key, f_list in males.iteritems():
      logging.info('\t %s %s' % (key, len(f_list)))

    # Optimization
    add_single_node = self.__add_single_node__
    add_two_nodes = self.__add_two_nodes__
    add_three_nodes = self.__add_three_nodes__
    add_four_nodes = self.__add_four_nodes__
    is_valid_node = self.__is_valid_node__

    people_dict = self.record_dict
    stats_dict = defaultdict(int)

    assert self.node_id >= max(self.G.nodes)
    print 'Asserted'

    # -------------------------------------------------------------------------
    logging.info('Generating Bb relations starting from {}'.format(
      self.node_id))
    for gender in [females, males]:
      for b in gender['Bb']:

        # Baby to birth parent
        prev = self.node_id
        link = 'Bb-Bp'
        for p in gender['Bp']:
          if is_valid_node(b, p, link):
            add_single_node(b, p, link, self.attributes)
        stats_dict['bb_bp_count'] += (self.node_id - prev)

        # Baby to marriage bride/groom
        prev = self.node_id
        link = 'Bb-Mm'
        for m in gender['Mm']:

          # Retrieve parent link Mbp or Mgp
          assert m[c.I_SEX] in [Sex.F, Sex.M]
          parent_link = 'Bp-Mbp' if m[c.I_SEX] == Sex.F else 'Bp-Mgp'

          if is_valid_node(b, m, link):
            add_three_nodes(b, m, [link, parent_link, parent_link])
          else:
            # Even if the babies are different, we need to add mother-mother
            # and father-father nodes corresponding to the siblings
            if b[c.I_MOTHER] is not None and m[c.I_MOTHER] is not None:
              add_two_nodes(people_dict[b[c.I_MOTHER]],
                            people_dict[m[c.I_MOTHER]],
                            [parent_link, parent_link])
        stats_dict['bb_mm_count'] += (self.node_id - prev)

        # Baby to marriage bride parent
        prev = self.node_id
        link = 'Bb-Mbp'
        for p in gender['Mbp']:
          if is_valid_node(b, p, link):
            add_single_node(b, p, link, self.attributes)
        stats_dict['bb_mbp_count'] += (self.node_id - prev)

        # Baby to marriage groom parent
        prev = self.node_id
        link = 'Bb-Mgp'
        for p in gender['Mgp']:
          if is_valid_node(b, p, link):
            add_single_node(b, p, link, self.attributes)
        stats_dict['bb_mgp_count'] += (self.node_id - prev)

        # Baby to death person
        prev = self.node_id
        link = 'Bb-Dd'
        for d in gender['Dd']:
          if is_valid_node(b, d, link):
            add_three_nodes(b, d, [link, 'Bp-Dp', 'Bp-Dp'])
          else:
            # Even if the babies are different, we need to add mother-mother
            # and father-father nodes corresponding to the siblings
            if b[c.I_MOTHER] is not None and d[c.I_MOTHER] is not None:
              add_two_nodes(people_dict[b[c.I_MOTHER]],
                            people_dict[d[c.I_MOTHER]], ['Bp-Dp', 'Bp-Dp'])
        stats_dict['bb_dd_count'] += (self.node_id - prev)

        # Baby to death spouse
        prev = self.node_id
        link = 'Bb-Ds'
        for s in gender['Ds']:
          if is_valid_node(b, s, link):
            add_single_node(b, s, link, self.attributes)
        stats_dict['bb_ds_count'] += (self.node_id - prev)

        # Baby to death parent
        prev = self.node_id
        link = 'Bb-Dp'
        for p in gender['Dp']:
          if is_valid_node(b, p, link):
            add_single_node(b, p, link, self.attributes)
        stats_dict['bb_dp_count'] += (self.node_id - prev)

    # -------------------------------------------------------------------------
    logging.info('Generating birth parent relations starting with {}'.format(
      self.node_id))

    for gender in [females, males]:
      for p in gender['Bp']:

        # Birth parent to birth parent
        prev = self.node_id
        link = 'Bp-Bp'
        for p2 in gender['Bp']:
          if p2 == p:
            continue
          if is_valid_node(p, p2, link):
            add_two_nodes(p, p2, [link, link])
        stats_dict['bp_bp_count'] += (self.node_id - prev)

        # Birth parent to marriage bride/groom
        prev = self.node_id
        link = 'Bp-Mm'
        for m in gender['Mm']:
          if is_valid_node(p, m, link):
            add_two_nodes(p, m, [link, link])
        stats_dict['bp_mm_count'] += (self.node_id - prev)

        # Birth parent to death person
        prev = self.node_id
        link = 'Bp-Dd'
        for d in gender['Dd']:
          if is_valid_node(p, d, link):
            add_two_nodes(p, d, [link, 'Bp-Ds'])
        stats_dict['bp_dd_count'] += (self.node_id - prev)

        # Birth parent to death spouse
        prev = self.node_id
        for s in gender['Ds']:
          add_two_nodes(p, s, ['Bp-Ds', 'Bp-Dd'])
        stats_dict['bp_ds_count'] += (self.node_id - prev)

    # -------------------------------------------------------------------------
    logging.info(
      'Generating marriage bride/groom relations starting with {}'.format(
        self.node_id))

    for gender in [females, males]:
      for m in gender['Mm']:

        assert m[c.I_SEX] in [Sex.F, Sex.M]
        parent_role = 'Mbp' if m[c.I_SEX] == Sex.F else 'Mgp'

        # Marriage bride/groom to marriage bride/groom
        prev = self.node_id
        link = 'Mm-Mm'
        for m2 in gender['Mm']:
          if m == m2:
            continue
          # because m and m2 are of the same sex
          parent_link = parent_role + '-' + parent_role
          if is_valid_node(m, m2, link):
            add_three_nodes(m, m2, [link, parent_link, parent_link])
        stats_dict['mm_mm_count'] += (self.node_id - prev)

        # Marriage bride/groom to marriage bride parent
        prev = self.node_id
        link = 'Mm-Mbp'
        for p in gender['Mbp']:
          if is_valid_node(m, p, link):
            add_two_nodes(m, p, [link, link])
        stats_dict['mm_mbp_count'] += (self.node_id - prev)

        # Marriage bride/groom to marriage groom parent
        prev = self.node_id
        link = 'Mm-Mgp'
        for p in gender['Mgp']:
          if is_valid_node(m, p, link):
            add_two_nodes(m, p, [link, link])
        stats_dict['mm_mgp_count'] += (self.node_id - prev)

        # Marriage bride/groom to death person
        prev = self.node_id
        link = 'Mm-Dd'
        for d in gender['Dd']:
          if d[c.I_MARRITAL_STATUS] == 's':
            continue
          parent_link = parent_role + '-' + 'Dp'
          if is_valid_node(m, d, link):
            add_four_nodes(m, d, [link, parent_link, parent_link, 'Mm-Ds'])
          else:
            # Even if the persons are different, we need to add mother-mother
            # and father-father nodes corresponding to the siblings
            if m[c.I_MOTHER] is not None and d[c.I_MOTHER] is not None:
              add_two_nodes(people_dict[m[c.I_MOTHER]],
                            people_dict[d[c.I_MOTHER]],
                            [parent_link, parent_link])
        stats_dict['mm_dd_count'] += (self.node_id - prev)

        # Marriage bride/groom to death spouse
        prev = self.node_id
        for s in gender['Ds']:
          add_two_nodes(m, s, ['Mm-Ds', 'Mm-Dd'])
        stats_dict['mm_ds_count'] += (self.node_id - prev)

        # Marriage bride/groom to death parent
        # NOTE : Removed constraint. We can not say the baby dies after the
        # parent's marriage. May be parents are not married when baby dies.
        prev = self.node_id
        for p in gender['Dp']:
          add_two_nodes(m, p, ['Mm-Dp', 'Mm-Dp'])
        stats_dict['mm_dp_count'] += (self.node_id - prev)

    # -------------------------------------------------------------------------
    logging.info('Generating marriage bride parent relations {}'.format(
      self.node_id))
    for gender in [females, males]:
      for p in gender['Mbp']:

        # Marriage parent to death person
        prev = self.node_id
        for d in gender['Dd']:
          add_two_nodes(p, d, ['Mbp-Dd', 'Mbp-Ds'])
        stats_dict['mbp_dd_count'] += (self.node_id - prev)

        # Marriage parent to death spouse
        prev = self.node_id
        for s in gender['Ds']:
          add_two_nodes(p, s, ['Mbp-Ds', 'Mbp-Dd'])
        stats_dict['mbp_ds_count'] += (self.node_id - prev)

    # -------------------------------------------------------------------------
    logging.info('Generating marriage groom parent relations {}'.format(
      self.node_id))

    for gender in [females, males]:
      for p in gender['Mgp']:

        # Marriage parent to death person
        prev = self.node_id
        for d in gender['Dd']:
          add_two_nodes(p, d, ['Mgp-Dd', 'Mgp-Ds'])
        stats_dict['mgp_dd_count'] += (self.node_id - prev)

        # Marriage parent to death spouse
        prev = self.node_id
        for s in gender['Ds']:
          add_two_nodes(p, s, ['Mgp-Ds', 'Mgp-Dd'])
        stats_dict['mgp_ds_count'] += (self.node_id - prev)

    # -------------------------------------------------------------------------
    logging.info('Generating death person relations {}'.format(self.node_id))

    for gender in [females, males]:
      for d in gender['Dd']:

        # Death person to death spouse
        prev = self.node_id
        for s in gender['Ds']:
          add_two_nodes(d, s, ['Dd-Ds', 'Dd-Ds'])
        stats_dict['dd_ds_count'] += (self.node_id - prev)

        # Death person to death parent
        prev = self.node_id
        for p in gender['Dp']:
          add_two_nodes(d, p, ['Dd-Dp', 'Ds-Dp'])
        stats_dict['dd_dp_count'] += (self.node_id - prev)

    # -------------------------------------------------------------------------
    logging.info('Generating death spouse relations {}'.format(self.node_id))

    for gender in [females, males]:
      for s in gender['Ds']:

        # Death spouse to death spouse
        prev = self.node_id
        for s2 in gender['Ds']:
          if s == s2:
            continue
          add_single_node(s, s2, 'Ds-Ds', self.attributes)
        stats_dict['ds_ds_count'] += (self.node_id - prev)

        # Death spouse to death parent
        prev = self.node_id
        for p in gender['Dp']:
          add_two_nodes(s, p, ['Ds-Dp', 'Dd-Dp'])
        stats_dict['ds_dp_count'] += (self.node_id - prev)

    # -------------------------------------------------------------------------
    logging.info('Generating death parent relations {}'.format(self.node_id))

    for gender in [females, males]:
      for p in gender['Dp']:

        # Death parent to death parent
        prev = self.node_id
        for p2 in gender['Dp']:
          if p == p2:
            continue
          add_two_nodes(p, p2, ['Dp-Dp', 'Dp-Dp'])
        stats_dict['dp_dp_count'] += (self.node_id - prev)

    # -------------------------------------------------------------------------
    logging.info(
      'Generating parent links for different gender siblings {}'.format(
        self.node_id))

    for gender in [(females, males), (males, females)]:
      for b in gender[0]['Bb']:

        prev = self.node_id
        for m in gender[1]['Mm']:
          if b[c.I_MOTHER] is not None and m[c.I_MOTHER] is not None:
            assert m[c.I_SEX] in [Sex.F, Sex.M]
            parent_link = 'Bp-Mbp' if m[c.I_SEX] == Sex.F else 'Bp-Mgp'
            add_two_nodes(people_dict[b[c.I_MOTHER]],
                          people_dict[m[c.I_MOTHER]],
                          [parent_link, parent_link])
        stats_dict['PARENT_bb_mm_count'] += (self.node_id - prev)

        prev = self.node_id
        for d in gender[1]['Dd']:
          if b[c.I_MOTHER] is not None and d[c.I_MOTHER] is not None:
            add_two_nodes(people_dict[b[c.I_MOTHER]],
                          people_dict[d[c.I_MOTHER]],
                          ['Bp-Dp', 'Bp-Dp'])
        stats_dict['PARENT-M_bb_dd_count'] += (self.node_id - prev)

      for m in gender[0]['Mm']:
        parent_role1 = 'Mbp' if m[c.I_SEX] == Sex.F else 'Mgp'

        prev = self.node_id
        for d in gender[1]['Dd']:
          if d[c.I_MARRITAL_STATUS] == 's':
            continue
          if m[c.I_MOTHER] is not None and d[c.I_MOTHER] is not None:
            parent_link = parent_role1 + '-' + 'Dp'
            add_two_nodes(people_dict[m[c.I_MOTHER]],
                          people_dict[d[c.I_MOTHER]],
                          [parent_link, parent_link])
        stats_dict['PARENT-M_mm_dd_count'] += (self.node_id - prev)

    ## Cross gender Bp-Bp node generation is not needed because we already
    # generated Bp-Bp links directly. Those links were not generated
    # indirectly based on Bb-Bb.

    ## Cross gender Dp-Dp node generation is not needed because we already
    # generated Dp-Dp links directly. Those links were not generated
    # indirectly based on Dd-Dd.

    logging.info('Generating cross gender marriage parents')
    link = 'Mbp-Mgp'
    for f_m in females['Mm']:
      for m_m in males['Mm']:
        if f_m == m_m:
          continue
        if f_m[c.I_MOTHER] is not None and m_m[c.I_MOTHER] is not None:
          add_two_nodes(people_dict[f_m[c.I_MOTHER]],
                        people_dict[m_m[c.I_MOTHER]],
                        [link, link])

    stats.print_node_stats(stats_dict)

  def __gen_relationship_nodes_wblocking__(self):
    """
    Function to generate the relationship nodes.
    """

    # People are separated based on gender.
    # A person without a proper gender is assigned to both categories.
    f_list = list(filter((lambda p: p[c.I_SEX] in [Sex.F, '']),
                         self.record_dict.values()))
    m_list = list(filter((lambda p: p[c.I_SEX] in [Sex.M, '']),
                         self.record_dict.values()))

    female_blocks = defaultdict(lambda: defaultdict(set))
    for female in f_list:
      fname_key = encode.dmetaphone(female[c.I_FNAME])
      sname_key = encode.dmetaphone(female[c.I_SNAME])
      female_blocks[fname_key][sname_key].add(female[c.I_ID])

    male_blocks = defaultdict(lambda: defaultdict(set))
    for male in m_list:
      fname_key = encode.dmetaphone(male[c.I_FNAME])
      sname_key = encode.dmetaphone(male[c.I_SNAME])
      male_blocks[fname_key][sname_key].add(male[c.I_ID])

    # Optimization
    add_single_node = self.__add_single_node__
    add_two_nodes = self.__add_two_nodes__
    add_three_nodes = self.__add_three_nodes__
    add_four_nodes = self.__add_four_nodes__
    is_valid_node = self.__is_valid_node__
    people_dict = self.record_dict
    stats_dict = defaultdict(int)

    for block in [female_blocks, male_blocks]:
      for fname_key, sname_block in block.iteritems():
        for sname_key, people_id_list in sname_block.iteritems():

          # People are categorized to each role based on gender
          #
          people_list = map(lambda p_id: self.record_dict[p_id], people_id_list)
          people = {
            'Bb': filter(lambda p: p[c.I_ROLE] == Role.Bb, people_list),
            'Bp': filter(lambda p: p[c.I_ROLE] == Role.Bp, people_list),
            'Mm': filter(lambda p: p[c.I_ROLE] == Role.Mm, people_list),
            'Mbp': filter(lambda p: p[c.I_ROLE] == Role.Mbp, people_list),
            'Mgp': filter(lambda p: p[c.I_ROLE] == Role.Mgp, people_list),
            'Dd': filter(lambda p: p[c.I_ROLE] == Role.Dd, people_list),
            'Ds': filter(lambda p: p[c.I_ROLE] == Role.Ds, people_list),
            'Dp': filter(lambda p: p[c.I_ROLE] == Role.Dp, people_list)
          }

          blocking_key = '%s-%s' % (fname_key, sname_key)
          logging.info('Blocking key %s length %s' % (blocking_key,
                                                      len(people_list)))
          util.print_memory_usage(blocking_key)
          for role in Role.list():
            logging.info('\t %s - %s' % (role, len(people[role])))

          # -------------------------------------------------------------------------
          logging.info(
            'Generating Bb relations starting {}'.format(self.node_id))
          for b in people['Bb']:

            # Baby to birth parent
            prev = self.node_id
            link = 'Bb-Bp'
            for p in people['Bp']:
              if is_valid_node(b, p, link):
                add_single_node(b, p, link, self.attributes)
            stats_dict['bb_bp_count'] += (self.node_id - prev)

            # Baby to marriage bride/groom
            prev = self.node_id
            link = 'Bb-Mm'
            for m in people['Mm']:

              # Retrieve parent link Mbp or Mgp
              assert m[c.I_SEX] in [Sex.F, Sex.M]
              parent_link = 'Bp-Mbp' if m[c.I_SEX] == Sex.F else 'Bp-Mgp'

              if is_valid_node(b, m, link):
                add_three_nodes(b, m, [link, parent_link, parent_link])
              else:
                # Even if the babies are different, we need to add mother-mother
                # and father-father nodes corresponding to the siblings
                if b[c.I_MOTHER] is not None and m[c.I_MOTHER] is not None:
                  add_two_nodes(people_dict[b[c.I_MOTHER]],
                                people_dict[m[c.I_MOTHER]],
                                [parent_link, parent_link])
            stats_dict['bb_mm_count'] += (self.node_id - prev)

            # Baby to marriage bride parent
            prev = self.node_id
            link = 'Bb-Mbp'
            for p in people['Mbp']:
              if is_valid_node(b, p, link):
                add_single_node(b, p, link, self.attributes)
            stats_dict['bb_mbp_count'] += (self.node_id - prev)

            # Baby to marriage groom parent
            prev = self.node_id
            link = 'Bb-Mgp'
            for p in people['Mgp']:
              if is_valid_node(b, p, link):
                add_single_node(b, p, link, self.attributes)
            stats_dict['bb_mgp_count'] += (self.node_id - prev)

            # Baby to death person
            prev = self.node_id
            link = 'Bb-Dd'
            for d in people['Dd']:
              if is_valid_node(b, d, link):
                add_three_nodes(b, d, [link, 'Bp-Dp', 'Bp-Dp'])
              else:
                # Even if the babies are different, we need to add mother-mother
                # and father-father nodes corresponding to the siblings
                if b[c.I_MOTHER] is not None and d[c.I_MOTHER] is not None:
                  add_two_nodes(people_dict[b[c.I_MOTHER]],
                                people_dict[d[c.I_MOTHER]], ['Bp-Dp', 'Bp-Dp'])
            stats_dict['bb_dd_count'] += (self.node_id - prev)

            # Baby to death spouse
            prev = self.node_id
            link = 'Bb-Ds'
            for s in people['Ds']:
              if is_valid_node(b, s, link):
                add_single_node(b, s, link, self.attributes)
            stats_dict['bb_ds_count'] += (self.node_id - prev)

            # Baby to death parent
            prev = self.node_id
            link = 'Bb-Dp'
            for p in people['Dp']:
              if is_valid_node(b, p, link):
                add_single_node(b, p, link, self.attributes)
            stats_dict['bb_dp_count'] += (self.node_id - prev)

          # -------------------------------------------------------------------------
          logging.info(
            'Generating birth parent relations starting with {}'.format(
              self.node_id))

          for p in people['Bp']:

            # Birth parent to birth parent
            prev = self.node_id
            link = 'Bp-Bp'
            for p2 in people['Bp']:
              if p2 == p:
                continue
              if is_valid_node(p, p2, link):
                add_two_nodes(p, p2, [link, link])
            stats_dict['bp_bp_count'] += (self.node_id - prev)

            # Birth parent to marriage bride/groom
            prev = self.node_id
            link = 'Bp-Mm'
            for m in people['Mm']:
              if is_valid_node(p, m, link):
                add_two_nodes(p, m, [link, link])
            stats_dict['bp_mm_count'] += (self.node_id - prev)

            # Birth parent to death person
            prev = self.node_id
            link = 'Bp-Dd'
            for d in people['Dd']:
              if is_valid_node(p, d, link):
                add_two_nodes(p, d, [link, 'Bp-Ds'])
            stats_dict['bp_dd_count'] += (self.node_id - prev)

            # Birth parent to death spouse
            prev = self.node_id
            for s in people['Ds']:
              add_two_nodes(p, s, ['Bp-Ds', 'Bp-Dd'])
            stats_dict['bp_ds_count'] += (self.node_id - prev)

          # -------------------------------------------------------------------------
          logging.info(
            'Generating marriage bride/groom relations starting with {}'.format(
              self.node_id))

          for m in people['Mm']:

            assert m[c.I_SEX] in [Sex.F, Sex.M]
            parent_role = 'Mbp' if m[c.I_SEX] == Sex.F else 'Mgp'

            # Marriage bride/groom to marriage bride/groom
            prev = self.node_id
            link = 'Mm-Mm'
            for m2 in people['Mm']:
              if m == m2:
                continue
              # because m and m2 are of the same sex
              parent_link = parent_role + '-' + parent_role
              if is_valid_node(m, m2, link):
                add_three_nodes(m, m2, [link, parent_link, parent_link])
            stats_dict['mm_mm_count'] += (self.node_id - prev)

            # Marriage bride/groom to marriage bride parent
            prev = self.node_id
            link = 'Mm-Mbp'
            for p in people['Mbp']:
              if is_valid_node(m, p, link):
                add_two_nodes(m, p, [link, link])
            stats_dict['mm_mbp_count'] += (self.node_id - prev)

            # Marriage bride/groom to marriage groom parent
            prev = self.node_id
            link = 'Mm-Mgp'
            for p in people['Mgp']:
              if is_valid_node(m, p, link):
                add_two_nodes(m, p, [link, link])
            stats_dict['mm_mgp_count'] += (self.node_id - prev)

            # Marriage bride/groom to death person
            prev = self.node_id
            link = 'Mm-Dd'
            for d in people['Dd']:
              if d[c.I_MARRITAL_STATUS] == 's':
                continue
              parent_link = parent_role + '-' + 'Dp'
              if is_valid_node(m, d, link):
                add_four_nodes(m, d, [link, parent_link, parent_link, 'Mm-Ds'])
              else:
                # Even if the persons are different, we need to add mother-mother
                # and father-father nodes corresponding to the siblings
                if m[c.I_MOTHER] is not None and d[c.I_MOTHER] is not None:
                  add_two_nodes(people_dict[m[c.I_MOTHER]],
                                people_dict[d[c.I_MOTHER]],
                                [parent_link, parent_link])
            stats_dict['mm_dd_count'] += (self.node_id - prev)

            # Marriage bride/groom to death spouse
            prev = self.node_id
            for s in people['Ds']:
              add_two_nodes(m, s, ['Mm-Ds', 'Mm-Dd'])
            stats_dict['mm_ds_count'] += (self.node_id - prev)

            # Marriage bride/groom to death parent
            # NOTE : Removed constraint. We can not say the baby dies after the
            # parent's marriage. May be parents are not married when baby dies.
            prev = self.node_id
            for p in people['Dp']:
              add_two_nodes(m, p, ['Mm-Dp', 'Mm-Dp'])
            stats_dict['mm_dp_count'] += (self.node_id - prev)

          # -------------------------------------------------------------------------
          logging.info('Generating marriage bride parent relations {}'.format(
            self.node_id))
          for p in people['Mbp']:

            # Marriage parent to death person
            prev = self.node_id
            for d in people['Dd']:
              add_two_nodes(p, d, ['Mbp-Dd', 'Mbp-Ds'])
            stats_dict['mbp_dd_count'] += (self.node_id - prev)

            # Marriage parent to death spouse
            prev = self.node_id
            for s in people['Ds']:
              add_two_nodes(p, s, ['Mbp-Ds', 'Mbp-Dd'])
            stats_dict['mbp_ds_count'] += (self.node_id - prev)

          # -------------------------------------------------------------------------
          logging.info('Generating marriage groom parent relations {}'.format(
            self.node_id))

          for p in people['Mgp']:

            # Marriage parent to death person
            prev = self.node_id
            for d in people['Dd']:
              add_two_nodes(p, d, ['Mgp-Dd', 'Mgp-Ds'])
            stats_dict['mgp_dd_count'] += (self.node_id - prev)

            # Marriage parent to death spouse
            prev = self.node_id
            for s in people['Ds']:
              add_two_nodes(p, s, ['Mgp-Ds', 'Mgp-Dd'])
            stats_dict['mgp_ds_count'] += (self.node_id - prev)

          # -------------------------------------------------------------------------
          logging.info(
            'Generating death person relations {}'.format(self.node_id))

          for d in people['Dd']:

            # Death person to death spouse
            prev = self.node_id
            for s in people['Ds']:
              add_two_nodes(d, s, ['Dd-Ds', 'Dd-Ds'])
            stats_dict['dd_ds_count'] += (self.node_id - prev)

            # Death person to death parent
            prev = self.node_id
            for p in people['Dp']:
              add_two_nodes(d, p, ['Dd-Dp', 'Ds-Dp'])
            stats_dict['dd_dp_count'] += (self.node_id - prev)

          # -------------------------------------------------------------------------
          logging.info(
            'Generating death spouse relations {}'.format(self.node_id))

          for s in people['Ds']:

            # Death spouse to death spouse
            prev = self.node_id
            for s2 in people['Ds']:
              if s == s2:
                continue
              add_single_node(s, s2, 'Ds-Ds', self.attributes)
            stats_dict['ds_ds_count'] += (self.node_id - prev)

            # Death spouse to death parent
            prev = self.node_id
            for p in people['Dp']:
              add_two_nodes(s, p, ['Ds-Dp', 'Dd-Dp'])
            stats_dict['ds_dp_count'] += (self.node_id - prev)

          # -------------------------------------------------------------------------
          logging.info(
            'Generating death parent relations {}'.format(self.node_id))

          for p in people['Dp']:

            # Death parent to death parent
            prev = self.node_id
            for p2 in people['Dp']:
              if p == p2:
                continue
              add_two_nodes(p, p2, ['Dp-Dp', 'Dp-Dp'])
            stats_dict['dp_dp_count'] += (self.node_id - prev)

    # Generating cross gender relationships
    #
    # -------------------------------------------------------------------------
    logging.info(
      'Generating parent links for different gender siblings {}'.format(
        self.node_id))
    for block1, block2 in [(female_blocks, male_blocks),
                           (male_blocks, female_blocks)]:

      for fname_key, sname_block1 in block1.iteritems():
        for sname_key, people_id_list in sname_block1.iteritems():

          if len(block2[fname_key][sname_key]) > 1:

            # Block 1 people lists
            people_list_1 = map(lambda p_id: self.record_dict[p_id],
                                block1[fname_key][sname_key])
            block1_people = {
              'Bb': filter(lambda p: p[c.I_ROLE] == Role.Bb, people_list_1),
              'Mm': filter(lambda p: p[c.I_ROLE] == Role.Mm, people_list_1)
            }

            # Block 2 people lists
            people_list_2 = map(lambda p_id: self.record_dict[p_id],
                                block2[fname_key][sname_key])
            block2_people = {
              'Mm': filter(lambda p: p[c.I_ROLE] == Role.Mm, people_list_2),
              'Dd': filter(lambda p: p[c.I_ROLE] == Role.Dd, people_list_2)
            }

            # Birth parent - Marriage parent and Birth parent - Death parent
            #
            for b in block1_people['Bb']:
              prev = self.node_id
              for m in block2_people['Mm']:
                if b[c.I_MOTHER] is not None and m[c.I_MOTHER] is not None:
                  assert m[c.I_SEX] in [Sex.F, Sex.M]
                  parent_link = 'Bp-Mbp' if m[c.I_SEX] == Sex.F else 'Bp-Mgp'
                  add_two_nodes(people_dict[b[c.I_MOTHER]],
                                people_dict[m[c.I_MOTHER]],
                                [parent_link, parent_link])
              stats_dict['PARENT_bb_mm_count'] += (self.node_id - prev)

              prev = self.node_id
              for d in block2_people['Dd']:
                if b[c.I_MOTHER] is not None and d[c.I_MOTHER] is not None:
                  add_two_nodes(people_dict[b[c.I_MOTHER]],
                                people_dict[d[c.I_MOTHER]],
                                ['Bp-Dp', 'Bp-Dp'])
              stats_dict['PARENT-M_bb_dd_count'] += (self.node_id - prev)

            # Marriage parent - Death parent
            #
            for m in block1_people['Mm']:
              parent_role1 = 'Mbp' if m[c.I_SEX] == Sex.F else 'Mgp'

              prev = self.node_id
              for d in block2_people['Dd']:
                if d[c.I_MARRITAL_STATUS] == 's':
                  continue
                if m[c.I_MOTHER] is not None and d[c.I_MOTHER] is not None:
                  parent_link = parent_role1 + '-' + 'Dp'
                  add_two_nodes(people_dict[m[c.I_MOTHER]],
                                people_dict[d[c.I_MOTHER]],
                                [parent_link, parent_link])
              stats_dict['PARENT-M_mm_dd_count'] += (self.node_id - prev)

    # -------------------------------------------------------------------------
    ## Cross gender Bp-Bp node generation is not needed because we already
    # generated Bp-Bp links directly. Those links were not generated
    # indirectly based on Bb-Bb.

    ## Cross gender Dp-Dp node generation is not needed because we already
    # generated Dp-Dp links directly. Those links were not generated
    # indirectly based on Dd-Dd.

    logging.info('Generating cross gender marriage parents')
    link = 'Mbp-Mgp'
    for fname_key, sname_block1 in female_blocks.iteritems():
      for sname_key, people_id_list in sname_block1.iteritems():

        if len(male_blocks[fname_key][sname_key]) > 1:
          female_list = map(lambda p_id: self.record_dict[p_id],
                            female_blocks[fname_key][sname_key])
          bride_list = filter(lambda p: p[c.I_ROLE] == Role.Mm, female_list)

          male_list = map(lambda p_id: self.record_dict[p_id],
                          male_blocks[fname_key][sname_key])
          groom_list = filter(lambda p: p[c.I_ROLE] == Role.Mm, male_list)

          for f_m in bride_list:
            for m_m in groom_list:
              if f_m == m_m:
                continue
              if f_m[c.I_MOTHER] is not None and m_m[c.I_MOTHER] is not None:
                add_two_nodes(people_dict[f_m[c.I_MOTHER]],
                              people_dict[m_m[c.I_MOTHER]], [link, link])

    stats.print_node_stats(stats_dict)

  def __is_valid_node__(self, p1, p2, link):
    """
    Checks if a node created from `p1` and `p2` is valid based on the temporal
    constraints.

    :param p1: Person 1
    :param p2: Person 2
    :param link: Link between two person records
    :return: True if valid
    """
    if link in ['Bb-Ds', 'Bb-Dp', 'Mm-Mbp', 'Mm-Mgp']:
      if util.is_after_onegen_dates(p1[c.I_EDATE], p2[c.I_EDATE]):
        return True
      return False

    if link in ['Bb-Mbp', 'Bb-Mgp']:
      if util.is_after_twogen(p1[c.I_EDATE], p2[c.I_EDATE]):
        return True
      return False

    if link in ['Bb-Dd', 'Mm-Dd']:
      if util.is_after(p1[c.I_EDATE], p2[c.I_EDATE]):
        # Check if the birth years are not far apart
        if p1[c.I_BIRTH_YEAR] is not None and p2[c.I_BIRTH_YEAR] is not None:
          if util.is_almost_same_years(p1[c.I_BIRTH_YEAR], p2[c.I_BIRTH_YEAR]):
            return True
        else:
          return True
      return False

    if link == 'Bb-Bp':
      if util.is_after_onegen_dates(p1[c.I_EDATE], p2[c.I_EDATE]):
        if p2[c.I_MARRIAGE_YEAR] is not None:
          if util.is_after_onegen_years(p1[c.I_BIRTH_YEAR],
                                        p2[c.I_MARRIAGE_YEAR]):
            return True
        else:
          return True
      return False

    if link == 'Bb-Mm':
      if util.is_after_onegen_dates(p1[c.I_EDATE], p2[c.I_EDATE]):
        # Check if the birth years are not far apart
        if p1[c.I_BIRTH_YEAR] is not None and p2[c.I_BIRTH_YEAR] is not None:
          if util.is_almost_same_years(p1[c.I_BIRTH_YEAR], p2[c.I_BIRTH_YEAR]):
            return True
        else:
          return True
      return False

    if link == 'Bp-Bp':
      if util.is_nine_months_apart(p1[c.I_EDATE], p2[c.I_EDATE]):
        if p1[c.I_MARRIAGE_YEAR] is not None and \
            p2[c.I_MARRIAGE_YEAR] is not None:
          if util.is_almost_same_marriage_years(p1[c.I_MARRIAGE_YEAR],
                                                p2[c.I_MARRIAGE_YEAR]):
            return True
        else:
          return True
      return False

    if link == 'Bp-Mm':
      if p1[c.I_MARRIAGE_YEAR] is not None and p2[c.I_MARRIAGE_YEAR] is not \
          None:
        if util.is_almost_same_marriage_years(p1[c.I_MARRIAGE_YEAR],
                                              p2[c.I_MARRIAGE_YEAR]):
          return True
      else:
        return True
      return False

    if link == 'Bp-Dd':
      # Need to check both because of empty gender values
      # Females, death should be after birth of baby
      if p1[c.I_SEX] == Sex.F or p2[c.I_SEX] == Sex.F:
        if util.is_after(p1[c.I_EDATE], p2[c.I_EDATE]):
          return True
      # Males, death should be at most 9 months before birth
      else:
        if util.is_atmost_nine_months_before(p1[c.I_EDATE], p2[c.I_EDATE]):
          return True
      return False

    if link == 'Mm-Mm':
      if p1[c.I_BIRTH_YEAR] is not None and p2[c.I_BIRTH_YEAR] is not None:
        if util.is_almost_same_years(p1[c.I_BIRTH_YEAR], p2[c.I_BIRTH_YEAR]):
          return True
      else:
        return True
      return False

    return True

  # ---------------------------------------------------------------------------
  # Serialization
  # ---------------------------------------------------------------------------
  def __get_people_dict__(self, people_file):
    """
    Deserializing people
    :param people_file:
    :return:
    """
    people_dict = {}
    people_list = pickle.load(open(people_file, 'r'))

    # fname_counter = Counter()
    # sname_counter = Counter()
    for p in people_list:
      people_dict[p[c.I_ID]] = p
      # fname_counter.update([p[c.I_FNAME]])
      # sname_counter.update([p[c.I_SNAME]])

    # logging.info('FNAME')
    # for key, value in sorted(fname_counter.iteritems(), key=lambda x: x[1],
    #                          reverse=True):
    #   logging.info('{} - {}'.format(key, value))
    #
    # logging.info('SNAME')
    # for key, value in sorted(sname_counter.iteritems(), key=lambda x: x[1],
    #                          reverse=True):
    #   logging.info('{} - {}'.format(key, value))

    #   if p[c.I_BIRTH_YEAR] == '':
    #     p[c.I_BIRTH_YEAR] = None
    #   if p[c.I_MARRIAGE_YEAR] == '':
    #     p[c.I_MARRIAGE_YEAR] = None
    #   if p[c.I_BIRTH_YEAR] is not None:
    #     p[c.I_BIRTH_YEAR] = int(p[c.I_BIRTH_YEAR])
    #   if p[c.I_MARRIAGE_YEAR] is not None:
    #     p[c.I_MARRIAGE_YEAR] = int(p[c.I_MARRIAGE_YEAR])
    #   people_dict[p[c.I_ID]] = p
    # # todo:remove
    # pickle.dump(people_dict.values(), open(people_file, 'wb'))
    logging.info('People count {}'.format(len(people_dict)))
    return people_dict

  # ---------------------------------------------------------------------------
  # Graph processing
  # ---------------------------------------------------------------------------
  def bootstrap(self, bootstrap_threshold_list):

    # Calculate node atomic similarity
    #
    calculate_sim_atomic = self.calculate_sim_atomic
    for node_id, node in self.G.nodes(data=True):
      if node[c.TYPE1] == c.N_ATOMIC:
        continue
      calculate_sim_atomic(node_id, node)

    # Enumerate graph groups
    #
    self.__enumerate_graph_groups_with_avg_similarity__()

    # Bootstrapping the graph
    #
    nodes = self.G.nodes
    enrich_node = self.__enrich_node__
    __is_valid_entity_role_merge__ = self.__is_valid_entity_role_merge__
    people_dict = self.record_dict
    relationship_nodes = self.relationship_nodes
    merge_threshold = hyperparams.merge_t

    group_dict = defaultdict(list)
    __get_group_avg_similarity__ = self.__get_group_avg_similarity__

    # Initializing the nodes queue for ablation study without relationship
    # structure
    if hyperparams.scenario.startswith('norel'):
      for cc in nx.strongly_connected_components(self.G):
        a_node = self.G.nodes[next(iter(cc))]

        # Skip atomic nodes and single relational nodes
        if a_node[c.TYPE1] == c.N_ATOMIC or len(cc) == 1:
          continue
        group_dict[1].append((cc, __get_group_avg_similarity__(cc)))
    # In the custom bootstrapping, we consider the
    #   relationships of 4, having 4 or 3 nodes,
    #   relationships of 3, having 3 or 2 nodes, &
    #   relationships of 2, having 2 nodes.
    # Group dict is a priority list where key 1 has the connected components
    # with highest priority and key 5 has the connected components with
    # lowest priority.
    #
    else:
      for cc in nx.strongly_connected_components(self.G):

        a_node = self.G.nodes[next(iter(cc))]

        # Skip atomic nodes and single relational nodes
        if a_node[c.TYPE1] == c.N_ATOMIC or len(cc) == 1:
          continue

        # Getting true cc size
        #
        node1_type = a_node[c.TYPE2]

        # Filter the groups of 4 relationships
        # Consider only the links having 4, 3 nodes
        if node1_type in ['Mm-Dd', 'Mm-Ds', 'Mbp-Dp', 'Mgp-Dp']:
          if len(cc) == 4:
            group_dict[1].append((cc, __get_group_avg_similarity__(cc)))
          elif len(cc) == 3:
            group_dict[2].append((cc, __get_group_avg_similarity__(cc)))


        # Filter the groups of 3 relationships
        # Consider only the links having 3,2 nodes
        elif node1_type in ['Bb-Mm', 'Bp-Mbp', 'Bp-Mgp', 'Bb-Dd', 'Bp-Dp',
                            'Mm-Mm', 'Mbp-Mbp', 'Mgp-Mgp']:
          if len(cc) == 3:
            group_dict[3].append((cc, __get_group_avg_similarity__(cc)))
          if len(cc) == 2:
            group_dict[4].append((cc, __get_group_avg_similarity__(cc)))

        # Filter the groups of 2 relationships
        # Consider only the links having 2 nodes
        elif node1_type in ['Bp-Bp', 'Bp-Mm',
                            'Bp-Dd', 'Bp-Ds', 'Mm-Mbp', 'Mm-Mgp', 'Mbp-Dp',
                            'Mgp-Dp', 'Mm-Ds', 'Mm-Dd', 'Mm-Dp', 'Mbp-Dd',
                            'Mbp-Ds', 'Mgp-Dd', 'Mgp-Ds', 'Dd-Ds', 'Dd-Dp',
                            'Ds-Dp', 'Dp-Dp']:
          if len(cc) == 2:
            group_dict[5].append((cc, __get_group_avg_similarity__(cc)))

    is_valid_merge = self.__is_valid_node_merge__
    merge_nodes = self.__merge_nodes__
    SIM = c.SIM

    for bootstrap_threshold in bootstrap_threshold_list:
      # Iterate the groups based on the priority in the similarity order
      for priority in [1, 2, 3, 4, 5]:
        for node_id_set, sim in \
            sorted(group_dict[priority], key=lambda a: a[1], reverse=True):

          logging.debug('Node id set {} with avg sim {}'.format(node_id_set,
                                                                sim))
          valid_nodes = True
          sim_list = list()
          for node_id in node_id_set:

            node = nodes[node_id]
            o1, o2 = enrich_node(node_id, node, attributes_meta.people_sim_func)

            if __is_valid_entity_role_merge__(o1, o2, node_id_set,
                                              people_dict,
                                              relationship_nodes, nodes,
                                              merge_threshold):
              sim_list.append(node[SIM])
            else:
              valid_nodes = False
              break

          if valid_nodes:
            s = sum(sim_list) * 1.0 / len(sim_list)
            if s >= bootstrap_threshold and \
                min(sim_list) >= (bootstrap_threshold - 0.2):
              if is_valid_merge(node_id_set):
                merge_nodes(node_id_set)

    print 'Bootstrap added nodes %s' % stats.added_neighbors_count
    if len(stats.added_group_size) > 0:
      print 'Bootstrap added group size avg %s' % (sum(
        stats.added_group_size) * 1.0 / len(stats.added_group_size))
      print 'Bootstrap added node group max %s' % (max(stats.added_group_size))
    logging.info('Bootstrapping finished')

  def link(self, merge_threshold):
    self.link_gdg(merge_threshold, attributes_meta.people_sim_func)

  def refine(self, merge_threshold):

    logging.info('REFINING STEP')
    relationship_nodes = self.relationship_nodes
    nodes = self.G.nodes
    is_valid_merge = self.__is_valid_node_merge__
    merge_nodes = self.__merge_nodes__
    enrich_node = self.__enrich_node__

    merged_node_count = 0
    refined_count = 0
    print merged_node_count

    for id, entity in self.entity_dict.items():
      for role in [c.MOTHER, c.FATHER]:
        person_list = list(entity[role])

        for i in xrange(len(person_list)):
          for j in xrange(i + 1, len(person_list)):

            key = (person_list[i], person_list[j])
            logging.debug('Entity {} relation {} : {}'
                          .format(id, role, '-'.join(str(key))))
            node_id = None
            if key in relationship_nodes:
              node_id = relationship_nodes[key]
            elif (key[1], key[0]) in self.relationship_nodes:
              node_id = relationship_nodes[(key[1], key[0])]
            if node_id is not None:
              node = nodes[node_id]
              if node[c.STATE] != State.MERGED:
                enrich_node(node_id, node, attributes_meta.people_sim_func)
                # calculate_sim(node_id, node, alpha)
                # if node[c.SIM] >= merge_threshold:
                if is_valid_merge([node_id]):
                  logging.debug('--- Is valid merge')
                  refined_count += 1
                  merge_nodes([node_id])
                else:
                  logging.debug('--- NOT Is valid merge')

      for role in [c.SPOUSE, c.CHILDREN]:
        person_list = list(entity[role])

        for i in xrange(len(person_list)):
          for j in xrange(i + 1, len(person_list)):

            key = (person_list[i], person_list[j])
            logging.debug('Entity {} relation {} : {}'
                          .format(id, role, '-'.join(str(key))))
            node_id = None
            if key in relationship_nodes:
              node_id = relationship_nodes[key]
            elif (key[1], key[0]) in self.relationship_nodes:
              node_id = relationship_nodes[(key[1], key[0])]
            if node_id is not None:
              node = nodes[node_id]
              if node[c.STATE] != State.MERGED:
                enrich_node(node_id, node, attributes_meta.people_sim_func)
                if node[c.SIM_ATOMIC] >= merge_threshold:
                  if is_valid_merge([node_id]):
                    logging.debug('--- Is valid merge')
                    refined_count += 1
                    merge_nodes([node_id])

    print 'Refined merged count {}'.format(refined_count)

  def __merge_nodes__(self, node_id_set):

    sim_attr_list = self.attributes.attr_index_list
    I_PID, I_ROLE, I_DATE, I_TIMESTAMP, I_MOTHER, I_FATHER, I_SPOUSE, \
    I_CHILDREN = c.I_ID, c.I_ROLE, c.I_EDATE, c.I_TIMESTAMP, c.I_MOTHER, \
                 c.I_FATHER, c.I_SPOUSE, c.I_CHILDREN

    logging.debug('--- Merging group {}'.format(node_id_set))

    for node_id in node_id_set:

      node = self.G.nodes[node_id]

      if c.SIM_REL not in node.iterkeys():
        node[c.SIM_REL] = 0.0

      if node[c.STATE] == State.MERGED:
        continue
      node[c.STATE] = State.MERGED

      logging.debug('--- Merging node {}-{}'.format(node[c.R1], node[c.R2]))

      p1 = self.record_dict[node[c.R1]]
      p2 = self.record_dict[node[c.R2]]

      self.__update_singleton_list__(node)

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

        # Update death year and exact birth year
        if p1[I_DATE] is not None:
          if p1[I_ROLE] == 'Bb':
            entity[c.EXACT_BIRTH_YEAR] = p1[I_DATE].year
          elif p1[I_ROLE] == 'Dd':
            entity[c.DEATH_YEAR] = p1[I_DATE].year

        if p2[I_DATE] is not None:
          if p2[I_ROLE] == 'Bb':
            entity[c.EXACT_BIRTH_YEAR] = p2[I_DATE].year
          elif p2[c.I_ROLE] == 'Dd':
            entity[c.DEATH_YEAR] = p2[I_DATE].year

        # Update min max event years
        entity[c.MIN_EY] = util.get_min_year(p1[I_DATE], p2[I_DATE])
        entity[c.MAX_EY] = util.get_max_year(p1[I_DATE], p2[I_DATE])

        for sim_attr in sim_attr_list:
          entity[sim_attr] = util.retrieve_merged(p1[sim_attr], p2[sim_attr])

        # Aggregate the relationship PID s
        entity[c.MOTHER] = util.retrieve_merged(p1[I_MOTHER], p2[I_MOTHER])
        entity[c.FATHER] = util.retrieve_merged(p1[I_FATHER], p2[I_FATHER])
        entity[c.CHILDREN] = util.retrieve_merged(p1[I_CHILDREN],
                                                  p2[I_CHILDREN])
        entity[c.SPOUSE] = util.retrieve_merged(p1[I_SPOUSE], p2[I_SPOUSE])

        # Each merged person has a dictionary of roles.
        # Each role has a list of tuples where each tuple corresponds to
        # the ID, timestamp and certificate ID corresponding to the person
        # in that role.
        entity[c.ROLES] = defaultdict(set)
        entity[c.ROLES][p1[I_ROLE]].add((p1[I_PID], p1[I_TIMESTAMP],
                                         p1[c.I_CERT_ID]))
        entity[c.ROLES][p2[I_ROLE]].add((p2[I_PID], p2[I_TIMESTAMP],
                                         p2[c.I_CERT_ID]))

        # Trace of entity evolution for ANALYSIS
        entity[c.TRACE].append(analysis.get_trace(node, p1, p2, True, 'RM_E'))

        self.entity_dict[self.entity_id] = entity
        p1[c.I_ENTITY_ID] = self.entity_id
        p2[c.I_ENTITY_ID] = self.entity_id

        # Add entity graph
        p_node1 = '%s-%s' % (p1[I_PID], p1[I_ROLE])
        p_node2 = '%s-%s' % (p2[I_PID], p2[I_ROLE])
        entity[c.GRAPH].add_edge(p_node1, p_node2, sim=round(node[c.SIM], 2),
                                 amb=round(node[c.SIM_AMB], 2),
                                 attr=round(node[c.SIM_ATTR], 2))

      # If both people are merged
      elif p1[c.I_ENTITY_ID] is not None and p2[c.I_ENTITY_ID] is not None:

        # Already merged
        if p1[c.I_ENTITY_ID] == p2[c.I_ENTITY_ID]:
          # Add edge to entity graph
          p_node1 = '%s-%s' % (p1[I_PID], p1[I_ROLE])
          p_node2 = '%s-%s' % (p2[I_PID], p2[I_ROLE])
          entity = self.entity_dict[p1[c.I_ENTITY_ID]]
          entity[c.GRAPH].add_edge(p_node1, p_node2, sim=round(node[c.SIM], 2),
                                   amb=round(node[c.SIM_AMB], 2),
                                   attr=round(node[c.SIM_ATTR], 2))

          self.linkage_reason_dict[node[c.TYPE2]][
            (node[c.R1], node[c.R2])] = 'RM_M'
          continue

        entity1 = self.entity_dict[p1[c.I_ENTITY_ID]]
        entity2 = self.entity_dict[p2[c.I_ENTITY_ID]]

        entity1[c.SEX] = util.get_node_gender(entity1[c.SEX], entity2[c.SEX])
        if entity1[c.SEX] == 0:
          if hyperparams.scenario.startswith('noprop'):
            return
          raise Exception('Different genders')
        # Delete entity 2 record
        del self.entity_dict[p2[c.I_ENTITY_ID]]

        # Aggregate the atomic attribute values
        for attribute in sim_attr_list:
          entity1[attribute] = entity1[attribute].union(entity2[attribute])

        # Aggregate the relationship PID s
        for relationship in Edge_Type.list():
          entity1[relationship] = entity1[relationship].union(
            entity2[relationship])

        for role in Role.list():
          entity1[c.ROLES][role] = \
            entity1[c.ROLES][role].union(entity2[c.ROLES][role])

          # Update entity_id for all people in entity2
          for record in entity2[c.ROLES][role]:
            pid = record[0]
            self.record_dict[pid][c.I_ENTITY_ID] = entity1[c.ENTITY_ID]

        # Update death year
        if entity2[c.DEATH_YEAR] is not None:
          entity1[c.DEATH_YEAR] = entity2[c.DEATH_YEAR]

        # Update birth year
        if entity2[c.EXACT_BIRTH_YEAR] is not None:
          entity1[c.EXACT_BIRTH_YEAR] = entity2[c.EXACT_BIRTH_YEAR]

        # Update min max event years
        entity1[c.MAX_EY] = max([entity1[c.MAX_EY], entity2[c.MAX_EY]])
        entity1[c.MIN_EY] = min([entity1[c.MIN_EY], entity2[c.MIN_EY]])

        # Trace of entity evolution for ANALYSIS
        entity1[c.TRACE].append(analysis.get_trace(node, p1, p2, True, 'RM_EE'))
        entity1[c.TRACE].append(analysis.get_trace(node, p1, p2, True, 'RM_EE'))
        entity1[c.TRACE].append(entity2[c.TRACE])

        # Merge entity graphs
        for edge_node1, edge_node2, edge_data in entity2[c.GRAPH].edges.data():
          entity1[c.GRAPH].add_edge(edge_node1, edge_node2, **edge_data)
        p_node1 = '%s-%s' % (p1[I_PID], p1[I_ROLE])
        p_node2 = '%s-%s' % (p2[I_PID], p2[I_ROLE])
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
          if hyperparams.scenario.startswith('noprop'):
            return
          raise Exception(' Different genders')

        if non_merged[I_DATE] is not None:
          # Update death year
          if non_merged[c.I_ROLE] == 'Dd':
            entity[c.DEATH_YEAR] = non_merged[I_DATE].year

          # Update birth year
          if non_merged[c.I_ROLE] == 'Bb':
            entity[c.EXACT_BIRTH_YEAR] = non_merged[I_DATE].year

          # Update min max event years
          entity[c.MIN_EY] = min([entity[c.MAX_EY], non_merged[I_DATE].year])
          entity[c.MAX_EY] = max([entity[c.MAX_EY], non_merged[I_DATE].year])

        # Aggregate the atomic attribute values
        for attribute in sim_attr_list:
          if non_merged[attribute] is not None:
            entity[attribute].add(non_merged[attribute])

        entity[c.ROLES][non_merged[c.I_ROLE]].add((non_merged[I_PID],
                                                   non_merged[I_TIMESTAMP],
                                                   non_merged[c.I_CERT_ID]))

        if non_merged[c.I_MOTHER] is not None:
          entity[c.MOTHER].add(non_merged[I_MOTHER])
        if non_merged[c.I_FATHER] is not None:
          entity[c.FATHER].add(non_merged[I_FATHER])
        if non_merged[c.I_SPOUSE] is not None:
          entity[c.SPOUSE].add(non_merged[I_SPOUSE])
        if non_merged[c.I_CHILDREN] is not None:
          entity[c.CHILDREN].add(non_merged[I_CHILDREN])

        # Trace of entity evolution for ANALYSIS
        entity[c.TRACE].append(analysis.get_trace(node, p1, p2, True, 'RM_EP'))

        p1[c.I_ENTITY_ID] = entity[c.ENTITY_ID]
        p2[c.I_ENTITY_ID] = entity[c.ENTITY_ID]

        # Add edge into entity graph
        p_node1 = '%s-%s' % (p1[I_PID], p1[I_ROLE])
        p_node2 = '%s-%s' % (p2[I_PID], p2[I_ROLE])
        entity[c.GRAPH].add_edge(p_node1, p_node2, sim=round(node[c.SIM], 2),
                                 amb=round(node[c.SIM_AMB], 2),
                                 attr=round(node[c.SIM_ATTR], 2))

      ## ANALYSIS
      # if node[c.TYPE2] == 'Bb-Dd' and analysis.is_ground_truth_link(p1, p2,
      #                                                               'Bb-Dd'):
      #   analysis.bb_dd_tp_list.add(node_id)
      logging.debug('--- Merged succesfully')

  def refine_gt_measures(self):

    # Local stats
    removed_edge_count = stats.removed_edge_count
    split_entity_count_d = stats.split_entity_count_d

    people_dict = self.record_dict

    # Bridges
    for id, entity in self.entity_dict.copy().iteritems():
      g = entity['g']

      if g.number_of_nodes() < hyperparams.bridges_n:
        continue

      bridge_list = list(nx.bridges(g))
      if len(bridge_list) == 0:
        continue

      # entity_analysis.visualize_entity(entity, people_dict)
      for bridge in bridge_list:

        g.remove_edge(*bridge)
        stats.removed_edge_count += 1

        # ANALYSIS
        p1_id = int(bridge[0].split('-')[0])
        p2_id = int(bridge[1].split('-')[0])
        if (p1_id, p2_id) in self.relationship_nodes:
          tmp_node = self.G.nodes[self.relationship_nodes[(p1_id, p2_id)]]
          analysis.reason_dict[tmp_node[c.TYPE2]][
            (tmp_node[c.R1], tmp_node[c.R2])] = 'S_B'

      stats.split_entity_count_b += 1
      del self.entity_dict[id]
      gender = entity[c.SEX]
      for sub_graph in nx.connected_component_subgraphs(g):
        if len(sub_graph) > 1:
          self.__add_new_entity__(sub_graph, gender)
        else:
          for pid in sub_graph.nodes():
            pid = int(pid.split('-')[0])
            person = people_dict[pid]
            person[c.I_ENTITY_ID] = None

    # Connectivity
    for id, entity in self.entity_dict.copy().iteritems():
      g = entity['g']

      if g.number_of_nodes() <= 3:
        continue

      density = nx.density(g) * 100.0

      if density <= 30.0:
        # entity_analysis.visualize_entity(entity, people_dict)

        sorted_node_list = sorted(g.degree, key=lambda x: x[1])

        if sorted_node_list[0][1] < sorted_node_list[1][1]:
          g.remove_node(sorted_node_list[0][0])
          removed_person = people_dict[
            int(sorted_node_list[0][0].split('-')[0])]
          removed_person[c.I_ENTITY_ID] = None

          stats.split_entity_count_d += 1
          del self.entity_dict[id]
          self.__add_new_entity__(g, entity[c.SEX])

    removed_edge_count = stats.removed_edge_count - removed_edge_count
    split_entity_count_d = stats.split_entity_count_d - split_entity_count_d

    logging.info('Removed edge count {}'.format(removed_edge_count))
    logging.info('Split entity count {}'.format(split_entity_count_d))
    logging.info('Refining gt measures finished')

  def __is_valid_node_merge__(self, node_id_set):

    nodes = self.G.nodes
    people_dict = self.record_dict

    if hyperparams.scenario.startswith('noprop'):
      for node_id in node_id_set:
        node = nodes[node_id]

        p1 = people_dict[node[c.R1]]
        p2 = people_dict[node[c.R2]]

        # Validate individual gender
        if p1[c.I_SEX] != p2[c.I_SEX]:
          return False

      return True

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
      if not self.__is_valid_link__(node):
        # update_reason(node_id_set, nodes, 'RP_S')
        return False

      # Validate individual gender
      gender = util.get_node_gender(p1[c.I_SEX], p2[c.I_SEX])
      if gender == 0:
        # update_reason(node_id_set, nodes, 'RP_G')
        return False

      # Validate birth years
      if p1[c.I_BIRTH_YEAR] is not None and p2[c.I_BIRTH_YEAR] is not None:
        if not is_almost_same_years(p1[c.I_BIRTH_YEAR], p2[c.I_BIRTH_YEAR]):
          # update_reason(node_id_set, nodes, 'RP_BY')
          return False

      if p1[c.I_ROLE] != p2[c.I_ROLE]:
        if p1[c.I_CERT_ID] == p2[c.I_CERT_ID]:
          return False

      # Validate entity based constraints
      #
      # If both people are merged
      if p1[c.I_ENTITY_ID] is not None and p2[c.I_ENTITY_ID] is not None:

        if p1[c.I_ENTITY_ID] == p2[c.I_ENTITY_ID]:
          continue

        logging.debug('--- Both have entities')
        entity1 = self.entity_dict[p1[c.I_ENTITY_ID]]
        entity2 = self.entity_dict[p2[c.I_ENTITY_ID]]

        # Gender constraint
        gender = util.get_node_gender(entity1[c.SEX], entity2[c.SEX])
        if gender == 0:
          # update_reason(node_id_set, nodes, 'RE2_G')
          # Trace of entity evolution for ANALYSIS
          entity1[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
                                                     'RE2_G'))
          entity2[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
                                                     'RE2_G'))
          return False

        # Singleton Bb, Dd per entity constraint
        if len(entity1[c.ROLES]['Bb']) >= 1 and \
            len(entity2[c.ROLES]['Bb']) >= 1:
          if next(iter(entity1[c.ROLES]['Bb']))[0] != \
              next(iter(entity2[c.ROLES]['Bb']))[0]:
            # update_reason(node_id_set, nodes, 'RE2_SBb')
            # Trace of entity evolution for ANALYSIS
            # entity1[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
            #                                            'RE2_SBb'))
            # entity2[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
            #                                            'RE2_SBb'))
            return False

        if len(entity1[c.ROLES]['Dd']) >= 1 and \
            len(entity2[c.ROLES]['Dd']) >= 1:
          if next(iter(entity1[c.ROLES]['Dd']))[0] != \
              next(iter(entity2[c.ROLES]['Dd']))[0]:
            # update_reason(node_id_set, nodes, 'RE2_SDd')
            # Trace of entity evolution for ANALYSIS
            # entity1[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
            #                                            'RE2_SDd'))
            # entity2[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
            #                                            'RE2_SDd'))
            return False

        # Same certificate ids cannot appear as different roles
        # For example, if the same cert id appear as Dd and Dp, these are two
        # different entities of a parent and child
        for role in Role.list():
          other_roles = filter(lambda x: x != role, Role.list())

          e1_role_cert_id_set = {certid for _, _, certid in
                                 entity1[c.ROLES][role]}
          e2_other_role_cert_id_set = {certid for role2 in other_roles for
                                       _, _, certid in entity2[c.ROLES][role2]}

          if len(
              e1_role_cert_id_set.intersection(e2_other_role_cert_id_set)) > 0:
            return False

        ## Temporal constraints

        # Validate birth years
        if entity1[c.EXACT_BIRTH_YEAR] is not None and \
            entity2[c.EXACT_BIRTH_YEAR] is not None:
          if not util.is_almost_same_years(entity1[c.EXACT_BIRTH_YEAR],
                                           entity2[c.EXACT_BIRTH_YEAR]):
            # update_reason(node_id_set, nodes, 'RE2_BY')
            # Trace of entity evolution for ANALYSIS
            # cause = ' {} {}'.format(entity1[c.EXACT_BIRTH_YEAR],
            #                         entity2[c.EXACT_BIRTH_YEAR])
            # entity1[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
            #                                            'RE2_BY', cause))
            # entity2[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
            #                                            'RE2_BY', cause))
            return False
        elif entity1[c.EXACT_BIRTH_YEAR] is not None and \
            entity2[c.EXACT_BIRTH_YEAR] is None:
          if entity1[c.EXACT_BIRTH_YEAR] > entity2[c.MIN_EY]:
            # update_reason(node_id_set, nodes, 'RE2_BY')
            # cause = ' {} {}'.format(entity1[c.EXACT_BIRTH_YEAR],
            #                         entity2[c.EXACT_BIRTH_YEAR])
            # entity1[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
            #                                            'RE2_BY', cause))
            # entity2[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
            #                                            'RE2_BY', cause))
            return False
        elif entity1[c.EXACT_BIRTH_YEAR] is None and \
            entity2[c.EXACT_BIRTH_YEAR] is not None:
          if entity2[c.EXACT_BIRTH_YEAR] > entity1[c.MIN_EY]:
            # update_reason(node_id_set, nodes, 'RE2_BY')
            # cause = ' {} {}'.format(entity1[c.EXACT_BIRTH_YEAR],
            #                         entity2[c.EXACT_BIRTH_YEAR])
            # entity1[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
            #                                            'RE2_BY', cause))
            # entity2[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
            #                                            'RE2_BY', cause))
            return False

        # Validate death years
        if entity1[c.DEATH_YEAR] is not None and \
            entity2[c.DEATH_YEAR] is not None:
          if not util.is_almost_same_years(entity1[c.DEATH_YEAR],
                                           entity2[c.DEATH_YEAR]):
            # update_reason(node_id_set, nodes, 'RE2_DY')
            return False
        elif entity1[c.DEATH_YEAR] is not None and \
            entity2[c.DEATH_YEAR] is None:
          if entity1[c.DEATH_YEAR] < entity2[c.MAX_EY]:
            # update_reason(node_id_set, nodes, 'RE2_DY')
            return False
        elif entity1[c.DEATH_YEAR] is None and \
            entity2[c.DEATH_YEAR] is not None:
          if entity2[c.DEATH_YEAR] < entity1[c.MAX_EY]:
            # update_reason(node_id_set, nodes, 'RE2_DY')
            return False

        for e1_birth_year in entity1[c.I_BIRTH_YEAR]:

          # Birth years should align
          for e2_birth_year in entity2[c.I_BIRTH_YEAR]:
            if not util.is_almost_same_years(e1_birth_year, e2_birth_year):
              # update_reason(node_id_set, nodes, 'RE2_BY')
              # Trace of entity evolution for ANALYSIS
              # cause = ' {} {}'.format(e1_birth_year, e2_birth_year)
              # entity1[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
              #                                            'RE2_BY', cause))
              # entity2[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
              #                                            'RE2_BY', cause))
              return False

        for e1_marriage_year in entity1[c.I_MARRIAGE_YEAR]:

          # Marriage years should align
          for e2_marriage_year in entity2[c.I_MARRIAGE_YEAR]:
            if not util.is_almost_same_marriage_years(e1_marriage_year,
                                                      e2_marriage_year):
              # Trace of entity evolution for ANALYSIS
              # cause = ' {} {}'.format(e1_marriage_year, e2_marriage_year)
              # entity1[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
              #                                            'RE2_MY', cause))
              # entity2[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
              #                                            'RE2_MY', cause))
              return False

      # If single person is merged and the other is not
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

        # Singleton Bb, Dd per entity constraint
        if non_merged[c.I_ROLE] in ['Bb', 'Dd']:
          if len(entity[c.ROLES][non_merged[c.I_ROLE]]) >= 1:
            # Trace of entity evolution for ANALYSIS
            # cause = 'RE1_S{}'.format(non_merged[c.I_ROLE])
            # update_reason(node_id_set, nodes, cause)
            # entity[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
            #                                           cause))
            return False

        # Same certificate ids cannot appear as different roles
        # For example, if the same cert id appear as Dd and Dp, these are two
        # different entities of a parent and child
        other_roles = filter(lambda x: x != non_merged[c.I_ROLE], Role.list())
        entity_role_cert_id_set = {certid for role2 in other_roles for
                                   _, _, certid in entity[c.ROLES][role2]}

        if non_merged[c.I_CERT_ID] in entity_role_cert_id_set:
          return False

        non_merged_by = non_merged[c.I_BIRTH_YEAR]
        non_merged_my = non_merged[c.I_MARRIAGE_YEAR]

        if non_merged[c.I_EDATE] is not None:

          # Validate exact birth years
          if entity[c.EXACT_BIRTH_YEAR] is not None:
            if non_merged[c.I_EDATE].year < entity[c.EXACT_BIRTH_YEAR]:
              # update_reason(node_id_set, nodes, 'RE1_BY')
              # Trace of entity evolution for ANALYSIS
              # entity[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
              #                                           'RE1_BY'))
              return False
          if non_merged[c.I_ROLE] == 'Bb':
            if non_merged[c.I_EDATE].year > entity[c.MIN_EY]:
              # update_reason(node_id_set, nodes, 'RE1_BY')
              # Trace of entity evolution for ANALYSIS
              # entity[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
              #                                           'RE1_BY'))
              return False

          # Validate death years
          if entity[c.DEATH_YEAR] is not None:
            if non_merged[c.I_EDATE].year > entity[c.DEATH_YEAR]:
              # update_reason(node_id_set, nodes, 'RE1_DY')
              return False

        for e1_birth_year in entity[c.I_BIRTH_YEAR]:

          if non_merged[c.I_BIRTH_YEAR] is not None:
            if not util.is_almost_same_years(e1_birth_year, non_merged_by):
              # update_reason(node_id_set, nodes, 'RE1_BY')
              # Trace of entity evolution for ANALYSIS
              # cause = ' {} {}'.format(e1_birth_year, non_merged_by)
              # entity[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
              #                                           'RE1_BY', cause))
              return False

        for e1_marriage_year in entity[c.I_MARRIAGE_YEAR]:

          if non_merged[c.I_MARRIAGE_YEAR] is not None:
            # Marriage years should align
            if not util.is_almost_same_marriage_years(e1_marriage_year,
                                                      non_merged_my):
              # Trace of entity evolution for ANALYSIS
              cause = ' {} {}'.format(e1_marriage_year, non_merged_my)
              entity[c.TRACE].append(analysis.get_trace(node, p1, p2, False,
                                                        'RE1_MY', cause))
              return False

    return True

  def __is_valid_link__(self, node):
    """
    Function to validate a link based on the singleton link constraints.
    For a particular person, a singleton link can occur only once.
    For example, for a Bb, there can only be a single Dd reference linked.
    This function checks whether the link in this node is still valid due to
    the constraints from the singleton links.
    :param node: Node
    :return: True/False
    """

    # Retrieve the relationship link string
    link = node[c.TYPE2]

    # Check if the relationship link is a singleton.
    if link in Links.get_singleton_links():

      p1_role = self.record_dict[node[c.R1]][c.I_ROLE]
      # Get the ID of the person other than 'Bb' or 'Dd'
      if p1_role in ['Bb', 'Dd']:
        candidate_person_pid = node[c.R2]
      else:
        candidate_person_pid = node[c.R1]

      # Check if the relationship is already merged
      if candidate_person_pid in self.singleton_links_dict[link]:
        return False
    return True

  def __is_valid_entity_role_merge__(self, o1, o2, node_id_set, people_dict,
                                     relationship_nodes, nodes, merge_t,
                                     node_id_set_queue=None):
    if hyperparams.scenario.startswith('noprop'):
      return True

    __is_valid_existing_node__ = self.__is_valid_existing_node__
    __is_valid_non_existing_pair__ = self.__is_valid_non_existing_pair__
    __update_reason__ = self.__update_reason__

    ROLES, MOTHER, FATHER, CHILDREN, SPOUSE = c.ROLES, c.MOTHER, c.FATHER, c.CHILDREN, c.SPOUSE

    I_ROLE, I_MOTHER, I_FATHER, I_CHILDREN, I_SPOUSE = c.I_ROLE, c.I_MOTHER, \
                                                       c.I_FATHER, c.I_CHILDREN, c.I_SPOUSE
    I_SEX = c.I_SEX

    existing_nodes_set = set()
    non_e_nodes_set = set()
    roles = Role.list()
    links = Links.list()

    # If both o1 and o2 are entities
    if type(o1) == dict and type(o2) == dict:

      if o1[c.ENTITY_ID] == o2[c.ENTITY_ID]:
        return True

      # Iterate through the roles of each entity
      for role1 in roles:
        if len(o1[ROLES][role1]) == 0:
          continue

        for role2 in roles:
          if len(o2[ROLES][role2]) == 0:
            continue

          link1 = '%s-%s' % (role1, role2)
          link2 = '%s-%s' % (role2, role1)

          if link1 in links or link2 in links:

            # Iterate through the person records assigned to each role
            for pid_1, _, _ in o1[ROLES][role1]:
              for pid_2, _, _ in o2[ROLES][role2]:

                if pid_1 == pid_2:
                  continue
                if (pid_1, pid_2) in relationship_nodes:
                  node_id = relationship_nodes[(pid_1, pid_2)]
                  if node_id in node_id_set or node_id in existing_nodes_set:
                    continue
                  else:
                    node = nodes[node_id]
                    if not __is_valid_existing_node__(node_id, node):
                      # __update_reason__(node_id_set, nodes, 'RER-SELF-E')
                      return False
                    existing_nodes_set.add(node_id)
                else:
                  link = link1 if link1 in links else link2
                  if (pid_1, pid_2, link) in non_e_nodes_set \
                      or (pid_2, pid_1, link) in non_e_nodes_set:
                    continue
                  else:
                    person1 = people_dict[pid_1]
                    person2 = people_dict[pid_2]
                    if not __is_valid_non_existing_pair__(person1, person2,
                                                          link, nodes, merge_t,
                                                          node_id_set_queue):
                      # __update_reason__(node_id_set, nodes, 'RER-SELF-NE')
                      return False
                    non_e_nodes_set.add((pid_1, pid_2, link))

    ## Neighbor node pairs
    #
    if type(o1) == dict:
      m1, f1, c1, s1 = o1[MOTHER], o1[FATHER], o1[CHILDREN], o1[SPOUSE]
    else:
      m1, f1, c1, s1 = {o1[I_MOTHER]}, {o1[I_FATHER]}, {o1[I_CHILDREN]}, \
                       {o1[I_SPOUSE]}

    if type(o2) == dict:
      m2, f2, c2, s2 = o2[MOTHER], o2[FATHER], o2[CHILDREN], o2[SPOUSE]
    else:
      m2, f2, c2, s2 = {o2[I_MOTHER]}, {o2[I_FATHER]}, {o2[I_CHILDREN]}, \
                       {o2[I_SPOUSE]}

    # Validate only mother and father neighbors
    # If we validate children and spouses, results goes down.
    # The reason is because children and spouses can be different by default.
    # Only mothers and fathers should be the same.
    for p1_list, p2_list, rel_type in [(m1, m2, 'M'), (f1, f2, 'F')]:

      # A copy is used because the method __is_valid_non_existing_pair__
      # can update the set we are iterating
      for p1 in p1_list.copy():
        if p1 is None:
          continue

        for p2 in p2_list.copy():

          if p2 is None or p1 == p2:
            continue

          if (p1, p2) in relationship_nodes:
            node_id = relationship_nodes[(p1, p2)]
            if (node_id in node_id_set) or (node_id in existing_nodes_set):
              continue
            else:
              node = nodes[node_id]
              if not __is_valid_existing_node__(node_id, node):
                # __update_reason__(node_id_set, nodes, 'RER-%s-E' % rel_type)
                return False
              existing_nodes_set.add(node_id)
          else:
            person1 = people_dict[p1]
            person2 = people_dict[p2]

            link1 = '%s-%s' % (person1[I_ROLE], person2[I_ROLE])
            link2 = '%s-%s' % (person2[I_ROLE], person1[I_ROLE])

            if link1 in links or link2 in links:
              link = link1 if link1 in links else link2
              if (p1, p2, link) in non_e_nodes_set \
                  or (p2, p1, link) in non_e_nodes_set:
                continue
              else:
                if not __is_valid_non_existing_pair__(person1, person2, link,
                                                      nodes, merge_t,
                                                      node_id_set_queue):
                  # __update_reason__(node_id_set, nodes, 'RER-%s-NE' % rel_type)
                  return False
                non_e_nodes_set.add((p1, p2, link))

    ## Add non existing nodes for spouse pairs and child pairs

    # Children
    for p1 in c1.copy():
      if p1 is None:
        continue

      for p2 in c2.copy():
        if p2 is None:
          continue

        if (p1, p2) in relationship_nodes:
          continue
        else:
          child1 = people_dict[p1]
          child2 = people_dict[p2]

          if child1[I_SEX] == child2[I_SEX]:
            link1 = '%s-%s' % (child1[I_ROLE], child2[I_ROLE])
            link2 = '%s-%s' % (child2[I_ROLE], child1[I_ROLE])

            if link1 in links or link2 in links:
              link = link1 if link1 in links else link2
              if (p1, p2, link) in non_e_nodes_set \
                  or (p2, p1, link) in non_e_nodes_set:
                continue
              else:
                # Only to add the node
                __is_valid_non_existing_pair__(child1, child2, link,
                                               nodes, merge_t,
                                               node_id_set_queue)
                non_e_nodes_set.add((p1, p2, link))

    # Spouse
    for p1 in s1.copy():
      if p1 is None:
        continue

      for p2 in s2.copy():

        if p2 is None or p1 == p2:
          continue

        if (p1, p2) in relationship_nodes:
          continue
        else:
          person1 = people_dict[p1]
          person2 = people_dict[p2]

          link1 = '%s-%s' % (person1[I_ROLE], person2[I_ROLE])
          link2 = '%s-%s' % (person2[I_ROLE], person1[I_ROLE])

          if link1 in links or link2 in links:
            link = link1 if link1 in links else link2
            if (p1, p2, link) in non_e_nodes_set \
                or (p2, p1, link) in non_e_nodes_set:
              continue
            else:
              # Only to add the node
              __is_valid_non_existing_pair__(person1, person2, link,
                                             nodes, merge_t, node_id_set_queue)
              non_e_nodes_set.add((p1, p2, link))

    return True

  def __is_valid_non_existing_pair__(self, person1, person2, link, nodes,
                                     merge_threshold, node_id_set_queue=None):
    """
    Checks if a pair of person records are valid to be added as a node.
    If valid the node is added.

    :param person1: Person 1
    :param person2: Person 2
    :param link: Link
    :param nodes: Set of nodes in the graph
    :param merge_threshold: Merge threshold
    :return: True if the pair is valid
    """
    if self.__is_valid_node__(person1, person2, link):
      new_node_id = self.__add_single_node_no_constraints__(person1, person2,
                                                            link)

      if new_node_id is not None:

        stats.added_neighbors_count += 1
        new_node = nodes[new_node_id]

        # todo : consider merging for this poor fellow
        self.linkage_reason_dict[new_node[c.TYPE2]][
          (new_node[c.R1], new_node[c.R2])] = 'NEW'

        self.calculate_sim_atomic(new_node_id, new_node, person1, person2)

        if self.__is_valid_node_merge__({new_node_id}):
          self.__add_neighbor_edges__(new_node_id, person1, person2, nodes,
                                      node_id_set_queue)
          # self.merge_nodes({new_node_id})
          return True

    return False

  def __add_neighbor_edges__(self, node_id, person1, person2, nodes,
                             node_id_set_queue=None):

    add_edge = self.G.add_edge
    relationship_nodes = self.relationship_nodes
    people_dict = self.record_dict
    I_ROLE = c.I_ROLE
    links = Links.list()

    node_id_set = {node_id}

    gender = util.get_node_gender(person1[c.I_SEX], person2[c.I_SEX])

    # Get child relations
    c1 = person1[c.I_CHILDREN]
    c2 = person2[c.I_CHILDREN]
    if c1 is not None and c2 is not None and c1 != c2:
      child1 = people_dict[c1]
      child2 = people_dict[c2]

      if child1[c.I_SEX] == child2[c.I_SEX]:
        if (c1, c2) in relationship_nodes:

          child_node_id = relationship_nodes[(c1, c2)]
          node_id_set.add(child_node_id)

          # Add edge between child-child node and mother-mother/father-father
          # node
          add_edge(child_node_id, node_id, t1=c.E_BOOL_S, t2=Edge_Type.CHILDREN)
          if gender == Sex.F:
            add_edge(node_id, child_node_id, t1=c.E_BOOL_W, t2=Edge_Type.MOTHER)
          elif gender == Sex.M:
            add_edge(node_id, child_node_id, t1=c.E_BOOL_W, t2=Edge_Type.FATHER)

        else:
          link1 = '%s-%s' % (child1[I_ROLE], child2[I_ROLE])
          link2 = '%s-%s' % (child2[I_ROLE], child1[I_ROLE])

          if link1 in links or link2 in links:
            link = link1 if link1 in links else link2

            if self.__is_valid_node__(child1, child2, link):
              child_node_id = \
                self.__add_single_node_no_constraints__(child1, child2, link)
              if child_node_id is not None:

                stats.added_neighbors_count += 1
                node_id_set.add(child_node_id)

                child_node = nodes[child_node_id]
                analysis.reason_dict[child_node[c.TYPE2]][
                  (child_node[c.R1], child_node[c.R2])] = 'NEW'
                self.calculate_sim_atomic(child_node_id, child_node, child1,
                                          child2)

                add_edge(child_node_id, node_id, t1=c.E_BOOL_S,
                         t2=Edge_Type.CHILDREN)

                if gender == Sex.F:
                  add_edge(node_id, child_node_id, t1=c.E_BOOL_W,
                           t2=Edge_Type.MOTHER)
                elif gender == Sex.M:
                  add_edge(node_id, child_node_id, t1=c.E_BOOL_W,
                           t2=Edge_Type.FATHER)

    # Get mother, father, and spouse relations
    for p1, p2, rel_type in [(person1[c.I_MOTHER], person2[c.I_MOTHER], 'm'),
                             (person1[c.I_FATHER], person2[c.I_FATHER], 'f'),
                             (person1[c.I_SPOUSE], person2[c.I_SPOUSE], 's')]:
      if p1 is None or p2 is None or p1 == p2:
        continue
      if (p1, p2) in relationship_nodes:

        rel_node_id = relationship_nodes[(p1, p2)]
        node_id_set.add(rel_node_id)

        if rel_type == 'm':
          add_edge(node_id, rel_node_id, t1=c.E_BOOL_S, t2=Edge_Type.CHILDREN)
          add_edge(rel_node_id, node_id, t1=c.E_BOOL_W, t2=Edge_Type.MOTHER)
        elif rel_type == 'f':
          add_edge(node_id, rel_node_id, t1=c.E_BOOL_S, t2=Edge_Type.CHILDREN)
          add_edge(rel_node_id, node_id, t1=c.E_BOOL_W, t2=Edge_Type.FATHER)
        elif rel_type == 's':
          add_edge(node_id, rel_node_id, t1=c.E_BOOL_W, t2=Edge_Type.SPOUSE)
          add_edge(rel_node_id, node_id, t1=c.E_BOOL_W, t2=Edge_Type.SPOUSE)
      else:

        n_person1 = people_dict[p1]
        n_person2 = people_dict[p2]

        link1 = '%s-%s' % (n_person1[I_ROLE], n_person2[I_ROLE])
        link2 = '%s-%s' % (n_person2[I_ROLE], n_person1[I_ROLE])

        if link1 in links or link2 in links:
          link = link1 if link1 in links else link2

          if self.__is_valid_node__(n_person1, n_person2, link):
            new_node_id = self.__add_single_node_no_constraints__(n_person1,
                                                                  n_person2,
                                                                  link)
            if new_node_id is not None:
              stats.added_neighbors_count += 1
              new_node = nodes[new_node_id]
              node_id_set.add(new_node_id)

              # todo : consider merging for this poor fellow
              analysis.reason_dict[new_node[c.TYPE2]][
                (new_node[c.R1], new_node[c.R2])] = 'NEW'

              self.calculate_sim_atomic(new_node_id, new_node, n_person1,
                                        n_person2)

              if rel_type == 'm':
                add_edge(node_id, new_node_id, t1=c.E_BOOL_S,
                         t2=Edge_Type.CHILDREN)
                add_edge(new_node_id, node_id, t1=c.E_BOOL_W,
                         t2=Edge_Type.MOTHER)
              elif rel_type == 'f':
                add_edge(node_id, new_node_id, t1=c.E_BOOL_S,
                         t2=Edge_Type.CHILDREN)
                add_edge(new_node_id, node_id, t1=c.E_BOOL_W,
                         t2=Edge_Type.FATHER)
              elif rel_type == 's':
                add_edge(node_id, new_node_id, t1=c.E_BOOL_W,
                         t2=Edge_Type.SPOUSE)
                add_edge(new_node_id, node_id, t1=c.E_BOOL_W,
                         t2=Edge_Type.SPOUSE)

    stats.added_group_size.append(len(node_id_set))
    stats.added_node_id_set.append(node_id_set)
    if node_id_set_queue is not None and len(node_id_set) > 1:
      stats.added_node_count += len(node_id_set)
      stats.added_group_count += 1
      node_id_set_queue.append(node_id_set)

  def __is_valid_existing_node__(self, node_id, node):
    """
    Checks if the records in the node are valid to be merged.
    We do not need to merge them or check for the similarity.
    Those links are generated from the entity merges.

    :param node_id: Node id
    :param node: Node
    :param merge_t: Merge threshold
    :return: True if the merge is valid
    """
    if node[c.STATE] == State.MERGED:
      return True
    elif self.__is_valid_node_merge__({node_id}):
      stats.activated_neighbors_count += 1
      return True
    return False

  def __add_new_entity__(self, sub_graph, gender):

    I_DATE = c.I_EDATE
    sim_attr_list = self.attributes.attr_index_list

    self.entity_id += 1
    entity = model.new_entity(self.entity_id, gender)
    entity[c.ROLES] = defaultdict(set)
    for attribute in sim_attr_list + [c.MOTHER, c.FATHER, c.SPOUSE, c.CHILDREN]:
      entity[attribute] = set()
    entity['g'] = sub_graph

    for pid in sub_graph.nodes():
      pid = int(pid.split('-')[0])
      person = self.record_dict[pid]

      if person[I_DATE] is not None:

        # Update death year
        if person[c.I_ROLE] == 'Dd':
          entity[c.DEATH_YEAR] = person[I_DATE].year

        # Update birth year
        if person[c.I_ROLE] == 'Bb':
          entity[c.EXACT_BIRTH_YEAR] = person[I_DATE].year

        # Update min max event years
        if entity[c.MIN_EY] is None:
          entity[c.MIN_EY] = person[I_DATE].year
          entity[c.MAX_EY] = person[I_DATE].year
        else:
          entity[c.MIN_EY] = min([entity[c.MAX_EY], person[I_DATE].year])
          entity[c.MAX_EY] = max([entity[c.MAX_EY], person[I_DATE].year])

      # Aggregate the atomic attribute values
      for attribute in sim_attr_list:
        if person[attribute] is not None:
          entity[attribute].add(person[attribute])

      entity[c.ROLES][person[c.I_ROLE]].add((person[c.I_ID],
                                             person[c.I_TIMESTAMP],
                                             person[c.I_CERT_ID]))

      if person[c.I_MOTHER] is not None:
        entity[c.MOTHER].add(person[c.I_MOTHER])
      if person[c.I_FATHER] is not None:
        entity[c.FATHER].add(person[c.I_FATHER])
      if person[c.I_SPOUSE] is not None:
        entity[c.SPOUSE].add(person[c.I_SPOUSE])
      if person[c.I_CHILDREN] is not None:
        entity[c.CHILDREN].add(person[c.I_CHILDREN])

      self.entity_dict[self.entity_id] = entity
      person[c.I_ENTITY_ID] = self.entity_id

  def __update_singleton_list__(self, node):
    # Retrieve the relationship link string
    link = node[c.TYPE2]

    # Check if the relationship link is a singleton.
    if link in Links.get_singleton_links():
      p1_role = self.record_dict[node[c.R1]][c.I_ROLE]
      # Get the ID of the person other than 'Bb' or 'Dd'
      if p1_role in ['Bb', 'Dd']:
        candidate_person_pid = node[c.R2]
      else:
        candidate_person_pid = node[c.R1]

      self.singleton_links_dict[link].add(candidate_person_pid)

  # ---------------------------------------------------------------------------
  # Ground truth processing
  # ---------------------------------------------------------------------------
  def validate_ground_truth(self, graph_scenario, t):
    merged_person_roles_list = list()
    for merged_person in self.entity_dict.itervalues():
      merged_person_roles = merged_person[c.ROLES]
      p = defaultdict(set)
      p['E_ID'] = merged_person[c.ENTITY_ID]
      p['S'] = merged_person[c.SEX]
      for role in Role.list():
        for role_tuple in merged_person_roles[role]:
          if c.data_set == 'bhic':
            p[role].add(role_tuple[0])  # Appending person ID
          else:
            p[role].add(role_tuple[2])  # Appending certificate ID
      merged_person_roles_list.append(p)

    processed_links_dict = util.enumerate_links(merged_person_roles_list,
                                                self.entity_dict)

    self.calculate_persist_measures(processed_links_dict, ['Bp-Bp', 'Bp-Dp'],
                                    graph_scenario, t, records=self.record_dict)

  def analyse_fp_fn(self, links_list, false_negative_link_dict,
                    false_positive_link_dict, records, secondary_records=None):
    # Analyse false negative reasons
    logging.info('FALSE NEGATIVES ----')
    reason_dict = self.linkage_reason_dict
    fn_reasons_dict = defaultdict(Counter)

    for link in links_list:
      logging.info('FALSE NEGATIVES LINK {}'.format(link))
      for id_1, id_2 in false_negative_link_dict[link]:
        logging.info('\n------------ {} - {}'.format(id_1, id_2))

        for id in [id_1, id_2]:
          p = records[id]
          logging.info(p)
          if p[c.I_MOTHER] is not None:
            logging.info('\t m {}'.format(records[p[c.I_MOTHER]]))
          if p[c.I_FATHER] is not None:
            logging.info('\t f {}'.format(records[p[c.I_FATHER]]))
          if p[c.I_CHILDREN] is not None:
            logging.info('\t c {}'.format(records[p[c.I_CHILDREN]]))
          if p[c.I_SPOUSE] is not None:
            logging.info('\t s {}'.format(records[p[c.I_SPOUSE]]))

    logging.info('FALSE POSITIVES ----')
    for link in links_list:
      logging.info('FALSE POSITIVES LINK {}'.format(link))
      for id_1, id_2 in false_positive_link_dict[link]:
        logging.info('\n------------ {} - {}'.format(id_1, id_2))
        for id in [id_1, id_2]:
          p = records[id]
          logging.info(p)
          if p[c.I_MOTHER] is not None:
            logging.info('\t m {}'.format(records[p[c.I_MOTHER]]))
          if p[c.I_FATHER] is not None:
            logging.info('\t f {}'.format(records[p[c.I_FATHER]]))
          if p[c.I_CHILDREN] is not None:
            logging.info('\t c {}'.format(records[p[c.I_CHILDREN]]))
          if p[c.I_SPOUSE] is not None:
            logging.info('\t s {}'.format(records[p[c.I_SPOUSE]]))

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

  def merge_attr_sim_baseline(self, links_list, merging_t):

    # atomic_average = sim.atomic_average
    atomic_average = sim.calculate_weighted_atomic_str_sim
    processed_link_dict = defaultdict(set)

    TYPE1, TYPE2, SIM, N_ATOMIC = c.TYPE1, c.TYPE2, c.SIM, c.N_ATOMIC
    R1, R2 = c.R1, c.R2
    I_CERT_ID, I_ROLE, I_SEX = c.I_CERT_ID, c.I_ROLE, c.I_SEX

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
        p1 = self.record_dict[node[R1]]
        role1_cert_id = util.append_role(p1[I_CERT_ID], p1[I_ROLE], p1[I_SEX])

        p2 = self.record_dict[node[R2]]
        role2_cert_id = util.append_role(p2[I_CERT_ID], p2[I_ROLE], p2[I_SEX])

        sorted_key = tuple(sorted({role1_cert_id, role2_cert_id}))
        processed_link_dict[node[TYPE2]].add(sorted_key)

    t = round(time.time() - start_time, 2)
    self.calculate_persist_measures(processed_link_dict, links_list,
                                    'baseline-attr-sim', t)
