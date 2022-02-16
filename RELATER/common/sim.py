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
import math
from collections import defaultdict

import networkx as nx

from febrl import stringcmp
from common import constants as c, settings, util, hyperparams

# A similarity dictionary is kept to optimize the performance.
# This reduces the similarity calculation o
from common.enums import Links

similarity_dict = defaultdict(dict)
similarity_dict['JW'] = {}

pair_sim_dict = None


def init_pair_sim_dict():
  global pair_sim_dict
  pair_sim_dict = nx.read_gpickle(settings.atomic_node_file)


def atomic_average(atomic_sim_dict):
  if len(atomic_sim_dict) == 0:
    return 0.0
  return sum(atomic_sim_dict.values()) / len(atomic_sim_dict) * 1.0


def atomic_amb_weighted_average(node, atomic_sim_dict, p1, p2, N, attributes):
  """
  Calculates the atomic similarity incorporating the ambiguity of a set of
  attributes. The set of of attributes include first name, lastname,
  and `settings.amb_f_attribute`.
  The final similarity is a combination of string similarities and ambiguity
  as a weighted average based on `hyperparams.gamma`.

  :param node_id:
  :param link:
  :param atomic_nodes: Atomic nodes connected with the node.
  :param o1: Person 1
  :param o2: Person 2
  :return:
  """
  if len(atomic_sim_dict) == 0:
    # return 0
    raise Exception('No atomic nodes')

  ## NO Ambiguity
  #
  # return __calculate_atomic_sim_no_amb__(atomic_nodes)

  SIM_AMB = c.SIM_AMB

  if SIM_AMB in node.iterkeys():
    amb_sim = node[SIM_AMB]
  else:
    # Why we are summing up frequencies
    # If one record has a string variation,
    # taking average will not work, only summation works
    frequency = p1[c.I_FREQ] + p2[c.I_FREQ]
    amb_sim = get_ambiguity(frequency, N)
    node[SIM_AMB] = amb_sim

  attr_sim = calculate_weighted_atomic_str_sim(atomic_sim_dict, attributes)
  node[c.SIM_ATTR] = attr_sim
  sim_val = (attr_sim * hyperparams.wa + amb_sim * (1 - hyperparams.wa))
  # sim_val = attr_sim

  # Set node similarity to atomic similarity.
  node[c.SIM_ATOMIC] = sim_val
  node[c.SIM] = sim_val


def calculate_weighted_atomic_str_sim(atomic_sim_dict, attributes_meta):
  must_attributes = attributes_meta.must_attr.attributes
  core_attributes = attributes_meta.core_attr.attributes
  extra_attributes = attributes_meta.extra_attr.attributes

  must_attribute_sim_list = [sim for attribute, sim in
                             atomic_sim_dict.iteritems() if
                             attribute in must_attributes]
  core_attribute_sim_list = [sim for attribute, sim in
                             atomic_sim_dict.iteritems() if
                             attribute in core_attributes]
  extra_attribute_sim_list = [sim for attribute, sim in
                              atomic_sim_dict.iteritems() if
                              attribute in extra_attributes]

  # If must attributes are absent, must attribute weight is given to core
  # attributes
  if len(must_attributes) > 0 and len(core_attributes) > 0:
    must_avg = sum(must_attribute_sim_list) * 1.0 / len(must_attributes)
    core_avg = sum(core_attribute_sim_list) * 1.0 / len(core_attributes)
  elif len(must_attributes) == 0 and len(core_attributes) > 0:
    core_avg = sum(core_attribute_sim_list) * 1.0 / len(core_attributes)
    must_avg = core_avg
  elif len(must_attributes) > 0 and len(core_attributes) == 0:
    must_avg = sum(must_attribute_sim_list) * 1.0 / len(must_attributes)
    core_avg = must_avg
  else:
    raise Exception('Both core and must can not be zero.')

  if len(extra_attribute_sim_list) == 0:
    s = core_avg * 0.3 + must_avg * 0.7
  else:
    extra_avg = sum(extra_attribute_sim_list) * 1.0 / len(
      extra_attribute_sim_list)
    s = core_avg * 0.2 + must_avg * 0.6 + extra_avg * 0.2
  return s


def relationship_average(strong_neighbor_count,
                         strong_neighbor_merged_count,
                         weak_neighbor_count,
                         weak_neighbor_merged_count):
  val = (2 * strong_neighbor_merged_count) + (1 * weak_neighbor_merged_count)
  tot = (2 * strong_neighbor_count) + (1 * weak_neighbor_count)
  if tot == 0:
    return 0.0
  return float(val) / tot


def get_avg_sim(i_attr, list1, list2):
  sim_list = list()
  for str1 in list1:
    for str2 in list2:
      s = 0
      if (str1, str2) in pair_sim_dict[i_attr]:
        s = pair_sim_dict[i_attr][(str1, str2)][1]
      elif (str2, str1) in pair_sim_dict[i_attr]:
        s = pair_sim_dict[i_attr][(str2, str1)][1]
      if s > 0:
        sim_list.append(s)
  if len(sim_list) == 0:
    return 0
  return sum(sim_list) * 1.0 / (len(sim_list))


def get_pair_sim(pair, function):
  if pair in similarity_dict[function]:
    return similarity_dict[function][pair]

  str1 = pair[0]
  str2 = pair[1]

  if function == 'JW':
    sim_val = stringcmp.sortwinkler(str1, str2)

  elif function == 'MAD':
    str1 = float(str1)
    str2 = float(str2)
    if abs(str1 - str2) <= settings.numerical_comparison_dmax:
      sim_val = 1.0 - (
        abs(str1 - str2)) * 1.0 / settings.numerical_comparison_dmax
    else:
      sim_val = 0.0
  else:
    raise Exception('Unsupported function')

  similarity_dict[function][pair] = sim_val
  similarity_dict[function][pair[1], pair[0]] = sim_val
  return sim_val


def get_pair_sim_no_cache(pair, function):
  str1 = pair[0]
  str2 = pair[1]

  if function == 'JW':
    sim_val = stringcmp.sortwinkler(str1, str2)

  elif function == 'MAD':
    str1 = float(str1)
    str2 = float(str2)
    if abs(str1 - str2) <= settings.numerical_comparison_dmax:
      sim_val = 1.0 - (
        abs(str1 - str2)) * 1.0 / settings.numerical_comparison_dmax
    else:
      sim_val = 0.0
  else:
    raise Exception('Unsupported function')

  return sim_val


# -----------------------------------------------------------------------------
# Functions to get attribute based similarity of an atomic node
# -----------------------------------------------------------------------------
# todo:
def def1_atomic_sim_high_priority_only(atomic_sim_dict, attributes):
  if len(atomic_sim_dict) == 0:
    return 0.0
  core_attribute_sim_list = [sim for attribute, sim in
                             atomic_sim_dict.iteritems() if
                             attribute in attributes.core_attributes]
  core_attribute_sim_list.sort(reverse=True)
  core_attribute_average = sum(core_attribute_sim_list[
                               0:attributes.core_least_count]) * 1.0 / \
                           attributes.core_least_count

  extra_attribute_sim_list = [sim for attribute, sim in
                              atomic_sim_dict.iteritems() if
                              attribute in attributes.extra_attributes]
  extra_attribute_sim_list.sort(reverse=True)
  extra_attribute_average = sum(extra_attribute_sim_list[
                                0:attributes.extra_least_count]) * 1.0 / \
                            attributes.extra_least_count

  return core_attribute_average * settings.beta + \
         extra_attribute_average * (1 - settings.beta)


def get_ambiguity(f, N):
  """
  Method to get the ambiguity of two strings

  :param attribute_index: Index of the attribute
  :param str1: String 1
  :param str2: String 2
  :return: Ambiguity measure based on attribute frequency dictionary
  """
  # Method 1
  # a = 1.0 / (math.log(f + 9, 10))

  # Method 2
  # N = sum(attribute_frequency_dict[attribute_index].values())
  # if N == 0:
  #   N = 1
  # a = -1 * math.log((f * 1.0) / N, 1000)

  # Method 3
  # old_range = (attribute_frequency_dict[attribute_index].most_common(1)[0][1]) \
  #             * 1.0
  # new_range = 0.5
  # a = 1.0 - (((f - 1) * new_range) / old_range)

  # Method 4
  # old_range = (attribute_frequency_dict[attribute_index].most_common(1)[0][1]) \
  #             * 1.0
  # new_range = 1.0
  # a = 1.0 - (((f - 1) * new_range) / old_range)

  # Method 5
  # TODO: changed
  # N = len(people_dict)
  # N = 109462

  if f == 0 or N == 0:
    raise ValueError()
  not_normalized_a = -1 * math.log((f * 1.0) / N, 2)
  highest_possible_a = -1 * math.log((1.0) / N, 2)

  old_range = highest_possible_a * 1.0
  new_range = 1.0
  normalized_a = (not_normalized_a * new_range) / old_range

  return normalized_a
