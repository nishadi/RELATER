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
import multiprocessing as mp
from collections import defaultdict, Counter

from febrl import encode
from common import constants as c, settings, sim

CLUSTER_FILE = '{}amb/frequency-clusters-90-{}-{}.gpickle'


def __get__clusters__(attr_comb_set):
  get_pair_sim = sim.get_pair_sim_no_cache
  cluster_dict = defaultdict(set)
  attr_comb_list = list(attr_comb_set)
  for i, val1 in enumerate(attr_comb_list):
    for j, val2 in enumerate(attr_comb_list):
      if i == j:
        continue

      combined_str1 = '-'.join(val1)
      combined_str2 = '-'.join(val2)

      is_candidate_str = True
      for k in xrange(len(val1)):
        if get_pair_sim((val1[k], val2[k]), 'JW') < 0.9:
          is_candidate_str = False
          break
      if is_candidate_str:
        cluster_dict[combined_str1].add(combined_str2)

  return cluster_dict


def cluster_attribute_combinations_wblocking(record_list, role_type,
                                             attribute_index_list,
                                             attribute_name):
  """
  Clusters attribute combinations in `record_list`.
  Attribute combination includes attributes defined
  by the `attribute_index_list`.
  Writes the clusters to a gpickle file

  :param record_list: List of records
  :param role_type: Role type
  :param attribute_index_list: List of attribute indexes
  :param attribute_name: Attribute combination name
  """

  amb_dir = '{}amb/'.format(settings.output_home_directory)
  if not os.path.isdir(amb_dir):
    os.makedirs(amb_dir)

  # Aggregating a list of attribute combinations
  attr_comb_blocks = defaultdict(set)
  i_attr_blocking = attribute_index_list[0]
  for p in record_list:
    if p[c.I_ROLE] == role_type:
      blocking_key = encode.dmetaphone(p[i_attr_blocking])
      attribute_list = [str(p[i_attribute]) for i_attribute in
                        attribute_index_list]

      attr_comb_blocks[blocking_key].add(__get_concat_tuple__(attribute_list))

  for blocking_key, blocked_list in attr_comb_blocks.iteritems():
    logging.info(
      'Blocking key %s with %s values' % (blocking_key, len(blocked_list)))

  # Clustering the attribute combinations parallel
  pool = mp.Pool(settings.parallel_process_count)
  result_objects = [pool.apply_async(__get__clusters__, args=(attr_comb_set,))
                    for attr_comb_set in attr_comb_blocks.itervalues()]
  pool.close()
  pool.join()

  attr_comb_dict = dict()
  for result in result_objects:
    attr_comb_dict.update(result.get())

  output_file = CLUSTER_FILE.format(
    settings.output_home_directory, role_type, attribute_name)
  pickle.dump(attr_comb_dict, open(output_file, 'wb'))

  logging.info('Clustering completed successfully')


def cluster_attribute_combinations(record_list, role_type, attribute_index_list,
                                   attribute_name):
  """
  Clusters attribute combinations in `record_list`.
  Attribute combination includes attributes defined
  by the `attribute_index_list`.
  Writes the clusters to a gpickle file

  :param record_list: List of records
  :param role_type: Role type
  :param attribute_index_list: List of attribute indexes
  :param attribute_name: Attribute combination name
  """

  output_file = CLUSTER_FILE.format(
    settings.output_home_directory, role_type, attribute_name)
  if os.path.isfile(output_file):
    return

  amb_dir = '{}amb/'.format(settings.output_home_directory)
  if not os.path.isdir(amb_dir):
    os.makedirs(amb_dir)

  # Aggregating a list of attribute combinations
  attr_comb_set = set()
  for p in record_list:
    if p[c.I_ROLE] == role_type:

      attribute_list = list()
      for i_attribute in attribute_index_list:
        attribute_list.append(str(p[i_attribute]))

      attr_comb_set.add(__get_concat_tuple__(attribute_list))
  attr_comb_list = list(attr_comb_set)

  # Clustering the attribute combinations
  attr_comb_dict = __get__clusters__(attr_comb_list)

  pickle.dump(attr_comb_dict, open(output_file, 'wb'))

  logging.info('Clustering completed successfully')


def append_clustered_attr_f(record_list, role_type, attribute_index_list,
                            attribute_name):
  """
  Combined Attribute Frequency with attribute combination clustering

  Each person records needs to have the frequency of a set of
  pre-defined attributes for calculating the ambiguity. This method appends
  the frequencies to each person record.

  To get the frequencies of attribute combinations, we consider the summation
  of the frequencies of the set of attribute combination clusters. This
  accounts for the spelling variations as well.

  In order to calculate the frequencies, we have used only the attribute
  values of only the role provided by `role_type`. This is because we need a
  unique set of entities to calculate the frequencies.

  :param record_list: List of records
  :param role_type: Role type
  :param attribute_index_list: List of attribute indexes
  :param attribute_name: Attribute combination name
  :return: Record list with appended frequencies
  """

  frequency_counter = Counter()
  for p in record_list:
    if p[c.I_ROLE] == role_type:

      attribute_list = list()
      for i_attribute in attribute_index_list:
        attribute_list.append(str(p[i_attribute]))

      # Aggregating combined frequencies
      frequency_counter.update([__get_concat_str__(attribute_list)])

  filename = CLUSTER_FILE.format(settings.output_home_directory, role_type,
                                 attribute_name)
  attr_cluster = pickle.load(open(filename, 'rb'))

  # Appending frequencies
  I_FREQ = c.I_FREQ
  for p in record_list:

    attribute_list = list()
    for i_attribute in attribute_index_list:
      attribute_list.append(str(p[i_attribute]))
    attr_str = __get_concat_str__(attribute_list)

    # Getting frequency of this particular string
    aggregated_frequency = frequency_counter.get(
      attr_str) if attr_str in frequency_counter else 1

    # Getting frequency of string variations
    if attr_str in attr_cluster:
      for variation in attr_cluster.get(attr_str):
        if variation in frequency_counter.iterkeys():
          aggregated_frequency += frequency_counter.get(variation)

    assert aggregated_frequency > 0
    p[I_FREQ] = aggregated_frequency

  return record_list


# -----------------------------------------------------------------------------
# Utility Functions for Ambiguity
# -----------------------------------------------------------------------------
def __get_concat_str__(str_tuple):
  str_list = list()
  for str in str_tuple:
    if str is None:
      str_list.append('')
    else:
      str_list.append(str)
  return '-'.join(str_list)


def __get_concat_tuple__(str_tuple):
  str_list = list()
  for str in str_tuple:
    if str is None:
      str_list.append('')
    else:
      str_list.append(str)
  return tuple(str_list)
