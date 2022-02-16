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

import json
import pickle
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

from data import data_loader
from common import settings, constants as c, util, hyperparams
from common.enums import Links, State, Role
import csv
import logging
import os.path
import fcntl

start_time = 0.0

atomic_enriched = 0
relationship_enriched = 0

removed_edge_count = 0
split_entity_count_b = 0
split_entity_count_d = 0

activated_neighbors_count = 0
merged_singleton_count = 0
added_neighbors_count = 0

merged_entity_count = 0
valid_merged_entity_count = 0
added_node_count = 0
added_group_count = 0
added_group_size = list()
added_node_id_set = list()
added_merged_node_id_count = 0

neighbour_enriched_node_set_count = 0
neighbour_added_node_set_count = 0

dong_total = 0
dong_atomic = 0
dong_relational = 0

a_time = 0
r_time = 0


def count_people_types(females, males):
  logging.info('Stats on females : ')
  logging.info('-- Birth baby girls : {}'.format(len(females['Bb'])))
  logging.info('-- Birth baby mothers : {}'.format(len(females['Bp'])))
  logging.info('-- Marriage brides : {}'.format(len(females['Mm'])))
  logging.info('-- Marriage bride mothers : {}'.format(len(females['Mbp'])))
  logging.info('-- Marriage groom mothers : {}'.format(len(females['Mgp'])))
  logging.info('-- Death females : {}'.format(len(females['Dd'])))
  logging.info('-- Death person\'s wives : {}'.format(len(females['Ds'])))
  logging.info('-- Death person\'s mothers : {}'.format(len(females['Dp'])))

  logging.info('Stats on males : ')
  logging.info('-- Birth baby boys : {}'.format(len(males['Bb'])))
  logging.info('-- Birth baby fathers : {}'.format(len(males['Bp'])))
  logging.info('-- Marriage grooms : {}'.format(len(males['Mm'])))
  logging.info('-- Marriage bride fathers : {}'.format(len(males['Mbp'])))
  logging.info('-- Marriage groom fathers : {}'.format(len(males['Mgp'])))
  logging.info('-- Death males : {}'.format(len(males['Dd'])))
  logging.info('-- Death person\'s husbands : {}'.format(len(males['Ds'])))
  logging.info('-- Death person\'s fathers : {}'.format(len(males['Dp'])))


def print_node_stats(stats_link_dict):
  for key in sorted(stats_link_dict.keys()):
    logging.info('---- {} : {}'.format(key, stats_link_dict[key]))


def persist_entities(entity_dict, graph_def, graph_scenario):
  filename = '{}entities-{}-{}.json'.format(settings.results_dir, graph_def,
                                            graph_scenario)
  with open(filename, 'w') as outfile2:
    outfile2.write(json.dumps(entity_dict, cls=util.SetEncoder))

  entity_list = entity_dict.values()
  role_count = defaultdict(list)
  person_records_per_entity = list()
  for entity in entity_list:
    person_count = 0
    for role in Role.list():
      if len(entity[c.ROLES][role]) > 0:
        person_count += len(entity[c.ROLES][role])
        role_count[role].append(len(entity[c.ROLES][role]))
    person_records_per_entity.append(person_count)

  assert len(entity_list) == len(person_records_per_entity)
  avg_entity_person_records = sum(person_records_per_entity) * 1.0 / len(
    person_records_per_entity)
  logging.info('Number of entities {}'.format(len(entity_list)))
  logging.info('Entity wise max-{} min-{} avg-{}'.format(max(
    person_records_per_entity), min(person_records_per_entity),
    avg_entity_person_records))
  for role in Role.list():
    avg = sum(role_count[role]) * 1.0 / len(role_count[role])
    logging.info('{}: max-{} min-{} avg-{}'
                 .format(role, max(role_count[role]),
                         min(role_count[role]), avg))


def get_link_indi_links(link):
  if link in ['Mm-Dd', 'Mm-Ds', 'Mbp-Dp', 'Mgp-Dp']:
    return 4
  if link in ['Bb-Mm', 'Bp-Mbp', 'Bp-Mgp', 'Bb-Dd', 'Bp-Dp',
              'Mm-Mm', 'Mbp-Mbp', 'Mgp-Mgp']:
    return 3
  if link in ['Bp-Bp', 'Bp-Mm',
              'Bp-Dd', 'Bp-Ds', 'Mm-Mbp', 'Mm-Mgp', 'Mbp-Dp',
              'Mgp-Dp', 'Mm-Ds', 'Mm-Dd', 'Mm-Dp', 'Mbp-Dd',
              'Mbp-Ds', 'Mgp-Dd', 'Mgp-Ds', 'Dd-Ds', 'Dd-Dp',
              'Ds-Dp', 'Dp-Dp']:
    return 2
  return 1


def persist_gt_analysis(gt_links_dict, processed_links_dict,
                        stats_dict, scenario, links_list, t):
  filename = settings.result_file
  file_exists = os.path.isfile(filename)

  with open(filename, 'a') as results_file:
    heading_list = ['link', 'individual_link', 'merge_t', 'atomic_t', 'def',
                    'scenario', 'GT', 'P', 'TP', 'FP', 'FN', 'Pre', 'Re',
                    'Fstar', 'time']
    writer = csv.DictWriter(results_file, fieldnames=heading_list)
    if not file_exists:
      writer.writeheader()

    fcntl.flock(results_file, fcntl.LOCK_EX)
    for link in links_list:

      if len(gt_links_dict[link]) < 100:
        continue

      tp = stats_dict['tp'][link]
      fp = stats_dict['fp'][link]
      fn = stats_dict['fn'][link]
      f_star = round(tp * 100.0 / (tp + fp + fn), 2)
      result_dict = {
        'link': link,
        'individual_link': get_link_indi_links(link),
        'def': 'def',
        'scenario': scenario,
        'merge_t': hyperparams.merge_t,
        'atomic_t': hyperparams.atomic_t,
        'GT': len(gt_links_dict[link]),
        'P': len(processed_links_dict[link]),
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Pre': stats_dict['pre'][link],
        'Re': stats_dict['re'][link],
        'Fstar': f_star,
        'time' : t
      }
      writer.writerow(result_dict)
    fcntl.flock(results_file, fcntl.LOCK_UN)


node_state_count = {}
node_state_count[State.ACTIVE] = 0
node_state_count[State.INACTIVE] = 0
node_state_count[State.MERGED] = 0
node_state_count[State.NONMERGE] = 0


def update_node_state_count(state):
  node_state_count[state] += 1


def print_node_state_count_stats(log_str):
  global node_state_count
  logging.info(log_str)
  logging.info(
    '-- Active nodes {}'.format(node_state_count[State.ACTIVE]))
  logging.info(
    '-- Merged nodes {}'.format(node_state_count[State.MERGED]))
  logging.info(
    '-- In Active nodes {}'.format(node_state_count[State.INACTIVE]))
  logging.info(
    '-- Non-Merge nodes {}'.format(node_state_count[State.NONMERGE]))

  node_state_count = {}
  node_state_count[State.ACTIVE] = 0
  node_state_count[State.INACTIVE] = 0
  node_state_count[State.MERGED] = 0
  node_state_count[State.NONMERGE] = 0


### STATS on the data sets

def data_set_charateristics():
  # IPUMS
  # c.data_set = 'ipums'
  # settings.__init_settings__(0.9, 1.0, 'F')
  #
  # people_list = pickle.load(open(settings.people_file, 'rb'))
  # p_1870_list = filter(lambda p:p[c.HH_I_YEAR] == 1870, people_list)
  # p_1880_list = filter(lambda p:p[c.HH_I_YEAR] == 1880, people_list)
  #
  # print '1870 : len {}'.format(len(p_1870_list))
  # print '1880 : len {}'.format(len(p_1880_list))
  #
  # for role in ['F', 'M', 'C']:
  #   print '1870 {} : {}'.format(role, len(filter(lambda p:p[c.I_ROLE] ==
  #                                                         role, p_1870_list)))
  #   print '1880 {} : {}'.format(role, len(filter(lambda p: p[c.I_ROLE] ==
  #                                                          role, p_1880_list)))

  # IOS
  c.data_set = 'kil'
  settings.__init_settings__(0.9, 0.8)
  people_list = pickle.load(open(settings.people_file, 'rb'))

  cert_role_dict = {'B': ['Bb', 'Bp'], 'M': ['Mm', 'Mbp', 'Mgp'],
                    'D': ['Dd', 'Dp', 'Ds']}
  for ds, role_list in cert_role_dict.iteritems():
    sum = 0
    for role in role_list:
      l = len(filter(lambda p: p[c.I_ROLE] == role, people_list))
      print role, l
      sum += l
    print ds, sum


def plot_bdm_annual_trend():
  year_list = list()
  b_list = list()
  d_list = list()
  m_list = list()

  settings.birth_file = settings.birth_file.format(1760)
  settings.marriage_file = settings.marriage_file.format(1760)
  settings.death_file = settings.death_file.format(1760)

  people_dict = data_loader.enumerate_records()

  # year_list = [1,2,3]
  # b_list = [10, 11, 12]
  # d_list = [5,6,7]
  # m_list = [10, 9, 11]

  for year in range(1759, 1970, 1):
    year_list.append(year)

    # Births
    b_count = len(filter(lambda p: p[c.I_ROLE] == 'Bb' and
                                   p[c.I_EDATE].year == year,
                         people_dict.itervalues()))
    b_list.append(b_count)

    # Marriages
    m_count = len(filter(lambda p: p[c.I_ROLE] == 'Mm' and
                                   p[c.I_EDATE].year == year and
                                   p[c.I_SEX] == 'F',
                         people_dict.itervalues()))
    m_list.append(m_count)

    # Deaths
    d_count = len(filter(lambda p: p[c.I_ROLE] == 'Dd' and
                                   p[c.I_EDATE].year == year,
                         people_dict.itervalues()))
    d_list.append(d_count)

    print year, b_count, m_count, d_count

  plt.plot(year_list, b_list, color='green', label='Birth')
  plt.plot(year_list, m_list, color='blue', label='Marriage')
  plt.plot(year_list, d_list, color='red', label='Death')
  plt.xlabel('Year')
  plt.ylabel('BDM count per year')
  plt.legend()
  # plt.show()
  plt.savefig('bhic-BDM-Trend.png')


def raw_data_set_stats_dutch_dataset(people_list):
  if not os.path.isdir(settings.stats_dir):
    os.makedirs(settings.stats_dir)

  columns = ['pid', 'entity_id', 'freq', 'sex', 'role', 'timestamp', 'edate',
             'cert_id', 'fname', 'sname', 'addr1', 'parish', 'occ', 'addr2',
             'b_year', 'm_year', 'm_place1', 'm_place2', 'm_status', 'mother',
             'father', 'children', 'spouse']
  people_df = pd.DataFrame(people_list, columns=columns)

  # Stats of numerical fields
  info_file = os.path.join(settings.stats_dir, 'numerical_fields.csv')
  people_df.describe().to_csv(info_file)

  for role in Role.list():
    for attr in ['m_year', 'b_year']:
      role_df = people_df[people_df['role'] == role]
      role_file = os.path.join(settings.stats_dir, 'hist_%s_%s.png' % (role,
                                                                       attr))
      role_df[attr].hist(bins=50)
      plt.savefig(role_file)
      plt.clf()

  # Combined attributes
  # people_df['fname_sname'] = list(zip(people_df.fname, people_df.sname))
  # people_df['fname_sname_a1'] = list(
  #   zip(people_df.fname_sname, people_df.addr1))
  # people_df['fname_sname_p'] = list(
  #   zip(people_df.fname_sname, people_df.parish))
  #
  # columns.append('fname_sname')
  # columns.append('fname_sname_a1')
  # columns.append('fname_sname_p')

  # Stats of categorical fields
  for field in ['fname', 'sname', 'm_place1']:
    # Total count
    file_name = os.path.join(settings.stats_dir, 'total-count-%s.csv' % field)
    people_df[field].value_counts().to_csv(file_name)

    # Unique count
    file_name = os.path.join(settings.stats_dir, 'unique-count-%s.csv' % field)
    people_df[field].unique().to_csv(file_name)


def raw_data_set_stats(people_dict):
  if not os.path.isdir(settings.stats_dir):
    os.makedirs(settings.stats_dir)

  columns = ['pid', 'sex', 'role', 'timestamp', 'edate', 'cert_id', 'fname',
             'sname', 'addr1', 'parish', 'addr2', 'occ', 'marital_status',
             'b_year', 'm_year', 'm_place1', 'm_place2', 'mother', 'father',
             'children', 'spouse', 'entity_id']
  # 'children', 'spouse', 'entity_id', 'date']
  # 'children', 'spouse', 'entity_id', 'addr1_f', 'parish_f', 'date']
  people_df = pd.DataFrame(people_dict.itervalues(), columns=columns)

  # Stats of numerical fields
  info_file = os.path.join(settings.stats_dir, 'numerical_fields.csv')
  people_df.describe().to_csv(info_file)

  for role in Role.list():
    for attr in ['m_year', 'b_year']:
      role_df = people_df[people_df['role'] == role]
      role_file = os.path.join(settings.stats_dir, 'hist_%s_%s.png' % (role,
                                                                       attr))
      role_df[attr].hist(bins=50)
      plt.savefig(role_file)
      plt.clf()

  # Combined attributes
  people_df['fname_sname'] = list(zip(people_df.fname, people_df.sname))
  people_df['fname_sname_a1'] = list(
    zip(people_df.fname_sname, people_df.addr1))
  people_df['fname_sname_p'] = list(
    zip(people_df.fname_sname, people_df.parish))

  columns.append('fname_sname')
  columns.append('fname_sname_a1')
  columns.append('fname_sname_p')

  # Stats of categorical fields
  for field in columns:
    file_name = os.path.join(settings.stats_dir, '%s.csv' % field)
    people_df[field].value_counts().to_csv(file_name)


def plot_gt_time_difference():
  gt_links_dict = pickle.load(open(settings.gt_file, 'rb'))
  people_list = pickle.load(open(settings.people_file, 'rb'))

  cert_id_role_person_id_dict = dict()
  for person in people_list:
    cert_id_role = util.append_role(person[c.I_CERT_ID], person[c.I_ROLE],
                                    person[c.I_SEX])
    cert_id_role_person_id_dict[cert_id_role] = person

  for link, gt_pair_list in gt_links_dict.iteritems():

    if len(gt_pair_list) == 0:
      continue

    year_diff_counter = defaultdict(int)
    for cert_id1, cert_id2 in gt_pair_list:
      p1 = cert_id_role_person_id_dict[cert_id1]
      p2 = cert_id_role_person_id_dict[cert_id2]

      year_diff = abs(p1[c.I_EDATE].year - p2[c.I_EDATE].year)
      year_diff_counter[year_diff] += 1

    # Plotting the graph
    x_values = list()
    y_values = list()
    for i in range(min(year_diff_counter.keys()),
                   max(year_diff_counter.keys())):
      x_values.append(i)
      y_values.append(year_diff_counter[i])

    plt.plot(x_values, y_values, color='red', label=link)
    plt.xlabel('Year Difference')
    plt.ylabel('Frequency')
    plt.legend()
    # plt.show()
    plt.savefig('{}-year-diff.png'.format(link))
    plt.clf()


if __name__ == '__main__':
  # c.__init_constants__('bhic')
  # settings.__init_settings__(1.0, 1.0, in_role_type='Bb')
  #
  # # people_dict = common.get_people_dict(settings.people_file)
  # # raw_data_set_stats(people_dict)
  #
  # people_list = pickle.load(open(settings.people_file, 'r'))
  # # people_list = list()
  # plot_bdm_annual_trend(people_list)

  # c.__init_constants__('ios')
  # settings.__init_settings__(1.0, 1.0)
  # plot_gt_time_difference()
  #
  # plot_bdm_annual_trend()

  data_set_charateristics()

  print 'Successfully completed!'
