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
import fcntl
import json
import logging
import math
from datetime import date

from febrl import stringcmp
from common import settings, constants as c, hyperparams
from common.enums import Links, Role, Sex
import os
import psutil

gt_link_enumerate = False


# -----------------------------------------------------------------------------
# Utility Functions for Graph Processing
# -----------------------------------------------------------------------------

def get_node_gender(gender1, gender2):
  if gender1 == '':
    return gender2
  if gender2 == '':
    return gender1
  # If genders are 'Female' and 'Male', it is invalid.
  # Return 0 for invalid
  if gender1 != gender2:
    return 0
  else:
    return gender1


def get_relationship_type_index(relationship_type):
  if relationship_type == c.MOTHER:
    return c.I_MOTHER
  if relationship_type == c.FATHER:
    return c.I_FATHER
  if relationship_type == c.SPOUSE:
    return c.I_SPOUSE
  if relationship_type == c.CHILDREN:
    return c.I_CHILDREN


def retrieve_link(role1, role2, link_list):
  link = '-'.join([role1, role2])
  if link in link_list:
    return link
  return '-'.join([role2, role1])


def retrieve_merged(str1, str2):
  merged = set()
  if str1 is not None:
    merged.add(str1)
  if str2 is not None:
    merged.add(str2)
  return merged


def retrieve_merged_lists(list1, list2):
  merged = set()
  if list1 is not None:
    for str1 in list1:
      merged.add(str1)
  if list2 is not None:
    for str2 in list2:
      merged.add(str2)
  return merged


def get_min_year(date1, date2):
  if date1 is None and date2 is None:
    return 0
  elif date1 is None:
    return date2.year
  elif date2 is None:
    return date1.year
  else:
    return min([date1.year, date2.year])


def get_max_year(date1, date2):
  if date1 is None and date2 is None:
    return 3000
  elif date1 is None:
    return date2.year
  elif date2 is None:
    return date1.year
  else:
    return max([date1.year, date2.year])


# -----------------------------------------------------------------------------
# Utils for Temporal Constraints
# -----------------------------------------------------------------------------


def is_after_onegen_dates(date1, date2):
  if date1 is None or date2 is None:
    return True
  year1 = date1.year
  year2 = date2.year
  if year1 == 0 or year2 == 0:
    return True
  elif (year2 - year1) >= 15:
    return True
  return False


def is_after_onegen_years(year1, year2):
  if year1 == 0 or year2 == 0 or year1 is None or year2 is None:
    return True
  elif (year2 - year1) >= 15:
    return True
  return False


def is_almost_same_years(year1, year2):
  if year1 == 0 or year2 == 0 or year1 is None or year2 is None:
    return True
  elif abs(year2 - year1) <= settings.year_gap:
    return True
  return False


def is_almost_same_marriage_years(year1, year2):
  if year1 == 0 or year2 == 0 or year1 is None or year2 is None:
    return True
  elif abs(year2 - year1) <= settings.marriage_year_gap:
    return True
  return False


def is_after_twogen(date1, date2):
  if date1 is None or date2 is None:
    return True
  year1 = date1.year
  year2 = date2.year
  if year1 == 0 or year2 == 0:
    return True
  elif (year2 - year1) >= 30:
    return True
  return False


def is_after(date1, date2):
  if date1 is None or date2 is None:
    return True
  # 2 days are allowed to cater for data quality issues
  if (date1 - date2).days <= 2:
    return True
  return False


def is_nine_months_apart(date1, date2):
  if date1 is None or date2 is None:
    return True
  month_diff = date2.month - date1.month + (date2.year - date1.year) * 12

  if abs(month_diff) >= 9:
    return True
  return False


def is_atmost_nine_months_before(date1, date2):
  if date1 is None or date2 is None:
    return True
  month_diff = date2.month - date1.month + (date2.year - date1.year) * 12

  if month_diff >= -9:
    return True
  return False


# -----------------------------------------------------------------------------
# Utils for Ground Truth Analysis
# -----------------------------------------------------------------------------


def enumerate_links(person_role_list, entity_dict=None):
  links_dict = {}

  for link in Links.list():
    links_dict[link] = set()

  role_list = Role.list()
  for p in person_role_list:
    person = {}
    for role in role_list:
      # Role based appending is not needed for pids in 'BHIC'
      if c.data_set != 'bhic':
        person[role] = [append_role(cert_id, role, p['S'])
                        for cert_id in p[role]]
      else:
        person[role] = p[role]

    # Generate Bb-X links
    #
    if len(person['Bb']) > 0:
      Bb_X_roles = ['Bp', 'Mm', 'Mbp', 'Mgp', 'Dd', 'Ds', 'Dp']
      add_role_specific_links(person, 'Bb', Bb_X_roles, links_dict, 'Bb-{}')

    # Generate Bp-X roles
    #
    if len(person['Bp']) > 0:
      Bp_X_roles = ['Bp', 'Mm', 'Mbp', 'Mgp', 'Dd', 'Ds', 'Dp']
      add_role_specific_links(person, 'Bp', Bp_X_roles, links_dict, 'Bp-{}')

    # Generate Mm-X roles
    #
    if len(person['Mm']) > 0:
      Mm_X_roles = ['Mm', 'Mbp', 'Mgp', 'Dd', 'Ds', 'Dp']
      add_role_specific_links(person, 'Mm', Mm_X_roles, links_dict, 'Mm-{}')

    # Generate Mbp-X roles
    #
    if len(person['Mbp']) > 0:
      Mbp_X_roles = ['Mbp', 'Mgp', 'Dd', 'Ds', 'Dp']
      add_role_specific_links(person, 'Mbp', Mbp_X_roles, links_dict, 'Mbp-{}')

    # Generate Mgp-X roles
    #
    if len(person['Mgp']) > 0:
      Mgp_X_roles = ['Mgp', 'Dd', 'Ds', 'Dp']
      add_role_specific_links(person, 'Mgp', Mgp_X_roles, links_dict, 'Mgp-{}')

    # Generate Dd-X roles
    #
    if len(person['Dd']) > 0:
      Dd_X_roles = ['Ds', 'Dp']
      add_role_specific_links(person, 'Dd', Dd_X_roles, links_dict, 'Dd-{}')

    # Generate Ds-X roles
    #
    if len(person['Ds']) > 0:
      Ds_X_roles = ['Ds', 'Dp']
      add_role_specific_links(person, 'Ds', Ds_X_roles, links_dict, 'Ds-{}')

    # Generate Dp-X roles
    #
    if len(person['Dp']) > 0:
      Dp_X_roles = ['Dp']
      add_role_specific_links(person, 'Dp', Dp_X_roles, links_dict, 'Dp-{}')

    if len(person['Bb']) + len(person['Bp']) + len(person['Mm']) + len(
        person['Mbp']) + len(person['Mgp']) + len(person['Dd']) + \
        len(person['Ds']) + len(person['Dp']) >= 100:
      logging.info('------ large entity')
      logging.info(entity_dict[p['E_ID']])

  return links_dict


def add_role_specific_links(person, role1, comparing_role_list,
                            links_dict, link_string):
  """
  Function to add role specific links.
  For example, if the role is Bb, it is compared with all roles in the
  comparing_role_list. For each role combination the certificate IDs of of
  both roles are compared and added to the links dictionary.
  :param person: Person
  :param role1: Specific role considered
  :param comparing_role_list: List of roles for which the the specific role
  is compared to.\
  :param links_dict: Dictionary holding the links
  """

  # Iterate through all comparing roles
  for role2 in comparing_role_list:
    # Iterate through all certificate IDs specific to role2
    for role2_cert_id in person[role2]:
      # Iterate through all certificate IDs specific to role1
      for role1_cert_id in person[role1]:
        # If comparing a person to the same role in same certificate
        if role1_cert_id == role2_cert_id:
          continue
        link = link_string.format(role2)
        # links_dict[link].add((role1_cert_id, role2_cert_id))
        sorted_key = tuple(sorted({role1_cert_id, role2_cert_id}))
        links_dict[link].add(sorted_key)


def append_role(cert_id, role, sex):
  # If we change here, we need to change in data_loader.py line 580+
  if role == 'Bb':
    return cert_id + '_bb'
  elif role == 'Bp':
    if sex == 'F':
      return cert_id + '_bm'
    elif sex == 'M':
      return cert_id + '_bf'
    else:
      raise Exception('Not possible')
  elif role == 'Dp':
    if sex == 'F':
      return cert_id + '_dm'
    elif sex == 'M':
      return cert_id + '_df'
    else:
      raise Exception('Not possible')
  elif role == 'Mbp':
    if sex == 'F':
      return cert_id + '_mbm'
    elif sex == 'M':
      return cert_id + '_mbf'
    else:
      raise Exception('Not possible')
  elif role == 'Mgp':
    if sex == 'F':
      return cert_id + '_mgm'
    elif sex == 'M':
      return cert_id + '_mgf'
    else:
      raise Exception('Not possible')
  # elif role in ['Bp', 'Dp', 'Mbp', 'Mgp']:
  #   if sex == 'F':
  #     return cert_id + '_m'
  #   elif sex == 'M':
  #     return cert_id + '_f'
  #   else:
  #     raise Exception('Not possible')
  elif role == 'Mm':
    if sex == 'F':
      return cert_id + '_b'
    elif sex == 'M':
      return cert_id + '_g'
    else:
      raise Exception('Not possible')
  elif role == 'Dd':
    return cert_id + '_dd'
  elif role == 'Ds':
    return cert_id + '_ds'


def retrieve_child_parent_role_set(sib_life_seg):
  """
  Find the gender to decide if the person is a bride or groom to obtain roles set.
  Roles set contains a set of tuples. Each tuple contains the main
  person considered in a certificate and the parent considered in
  each certificate.
  :param sib_life_seg:
  :return:
  """
  if sib_life_seg['S'] in ['F', 'M']:
    m_parent_role = 'Mbp' if sib_life_seg['S'] == 'F' else 'Mgp'
    sib_roles_set = {('Bb', 'Bp'), ('Dd', 'Dp'), ('Mm', m_parent_role)}
  else:
    sib_roles_set = {('Bb', 'Bp'), ('Dd', 'Dp')}
  return sib_roles_set


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def persist_graph_gen_times(rule, a_time, r_time):
  filename = '{}graph-gen-times.csv'.format(settings.output_home_directory)
  file_exists = os.path.isfile(filename)

  with open(filename, 'a') as results_file:
    heading_list = ['atomic_t', 'rule', 'a_time', 'r_time']
    writer = csv.DictWriter(results_file, fieldnames=heading_list)
    if not file_exists:
      writer.writeheader()

    fcntl.flock(results_file, fcntl.LOCK_EX)
    result_dict = {
      'atomic_t': hyperparams.atomic_t,
      'rule': rule,
      'a_time': round(a_time, 4),
      'r_time': round(r_time, 4)
    }
    writer.writerow(result_dict)
    fcntl.flock(results_file, fcntl.LOCK_UN)


def get_logfilename(basename):
  return '{}-{}-{}-{}-{}.log'.format(basename, hyperparams.data_set,
                                  hyperparams.atomic_t,
                                  hyperparams.merge_t, settings.today_str)


def print_memory_usage(func):
  process = psutil.Process(os.getpid())
  logging.info('{} : {}'.format(func, process.memory_info().rss / (
      1024.0 * 1024.0 * 1024.0)))


def print_atomic_nodes_combined(atomic_nodes):
  atomic_nodes_type_node_dict = {}
  for node in atomic_nodes:
    atomic_nodes_type_node_dict[node[c.TYPE2]] = node

  for i in range(6, 17):

    if i == 12:
      continue

    attr_name = get_attribute_name(i)
    if i in atomic_nodes_type_node_dict.iterkeys():
      node = atomic_nodes_type_node_dict[i]
      sim_val = node[c.SIM]

      attr_val = '{}-{}'.format(node[c.R1], node[c.R2])

      attr_name_padd_str = ' ' * abs(8 - len(attr_name))
      attr_val1_padd_str = ' ' * abs(25 - len(str(attr_val)))

      line_str = '   %s%s | %s%s | %.2f' % (attr_name_padd_str,
                                            attr_name,
                                            attr_val1_padd_str,
                                            attr_val, sim_val)
      print(line_str)


def print_atomic_nodes(atomic_nodes, p1, p2):
  atomic_nodes_type_node_dict = {}
  for node in atomic_nodes:
    atomic_nodes_type_node_dict[node[c.TYPE2]] = node

  for i in range(6, 17):

    if i == 12:
      continue

    attr_name = get_attribute_name(i)
    if i in atomic_nodes_type_node_dict.iterkeys():
      node = atomic_nodes_type_node_dict[i]
      sim_val = node[c.SIM]
    else:
      function = 'MAD' if i in [13, 14] else 'JW'
      if p1[i] is None or p2[i] is None:
        sim_val = 0.0
      else:

        if function == 'JW':
          sim_val = stringcmp.sortwinkler(p1[i], p2[i])

        else:
          if abs(p1[i] - p2[i]) <= settings.numerical_comparison_dmax:
            sim_val = 1.0 - (
              abs(p1[i] - p2[i])) * 1.0 / settings.numerical_comparison_dmax
          else:
            sim_val = 0.0

    str1 = '' if p1[i] is None else p1[i]
    str2 = '' if p2[i] is None else p2[i]

    attr_name_padd_str = ' ' * abs(8 - len(attr_name))
    attr_val1_padd_str = ' ' * abs(15 - len(str(str1)))
    attr_val2_padd_str = ' ' * abs(15 - len(str(str2)))

    line_str = '   %s%s | %s%s | %s%s | %.2f' % (attr_name_padd_str,
                                                 attr_name,
                                                 attr_val1_padd_str,
                                                 str1,
                                                 attr_val2_padd_str,
                                                 str2, sim_val)
    print(line_str)


def get_attribute_name(attr_index):
  if attr_index == 6:
    return 'fname'
  if attr_index == 7:
    return 'sname'
  if attr_index == 8:
    return 'addr1'
  if attr_index == 9:
    return 'parish'
  if attr_index == 10:
    return 'addr2'
  if attr_index == 11:
    return 'occp'
  if attr_index == 13:
    return 'birth_y'
  if attr_index == 14:
    return 'mrg_y'
  if attr_index == 15:
    return 'mrg_plc1'
  if attr_index == 16:
    return 'mrg_plc2'


def get_people_dict(people_file):
  # Deserializing people
  people_dict = {}
  with open(people_file) as f:
    people_list = json.load(f)
  for p in people_list:
    people_dict[p[c.I_ID]] = p

  if c.data_set != 'IPUMS':
    I_TIMESTAMP = c.I_TIMESTAMP
    for person in people_dict.itervalues():
      event_date = date.fromtimestamp(person[I_TIMESTAMP])
      person.append(event_date)

  logging.info('People count {}'.format(len(people_dict)))

  return people_dict

def geocode_distance(origin, destination):
  lat1, lon1 = origin
  lat2, lon2 = destination
  radius = 6371.0  # km

  dlat = math.radians(lat2 - lat1)
  dlon = math.radians(lon2 - lon1)
  a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
      * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
  d = radius * c

  return d

class SetEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, set):
      return list(obj)
    return json.JSONEncoder.default(self, obj)
