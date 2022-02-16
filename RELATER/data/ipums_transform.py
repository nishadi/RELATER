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
from collections import defaultdict

import pandas as pd

from common import settings, constants as c
from common.enums import Sex, HH_Role


def __retrieve_index__(filename):
  index_value_dict = dict()
  with open(filename, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
      index_value_dict[row[0]] = row[1].strip()
  return index_value_dict


def __get__year__(year_str):
  if year_str == '':
    return None
  return int(float(year_str))


def __get_byear__(age, census_year):
  if age != '':
    return int(float(census_year) - float(age))
  return None


def __get_gender(sex_id):
  if sex_id == '1.0':
    return Sex.M
  elif sex_id == '2.0':
    return Sex.F
  else:
    raise Exception('Incorrect sex')


def __get_role__(relate, sex_id):
  if relate == '3.0':
    return HH_Role.C
  if relate in ['1.0', '2.0']:
    if sex_id == '1.0':
      return HH_Role.F
    if sex_id == '2.0':
      return HH_Role.M
  raise Exception('Incorrect role')


def __get_bpl__(bpl_dict, index):
  index = index.split('.')[0]
  if index == '999':
    return None
  return bpl_dict[index]


def __get_state__(state_dict, index):
  index = index.split('.')[0]
  if index == '99':
    return None
  return state_dict[index]


def __get_city__(city_dict, index):
  index = index.split('.')[0]
  if index == '0':
    return None
  return city_dict[index]


def __get_occ__(occ_dict, index):
  index = index.split('.')[0]
  if index in ['995', '997', '999']:
    return None
  return occ_dict[index]


def generate_census_csv(race_dict, bpl_dict, occ_dict, city_dict, state_dict):
  with open(settings.census_file, 'w') as csfile:
    writer = csv.DictWriter(csfile, [c.HH_ID, c.HH_YEAR, c.HH_SERIAL,
                                     c.HH_PER_NUM, c.HH_RACE, c.HH_BPL,
                                     c.HH_FNAME, c.HH_SNAME, c.HH_BYEAR,
                                     c.HH_STATE, c.HH_CITY, c.HH_OCC,
                                     c.HH_GENDER, c.HH_ROLE])
    writer.writeheader()
    linked_file = csv.DictReader((open(settings.couples_file, 'r')))

    pid = 0
    person_dict = dict()
    gt_dict = defaultdict(set)

    relationship_dict = {
      '1.0' : 'Head/Householder',
      '2.0' : 'Spouse',
      '3.0' : 'Child',
      '4.0' : 'Child-in-law',
      '5.0' : 'Parent',
      '6.0' : 'Parent-in-Law',
      '7.0' : 'Sibling',
      '8.0' : 'Sibling-in-Law',
      '9.0' : 'Grandchild',
      '10.0' : 'Other relatives',
      '11.0' : 'Partner, friend, visitor',
      '12.0' : 'Other non-relatives',
      '13.0' : 'Institutional inmates'
    }

    for row in linked_file:

      if 'relate_1' not in row.iterkeys():
        continue

      key = row['relate_1'] + '-' + row['relate_2']
      if key not in ['1.0-1.0', '2.0-2.0', '3.0-3.0']:
        continue

      p1_id, p2_id, p1_role, p2_role = None, None, None, None

      # Writing first census year related record
      if row['namelast_1'] == '' and row['namefrst_1'] == '':
        pass
      elif row['relate_1'] in ['1.0', '2.0', '3.0']:
        p_str = '%s-%s-%s' % (row['year_1'], row['serial_1'], row['pernum_1'])
        if p_str in person_dict.iterkeys():
          p1_dict = person_dict[p_str]
          assert p1_dict[c.HH_FNAME] == row['namefrst_1']
          assert p1_dict[c.HH_SNAME] == row['namelast_1']
          p1_id, p1_role = p1_dict[c.HH_ID], p1_dict[c.HH_ROLE]
        else:
          pid += 1
          p1_id = pid
          p1_role = __get_role__(row['relate_1'], row['sex_1'])
          p1_dict = {
            c.HH_ID: p1_id,
            c.HH_GENDER: __get_gender(row['sex_1']),
            c.HH_ROLE: p1_role,
            c.HH_YEAR: __get__year__(row['year_1']),
            c.HH_SERIAL: row['serial_1'],
            c.HH_PER_NUM: row['pernum_1'],
            c.HH_RACE: race_dict[row['race_1'].split('.')[0]],
            c.HH_BPL: __get_bpl__(bpl_dict, row['bpl_1']),
            c.HH_FNAME: row['namefrst_1'],
            c.HH_SNAME: row['namelast_1'],
            c.HH_BYEAR: __get_byear__(row['age_1'], row['year_1']),
            c.HH_STATE: __get_state__(state_dict, row['stateicp_1']),
            c.HH_CITY: __get_city__(city_dict, row['city_1']),
            c.HH_OCC: __get_occ__(occ_dict, row['occ1950_1'])
          }
          writer.writerow(p1_dict)
          person_dict[p_str] = p1_dict

      # Writing second census year related record
      if row['namelast_2'] == '' and row['namefrst_2'] == '':
        pass
      elif row['relate_2'] in ['1.0', '2.0', '3.0']:
        p_str = '%s-%s-%s' % (row['year_2'], row['serial_2'], row['pernum_2'])
        if p_str in person_dict.iterkeys():
          p2_dict = person_dict[p_str]
          assert p2_dict[c.HH_FNAME] == row['namefrst_2']
          assert p2_dict[c.HH_SNAME] == row['namelast_2']
          p2_id, p2_role = p2_dict[c.HH_ID], p2_dict[c.HH_ROLE]
        else:
          pid += 1
          p2_id = pid
          p2_role = __get_role__(row['relate_2'], row['sex_2'])
          p2_dict = {
            c.HH_ID: p2_id,
            c.HH_GENDER: __get_gender(row['sex_2']),
            c.HH_ROLE: p2_role,
            c.HH_YEAR: __get__year__(row['year_2']),
            c.HH_SERIAL: row['serial_2'],
            c.HH_PER_NUM: row['pernum_2'],
            c.HH_RACE: race_dict[row['race_2'].split('.')[0]],
            c.HH_BPL: __get_bpl__(bpl_dict, row['bpl_2']),
            c.HH_FNAME: row['namefrst_2'],
            c.HH_SNAME: row['namelast_2'],
            c.HH_BYEAR: __get_byear__(row['age_2'], row['year_2']),
            c.HH_STATE: __get_state__(state_dict, row['stateicp_2']),
            c.HH_CITY: __get_city__(city_dict, row['city_2']),
            c.HH_OCC: __get_occ__(occ_dict, row['occ1950_2'])
          }
          person_dict[p_str] = p2_dict
          writer.writerow(p2_dict)

      # Generate ground truth links
      if p1_id is not None and p2_id is not None:
        assert p1_dict[c.HH_YEAR] != p2_dict[c.HH_YEAR]
        link = '%s-%s' % (p1_role, p2_role)
        key = tuple(sorted({p1_id, p2_id}))
        gt_dict[link].add(key)


    for link, link_list in gt_dict.iteritems():
      logging.info('Ground truth %s - %s' % (link, len(link_list)))
    pickle.dump(gt_dict, open(settings.gt_file, 'wb'))


def transform_stata_csv():
  file_list = ['linked_1850-1880_couples.dta', 'linked_1850-1880_females.dta',
               'linked_1850-1880_males.dta',
               'linked_1860-1880_couples.dta', 'linked_1860-1880_females.dta',
               'linked_1860-1880_males.dta', 'linked_1870-1880_couples.dta',
               'linked_1870-1880_females.dta', 'linked_1870-1880_males.dta',
               'linked_1880-1900_couples.dta', 'linked_1880-1900_females.dta',
               'linked_1880-1900_males.dta', 'linked_1880-1910_couples.dta',
               'linked_1880-1910_females.dta', 'linked_1880-1910_males.dta',
               'linked_1880-1920_couples.dta', 'linked_1880-1920_females.dta',
               'linked_1880-1920_males.dta', 'linked_1880-1930_couples.dta',
               'linked_1880-1930_females.dta', 'linked_1880-1930_males.dta']

  for filename in file_list:
    df = pd.read_stata(settings.data_set_dir + filename,
                       convert_categoricals=False, preserve_dtypes=False)
    csv_filename = filename.split('.')[0] + '.csv'
    print '%s %s' % (csv_filename, df.shape[0])
    df.to_csv(settings.data_set_dir + csv_filename, encoding='utf-8')


def analyse_hh():
  males_file = csv.DictReader((open(settings.males_file, 'r')))
  females_file = csv.DictReader((open(settings.females_file, 'r')))
  couples_file = csv.DictReader((open(settings.couples_file, 'r')))


  males_hh_set = set()
  for row in males_file:
    males_hh_set.add(row['serial_1'])
    males_hh_set.add(row['serial_2'])

  females_hh_set = set()
  for row in females_file:
    females_hh_set.add(row['serial_1'])
    females_hh_set.add(row['serial_2'])

  couples_hh_set = set()
  for row in couples_file:
    couples_hh_set.add(row['serial_1'])
    couples_hh_set.add(row['serial_2'])

  print 'couples households %s' % len(couples_hh_set)
  print 'female households %s' % len(females_hh_set)
  print 'male households %s' % len(males_hh_set)

  print 'm-f households %s' % len(males_hh_set & females_hh_set)
  print 'c-m households %s' % len(males_hh_set & couples_hh_set)
  print 'c-f households %s' % len(couples_hh_set & females_hh_set)
  print 'c-f-m households %s' % len(couples_hh_set & females_hh_set & males_hh_set)




if __name__ == '__main__':
  c.data_set = 'ipums'
  settings.__init_settings__(0.0, 0.0)

  race_dict = __retrieve_index__(settings.data_set_dir + 'race.csv')
  bpl_dict = __retrieve_index__(settings.data_set_dir + 'bpl.csv')
  occ_dict = __retrieve_index__(settings.data_set_dir + 'occ.csv')
  city_dict = __retrieve_index__(settings.data_set_dir + 'city.csv')
  state_dict = __retrieve_index__(settings.data_set_dir + 'state.csv')

  # transform_stata_csv()
  generate_census_csv(race_dict, bpl_dict, occ_dict, city_dict, state_dict)
  # analyse_hh()