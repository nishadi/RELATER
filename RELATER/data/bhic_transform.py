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
from collections import defaultdict
import pandas as pd
from lxml import etree
from common import constants as c, settings
import matplotlib.pyplot as plt

from common.enums import Role, Sex

BIRTH_XML_FILE = 'bhic_a2a_bs_g-202005.xml'
MARRIAGE_XML_FILE = 'bhic_a2a_bs_h-202005.xml'
DEATH_XML_FILE = 'bhic_a2a_bs_o-202005.xml'

start = 1760
b_end = 1970
m_start = 1760
end = 1970


def get_tags():
  for event, file in [('BIRTH', BIRTH_XML_FILE),
                      ('MARRIAGE', MARRIAGE_XML_FILE),
                      ('DEATH', DEATH_XML_FILE)]:
    print event
    tag_set = set()
    context = etree.iterparse(settings.data_set_dir + file,
                              events=('end',))

    for event, element in context:
      tag_set.add(element.tag)

    for tag in sorted(tag_set):
      print tag


def __get_text__(element):
  if element is not None:
    str = element.text.encode('utf-8')
    if ',' in str:
      return '"' + str + '"'
    return str
  return None


def __get_gender__(element):
  if element is not None:
    gender_str = element.text.encode('utf-8')
    if gender_str == 'Man':
      return 'm'
    elif gender_str == 'Vrouw':
      return 'f'
    else:
      return None
  return None


def transform_deaths():
  tg_header = '{http://www.openarchives.org/OAI/2.0/}header'
  tg_identifier = '/{http://www.openarchives.org/OAI/2.0/}identifier'
  tg_metadata = '{http://www.openarchives.org/OAI/2.0/}metadata/{http://Mindbus.nl/A2A}A2A'
  tg_person = '{http://Mindbus.nl/A2A}Person'
  tg_person_gender = '{http://Mindbus.nl/A2A}Gender'
  tg_person_name = '{http://Mindbus.nl/A2A}PersonName'
  tg_person_fname = '{http://Mindbus.nl/A2A}PersonNameFirstName'
  tg_person_sname = '{http://Mindbus.nl/A2A}PersonNameLastName'
  tg_age = '{http://Mindbus.nl/A2A}Age/{http://Mindbus.nl/A2A}PersonAgeLiteral'
  tg_event = '{http://Mindbus.nl/A2A}Event'
  tg_event_type = '{http://Mindbus.nl/A2A}EventType'
  tg_event_date = '{http://Mindbus.nl/A2A}EventDate'
  tg_event_year = '{http://Mindbus.nl/A2A}Year'
  tg_event_month = '{http://Mindbus.nl/A2A}Month'
  tg_event_day = '{http://Mindbus.nl/A2A}Day'
  tg_event_place = '{http://Mindbus.nl/A2A}EventPlace'
  tg_place = '{http://Mindbus.nl/A2A}Place'
  tg_relationship = '{http://Mindbus.nl/A2A}RelationEP'
  tg_person_ref = '{http://Mindbus.nl/A2A}PersonKeyRef'
  tg_relation_type = '{http://Mindbus.nl/A2A}RelationType'

  columns = ['d_id', 'event_place', 'event_date', 'event_year',
             'event_month', 'event_day',
             'deceased_id', 'deceased_fname', 'deceased_sname',
             'deceased_gender', 'deceased_age',
             'mother_id', 'mother_fname', 'mother_sname',
             'father_id', 'father_fname', 'father_sname']

  min_year = 3000
  max_year = 0
  record_count = 0
  with open(settings.death_file, 'w') as death_csv:

    writer = csv.DictWriter(death_csv, columns)
    writer.writeheader()

    context = etree.iterparse(settings.data_set_dir + DEATH_XML_FILE,
                              events=('end',),
                              tag='{http://www.openarchives.org/OAI/2.0/}record')

    for event, element in context:

      metadata = element.find(tg_metadata)

      # Assert the event is a birth
      event_type = metadata.find('/'.join([tg_event, tg_event_type]))
      assert event_type.text == 'Overlijden'

      event_year = __get_text__(
        metadata.find('/'.join([tg_event, tg_event_date,
                                tg_event_year])))
      if event_year is None or event_year == '':
        continue
      event_year = int(event_year)
      if (event_year < start) or (event_year > end):
        continue

      if event_year < min_year:
        min_year = event_year
      if event_year > max_year:
        max_year = event_year

      event_dict = {
        'd_id': element.find('/'.join([tg_header, tg_identifier])).text,
        'event_place': __get_text__(metadata.find('/'.join([tg_event,
                                                            tg_event_place,
                                                            tg_place]))),
        'event_year': event_year,
        'event_month': __get_text__(
          metadata.find('/'.join([tg_event, tg_event_date,
                                  tg_event_month]))),
        'event_day': __get_text__(
          metadata.find('/'.join([tg_event, tg_event_date,
                                  tg_event_day])))}
      event_dict['event_date'] = '%s/%s/%s' % (event_dict['event_day'],
                                               event_dict['event_month'],
                                               event_dict['event_year'])

      person_dict = {}
      for person in metadata.iterchildren(tag=tg_person):
        p = {
          'fname': __get_text__(person.find('/'.join([tg_person_name,
                                                      tg_person_fname]))),
          'sname': __get_text__(person.find('/'.join([tg_person_name,
                                                      tg_person_sname]))),
          'gender': __get_gender__(person.find(tg_person_gender)),
          'age': __get_text__(person.find(tg_age))

        }
        person_dict[person.get('pid')] = p

      for relationship in metadata.iterchildren(tag=tg_relationship):

        person_ref = relationship.find(tg_person_ref).text
        person = person_dict[person_ref]
        relation = relationship.find(tg_relation_type).text
        pid = person_ref.split(':')[1]

        if relation == 'Moeder':
          event_dict['mother_id'] = pid
          event_dict['mother_fname'] = person['fname']
          event_dict['mother_sname'] = person['sname']
        elif relation == 'Vader':
          event_dict['father_id'] = pid
          event_dict['father_fname'] = person['fname']
          event_dict['father_sname'] = person['sname']
        elif relation == 'Overledene':
          event_dict['deceased_id'] = pid
          event_dict['deceased_fname'] = person['fname']
          event_dict['deceased_sname'] = person['sname']
          event_dict['deceased_gender'] = person['gender']
          event_dict['deceased_age'] = person['age']
        else:
          raise Exception('Invalid person %s' % relation)

      record_count += 1
      writer.writerow(event_dict)

      # Now clearing the element and the parent of elements
      element.clear()
      while element.getprevious() is not None:
        del element.getparent()[0]

  logging.info('Death {} records'.format(record_count))
  logging.info('\t min year {}'.format(min_year))
  logging.info('\t max year {}'.format(max_year))


def transform_marriages():
  tg_header = '{http://www.openarchives.org/OAI/2.0/}header'
  tg_identifier = '/{http://www.openarchives.org/OAI/2.0/}identifier'
  tg_metadata = '{http://www.openarchives.org/OAI/2.0/}metadata/{http://Mindbus.nl/A2A}A2A'
  tg_person = '{http://Mindbus.nl/A2A}Person'
  tg_person_gender = '{http://Mindbus.nl/A2A}Gender'
  tg_person_name = '{http://Mindbus.nl/A2A}PersonName'
  tg_person_fname = '{http://Mindbus.nl/A2A}PersonNameFirstName'
  tg_person_sname = '{http://Mindbus.nl/A2A}PersonNameLastName'
  tg_person_bplace = '{http://Mindbus.nl/A2A}BirthPlace/{http://Mindbus.nl/A2A}Place'
  tg_person_bdate = '{http://Mindbus.nl/A2A}BirthDate'
  tg_person_bdate_year = tg_person_bdate + '/{http://Mindbus.nl/A2A}Year'
  tg_person_bdate_month = tg_person_bdate + '/{http://Mindbus.nl/A2A}Month'
  tg_person_bdate_day = tg_person_bdate + '/{http://Mindbus.nl/A2A}Day'
  tg_age = '{http://Mindbus.nl/A2A}Age/{http://Mindbus.nl/A2A}PersonAgeLiteral'
  tg_event = '{http://Mindbus.nl/A2A}Event'
  tg_event_type = '{http://Mindbus.nl/A2A}EventType'
  tg_event_date = '{http://Mindbus.nl/A2A}EventDate'
  tg_event_year = '{http://Mindbus.nl/A2A}Year'
  tg_event_month = '{http://Mindbus.nl/A2A}Month'
  tg_event_day = '{http://Mindbus.nl/A2A}Day'
  tg_event_place = '{http://Mindbus.nl/A2A}EventPlace'
  tg_place = '{http://Mindbus.nl/A2A}Place'
  tg_relationship = '{http://Mindbus.nl/A2A}RelationEP'
  tg_person_ref = '{http://Mindbus.nl/A2A}PersonKeyRef'
  tg_relation_type = '{http://Mindbus.nl/A2A}RelationType'
  tg_source_place = '{http://Mindbus.nl/A2A}Source/{http://Mindbus.nl/A2A}SourcePlace/{http://Mindbus.nl/A2A}Place'

  columns = ['m_id', 'event_place', 'event_date', 'event_year',
             'event_month', 'event_day',
             'bride_id', 'bride_fname', 'bride_sname', 'bride_birth_place',
             'bride_age', 'bride_birth_year', 'bride_birth_month',
             'bride_birth_day',
             'bride_mother_id', 'bride_mother_fname', 'bride_mother_sname',
             'bride_father_id', 'bride_father_fname', 'bride_father_sname',
             'groom_id', 'groom_fname', 'groom_sname', 'groom_birth_place',
             'groom_age', 'groom_birth_year', 'groom_birth_month',
             'groom_birth_day',
             'groom_mother_id', 'groom_mother_fname', 'groom_mother_sname',
             'groom_father_id', 'groom_father_fname', 'groom_father_sname']

  min_year = 3000
  max_year = 0
  record_count = 0
  with open(settings.marriage_file, 'w') as marriage_csv:

    writer = csv.DictWriter(marriage_csv, columns)
    writer.writeheader()

    context = etree.iterparse(settings.data_set_dir + MARRIAGE_XML_FILE,
                              events=('end',),
                              tag='{http://www.openarchives.org/OAI/2.0/}record')

    for event, element in context:

      metadata = element.find(tg_metadata)

      # Assert the event is a birth
      event_type = metadata.find('/'.join([tg_event, tg_event_type]))
      assert event_type.text == 'Huwelijk'

      event_year = __get_text__(
        metadata.find('/'.join([tg_event, tg_event_date,
                                tg_event_year])))
      if event_year is None or event_year == '':
        continue
      event_year = int(event_year)
      if (event_year < m_start) or (event_year > end):
        continue

      if event_year < min_year:
        min_year = event_year
      if event_year > max_year:
        max_year = event_year

      source_place = __get_text__(metadata.find(tg_source_place))

      event_dict = {
        'm_id': element.find('/'.join([tg_header, tg_identifier])).text,
        'event_place': __get_text__(metadata.find('/'.join([
          tg_event, tg_event_place, tg_place]))),
        'event_year': event_year,
        'event_month': __get_text__(
          metadata.find('/'.join([tg_event, tg_event_date,
                                  tg_event_month]))),
        'event_day': __get_text__(
          metadata.find('/'.join([tg_event, tg_event_date,
                                  tg_event_day])))}
      event_dict['event_date'] = '%s/%s/%s' % (event_dict['event_day'],
                                               event_dict['event_month'],
                                               event_dict['event_year'])

      assert source_place == event_dict['event_place']

      person_dict = {}
      for person in metadata.iterchildren(tag=tg_person):
        p = {
          'fname': __get_text__(person.find('/'.join([tg_person_name,
                                                      tg_person_fname]))),
          'sname': __get_text__(person.find('/'.join([tg_person_name,
                                                      tg_person_sname]))),
          'gender': __get_gender__(person.find(tg_person_gender)),
          'birth_place': __get_text__(person.find(tg_person_bplace)),
          'age': __get_text__(person.find(tg_age)),
          'birth_year': __get_text__(person.find(tg_person_bdate_year)),
          'birth_month': __get_text__(person.find(tg_person_bdate_month)),
          'birth_day': __get_text__(person.find(tg_person_bdate_day))
        }
        person_dict[person.get('pid')] = p

      for relationship in metadata.iterchildren(tag=tg_relationship):

        person_ref = relationship.find(tg_person_ref).text
        person = person_dict[person_ref]
        relation = relationship.find(tg_relation_type).text
        pid = person_ref.split(':')[1]
        if relation == 'Bruidegom':
          event_dict['groom_id'] = pid
          event_dict['groom_fname'] = person['fname']
          event_dict['groom_sname'] = person['sname']
          event_dict['groom_birth_place'] = person['birth_place']
          event_dict['groom_age'] = person['age']
          event_dict['groom_birth_year'] = person['birth_year']
          event_dict['groom_birth_month'] = person['birth_month']
          event_dict['groom_birth_day'] = person['birth_day']
        elif relation == 'Moeder van de bruidegom':
          event_dict['groom_mother_id'] = pid
          event_dict['groom_mother_fname'] = person['fname']
          event_dict['groom_mother_sname'] = person['sname']
          assert person['age'] is None
          assert person['birth_year'] is None
          assert person['birth_place'] is None
        elif relation == 'Vader van de bruidegom':
          event_dict['groom_father_id'] = pid
          event_dict['groom_father_fname'] = person['fname']
          event_dict['groom_father_sname'] = person['sname']
          assert person['age'] is None
          assert person['birth_year'] is None
          assert person['birth_place'] is None
        elif relation == 'Bruid':
          event_dict['bride_id'] = pid
          event_dict['bride_fname'] = person['fname']
          event_dict['bride_sname'] = person['sname']
          event_dict['bride_birth_place'] = person['birth_place']
          event_dict['bride_age'] = person['age']
          event_dict['bride_birth_year'] = person['birth_year']
          event_dict['bride_birth_month'] = person['birth_month']
          event_dict['bride_birth_day'] = person['birth_day']
        elif relation == 'Moeder van de bruid':
          event_dict['bride_mother_id'] = pid
          event_dict['bride_mother_fname'] = person['fname']
          event_dict['bride_mother_sname'] = person['sname']
          assert person['age'] is None
          assert person['birth_year'] is None
          assert person['birth_place'] is None
        elif relation == 'Vader van de bruid':
          event_dict['bride_father_id'] = pid
          event_dict['bride_father_fname'] = person['fname']
          event_dict['bride_father_sname'] = person['sname']
          assert person['age'] is None
          assert person['birth_year'] is None
          assert person['birth_place'] is None
        else:
          raise Exception('Invalid person %s' % relation)

      record_count += 1
      writer.writerow(event_dict)

      # Now clearing the element and the parent of elements
      element.clear()
      while element.getprevious() is not None:
        del element.getparent()[0]

  logging.info('Marriage {} records'.format(record_count))
  logging.info('\t min year {}'.format(min_year))
  logging.info('\t max year {}'.format(max_year))



def transform_births():
  # XML Tags
  tg_header = '{http://www.openarchives.org/OAI/2.0/}header'
  tg_identifier = '/{http://www.openarchives.org/OAI/2.0/}identifier'
  tg_metadata = '{http://www.openarchives.org/OAI/2.0/}metadata/{http://Mindbus.nl/A2A}A2A'
  tg_person = '{http://Mindbus.nl/A2A}Person'
  tg_person_gender = '{http://Mindbus.nl/A2A}Gender'
  tg_person_name = '{http://Mindbus.nl/A2A}PersonName'
  tg_person_fname = '{http://Mindbus.nl/A2A}PersonNameFirstName'
  tg_person_sname = '{http://Mindbus.nl/A2A}PersonNameLastName'
  tg_relationship = '{http://Mindbus.nl/A2A}RelationEP'
  tg_person_ref = '{http://Mindbus.nl/A2A}PersonKeyRef'
  tg_relation_type = '{http://Mindbus.nl/A2A}RelationType'
  tg_event_place = '{http://Mindbus.nl/A2A}EventPlace'
  tg_birth_place = '{http://Mindbus.nl/A2A}BirthPlace/{http://Mindbus.nl/A2A}Place'
  tg_source_place = '{http://Mindbus.nl/A2A}Source/{http://Mindbus.nl/A2A}SourcePlace/{http://Mindbus.nl/A2A}Place'
  tg_place = '{http://Mindbus.nl/A2A}Place'
  tg_event = '{http://Mindbus.nl/A2A}Event'
  tg_event_type = '{http://Mindbus.nl/A2A}EventType'
  tg_event_date = '{http://Mindbus.nl/A2A}EventDate'
  tg_event_year = '{http://Mindbus.nl/A2A}Year'
  tg_event_month = '{http://Mindbus.nl/A2A}Month'
  tg_event_day = '{http://Mindbus.nl/A2A}Day'

  columns = ['b_id', 'event_place', 'event_date', 'event_year',
             'event_month', 'event_day',
             'mother_id', 'mother_fname', 'mother_sname',
             'father_id', 'father_fname', 'father_sname',
             'child_id', 'child_fname', 'child_sname', 'child_gender',
             'birth_place', 'source_place']

  min_year = 3000
  max_year = 0
  record_count = 0
  with open(settings.birth_file, 'w') as birth_csv:

    writer = csv.DictWriter(birth_csv, columns)
    writer.writeheader()

    context = etree.iterparse(settings.data_set_dir + BIRTH_XML_FILE,
                              events=('end',),
                              tag='{http://www.openarchives.org/OAI/2.0/}record')

    for event, element in context:

      metadata = element.find(tg_metadata)

      # Assert the event is a birth
      event_type = metadata.find('/'.join([tg_event, tg_event_type]))
      assert event_type.text == 'Geboorte'

      event_year = __get_text__(
          metadata.find('/'.join([tg_event, tg_event_date, tg_event_year])))
      if event_year is None or event_year == '':
        continue
      event_year = int(event_year)
      if (event_year < start) or (event_year > b_end):
        continue


      if event_year < min_year:
        min_year = event_year
      if event_year > max_year:
        max_year = event_year

      event_dict = {
        'b_id': element.find('/'.join([tg_header, tg_identifier])).text,
        'event_place': __get_text__(metadata.find('/'.join([tg_event,
                                                            tg_event_place,
                                                            tg_place]))),
        'event_year': event_year,
        'event_month': __get_text__(
          metadata.find('/'.join([tg_event, tg_event_date,
                                  tg_event_month]))),
        'event_day': __get_text__(
          metadata.find('/'.join([tg_event, tg_event_date,
                                  tg_event_day]))),
        'source_place': __get_text__(metadata.find(tg_source_place))
      }
      event_dict['event_date'] = '%s/%s/%s' % (event_dict['event_day'],
                                               event_dict['event_month'],
                                               event_dict['event_year'])

      person_dict = {}
      for person in metadata.iterchildren(tag=tg_person):
        p = {
          'fname': __get_text__(person.find('/'.join([tg_person_name,
                                                      tg_person_fname]))),
          'sname': __get_text__(person.find('/'.join([tg_person_name,
                                                      tg_person_sname]))),
          'gender': __get_gender__(person.find(tg_person_gender)),
          'birth_place': __get_text__(person.find(tg_birth_place))

        }
        person_dict[person.get('pid')] = p

      for relationship in metadata.iterchildren(tag=tg_relationship):

        person_ref = relationship.find(tg_person_ref).text
        person = person_dict[person_ref]
        relation = relationship.find(tg_relation_type).text
        pid = person_ref.split(':')[1]

        if relation == 'Moeder':
          event_dict['mother_id'] = pid
          event_dict['mother_fname'] = person['fname']
          event_dict['mother_sname'] = person['sname']
        elif relation == 'Vader':
          event_dict['father_id'] = pid
          event_dict['father_fname'] = person['fname']
          event_dict['father_sname'] = person['sname']
        elif relation == 'Kind':
          event_dict['child_id'] = pid
          event_dict['child_fname'] = person['fname']
          event_dict['child_sname'] = person['sname']
          event_dict['child_gender'] = person['gender']
          event_dict['birth_place'] = person['birth_place']
        else:
          raise Exception('Invalid person %s' % relation)

      record_count += 1
      writer.writerow(event_dict)

      # Now clearing the element and the parent of elements
      element.clear()
      while element.getprevious() is not None:
        del element.getparent()[0]

  logging.info('Birth {} records'.format(record_count))
  logging.info('\t min year {}'.format(min_year))
  logging.info('\t max year {}'.format(max_year))



def data_stats():
  if not os.path.isdir(settings.stats_dir):
    os.makedirs(settings.stats_dir)

  for event, file in [('BIRTH', settings.birth_file),
                      ('MARRIAGE', settings.marriage_file),
                      ('DEATH', settings.death_file)]:
    df = pd.read_csv(file)
    print df.info()

    for field in df.columns:
      file_name = os.path.join(settings.stats_dir, '%s_%s.csv' % (event, field))
      df[field].value_counts().to_csv(file_name)

      logging.info('%s_%s_unique.csv' % (event, field))
      logging.info(df[field].unique())




def analyse_marriages():
  marriage_df = pd.read_csv(settings.marriage_file)
  marriage_df.hist(bins=20, figsize=(20, 15))
  plt.savefig('marriages.png')
  plt.clf()

  marriage_w_age_df = marriage_df[marriage_df.bride_age.notnull()]
  print 'Bride age'
  print marriage_w_age_df.info()
  marriage_w_age_df.hist(figsize=(20, 15))
  plt.savefig('bride_age.png')
  plt.clf()

  marriage_w_age_df = marriage_df[marriage_df.groom_age.notnull()]
  print 'Groom age'
  print marriage_w_age_df.info()
  marriage_w_age_df.hist(figsize=(20, 15))
  plt.savefig('groom_age.png')
  plt.clf()

  marriage_w_age_df = marriage_df[(marriage_df.groom_age.notnull()) & (
    marriage_df.bride_age.notnull())]
  print 'Bride and Groom age'
  print marriage_w_age_df.info()
  marriage_w_age_df.hist(figsize=(20, 15))
  plt.savefig('bride_groom_age.png')
  plt.clf()


def analyse_ids():
  people_list = pickle.load(open(settings.people_file, 'rb'))
  people_list = people_list.values()

  bb_list = filter(lambda p: p[c.I_ROLE] == 'Bb',
                   people_list)
  print(len(bb_list))
  bm_list = filter(lambda p: p[c.I_ROLE] == 'Bp' and p[c.I_SEX] == 'M',
                   people_list)
  bm_id_list = map(lambda p: p[c.I_CERT_ID], bm_list)

  id_parts_dict = defaultdict(list)
  for id in bm_id_list:
    for i, id_part in enumerate(id.split('-')):
      id_parts_dict[i].append(id_part)

  print 'Total %s' % len(bm_id_list)
  for i in id_parts_dict.iterkeys():
    print 'ID part %s : all %s' % (i, len(id_parts_dict[i]))
    print 'ID part %s : unique %s' % (i, len(set(id_parts_dict[i])))


def analyse_min_max_years(filename):
  min_year = 3000
  max_year = 0
  with open(filename, 'r') as data:
    reader = csv.DictReader(data)
    for record in reader:
      year = record['event_year']
      if year is None or year == '':
        continue
      year = int(year)
      if year < min_year:
        min_year = year
      if year > max_year:
        max_year = year
  print '\tMin year : {}'.format(min_year)
  print '\tMax year : {}'.format(max_year)


def count_records():
  for filename, s, e in [(settings.birth_file, start, b_end),
                         (settings.marriage_file, m_start, end),
                         (settings.death_file, start, end)]:
    print filename
    print '\t {} - {}'.format(s, e)
    with open(filename, 'r') as data:
      reader = csv.DictReader(data)
      count = 0
      for record in reader:
        year = record['event_year']
        if year is None or year == '':
          continue
        year = int(year)
        if year >= s and year <= e:
          count += 1
    print '\t Count {}'.format(count)


def generate_groundtruth_links():
  print 'started'

  people_list = pickle.load(open(settings.people_file, 'rb'))

  bb_f_list = filter(lambda p: p[c.I_ROLE] == Role.Bb and p[c.I_SEX] == Sex.F,
                     people_list)
  bb_m_list = filter(lambda p: p[c.I_ROLE] == Role.Bb and p[c.I_SEX] == Sex.M,
                     people_list)
  mm_f_list = filter(lambda p: p[c.I_ROLE] == Role.Mm and p[c.I_SEX] == Sex.F,
                     people_list)
  mm_m_list = filter(lambda p: p[c.I_ROLE] == Role.Mm and p[c.I_SEX] == Sex.M,
                     people_list)

  gt_dict = defaultdict(list)
  for bb_list, mm_list in [(bb_f_list, mm_f_list), (bb_m_list, mm_m_list)]:

    print 'Bb list len %s' % len(bb_list)
    print 'Mm list len %s' % len(mm_list)
    for bb in bb_list:
      for mm in mm_list:
        if bb[c.I_FNAME] == mm[c.I_FNAME] and bb[c.I_SNAME] == mm[c.I_SNAME] \
            and bb[c.I_BIRTH_YEAR] == mm[c.I_BIRTH_YEAR]:
          key = tuple(sorted({bb[c.I_ID], mm[c.I_ID]}))
          gt_dict['Bb-Mm'].append(key)

  pickle.dump(gt_dict, open(settings.gt_file, 'wb'))
  print 'Gt links Bb-Mm %s' % len(gt_dict['Bb-Mm'])

def aggregate_ground_truth():
  gt_filenames = [
    '{}gt-bhic-exact.pickle'.format(settings.output_home_directory),
    '{}gt-bhic-00.pickle'.format(settings.output_home_directory),
    '{}gt-bhic-01.pickle'.format(settings.output_home_directory),
    '{}gt-bhic-10.pickle'.format(settings.output_home_directory),
    '{}gt-bhic-11.pickle'.format(settings.output_home_directory),
    '{}gt-bhic-20.pickle'.format(settings.output_home_directory),
    '{}gt-bhic-21.pickle'.format(settings.output_home_directory)
  ]

  gt_dict = defaultdict(set)
  for filename in gt_filenames:
    tmp_gt_dict = pickle.load(open(filename, 'r'))
    for link in tmp_gt_dict.iterkeys():
      gt_dict[link].update(tmp_gt_dict[link])

  for link, pair_list in gt_dict.iteritems():
    logging.info('{} {}'.format(link, len(pair_list)))
  pickle.dump(gt_dict, open(settings.gt_file, 'wb'))



if __name__ == '__main__':
  c.data_set = 'bhic'
  settings.__init_settings__(0.0, 0.0, 'Bb')

  log_file_name = settings.output_home_directory + 'data-gen.log'
  logging.basicConfig(filename=log_file_name, level=logging.INFO)
  # aggregate_ground_truth()

  settings.birth_file = settings.birth_file.format(1760)
  settings.marriage_file = settings.marriage_file.format(1760)
  settings.death_file = settings.death_file.format(1760)

  # analyse_ids()
  transform_births()
  transform_marriages()
  transform_deaths()
  # data_stats()
  # get_tags()
  # analyse_marriages()
  # generate_groundtruth_links()

  # print 'Births'
  # analyse_min_max_years(settings.birth_file)
  #
  # print 'Marriages'
  # analyse_min_max_years(settings.marriage_file)

  # count_records()
