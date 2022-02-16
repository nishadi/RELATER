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
import re
import string
from collections import defaultdict
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from data import model
from common import settings, constants as c

DATA_SET1 = settings.data_set_dir + '/tableA.csv'
DATA_SET2 = settings.data_set_dir + '/tableB.csv'
GT_TEST_FILE = settings.data_set_dir + '/test.csv'
GT_TRAIN_FILE = settings.data_set_dir +  '/train.csv'
GT_VALID_FILE = settings.data_set_dir + '/valid.csv'


def __retrieve_data__(filename, origin_id):
  """
  Method to retrieve data from the original CSV files.

  :param filename: File name
  :param cert_id: Certificate ID
  :return: A dictionary with IDs as keys
  """
  record_dict = {}

  # Read file
  with open(filename, 'r') as data:
    reader = csv.DictReader(data)
    i = 0
    for record in reader:
      # if i in lines_to_grab:
      record_dict[record[origin_id]] = record
      i += 1
  logging.info(
    'Retrieved records of {} : {}'.format(origin_id, len(record_dict.keys())))
  return record_dict


def enumerate_records(author_file, pubs_file):
  """
  Method to enumerate people records and publication records from
  original data set

  :return: A dictionary of people with person ID as keys.
  """

  ds1 = __retrieve_data__(DATA_SET1, c.BB_ID) #DBLP
  ds2 = __retrieve_data__(DATA_SET2, c.BB_ID) #ACM

  for ds, ds_name in [(ds1, 'A'), (ds2, 'B')]:

    people_dict = {}
    p_id = 0
    publication_dict = {}
    pub_id = 0

    for r in ds.itervalues():

      # Adding authors
      author_id_list = list()
      for author_fullname in r[c.BB_AUTHORS].strip().split(','):
        p_id += 1

        # Remove unicode bytes starting with & and ending with ;
        author_fullname = re.sub('\s&.*;\s', '', author_fullname)

        # Remove punctuations and digits and split by spaces
        for punct_char in string.punctuation:
          author_fullname = author_fullname.replace(punct_char, ' ')
        names = re.sub('\s+', ' ', author_fullname).strip().split(' ')

        fname, sname, mname = None, None, None
        if len(names) == 1:
          fname = names[0]
          sname = names[0]
        elif len(names) == 2:
          fname = names[0]
          sname = names[1]
        elif len(names) > 2:
          fname = names[0]
          mname = names[1]
          sname = names[2]
        else:
          raise Exception('No names')

        author = model.new_author(p_id, r[c.BB_ID], r[c.BB_YEAR],
                                  r[c.BB_VENUE], r[c.BB_TITLE], fname, sname,
                                  mname, author_fullname)
        people_dict[p_id] = author
        author_id_list.append(p_id)

      # Adding publication
      pub_id += 1
      publication = model.new_publication(pub_id, r[c.BB_ID], r[c.BB_YEAR],
                                          r[c.BB_VENUE], r[c.BB_TITLE],
                                          author_id_list)
      publication_dict[pub_id] = publication

      # Updating co authors and publication id for each author
      for author_id in author_id_list:
        author = people_dict[author_id]

        coauthor_list = [p for p in author_id_list if p != author_id]
        author[c.BB_I_COAUTHOR_LIST] = coauthor_list

        author[c.BB_I_PUBID] = pub_id

    nx.write_gpickle(people_dict, (author_file).format(ds_name))
    nx.write_gpickle(publication_dict, (pubs_file).format(ds_name))

    # columns = ['id', 'entity_id', 'freq', 'sex', 'role', 'origin_id', 'year',
    #            'venue', 'pub_name', 'fname', 'sname', 'mname', 'fullname',
    #            'coauthor_list', 'pubid']
    # raw_data_set_stats(people_dict, columns, 'authors-{}'.format(ds_name))
    #
    # columns = ['id', 'entity_id', 'freq', 'sex', 'role', 'origin_id', 'year',
    #            'venue', 'pub_name', 'author_list']
    # raw_data_set_stats(publication_dict, columns, 'pubs-{}'.format(ds_name))

  return people_dict, publication_dict


def raw_data_set_stats(data_dict, columns, name):
  if not os.path.isdir(settings.stats_dir):
    os.makedirs(settings.stats_dir)

  df = pd.DataFrame(data_dict.itervalues(), columns=columns)

  # Stats of numerical fields
  info_file = os.path.join(settings.stats_dir, 'numerical_fields.csv')
  df.describe().to_csv(info_file)

  df.hist()
  plt.savefig(settings.stats_dir + name + '.png')

  # Combined attributes
  if 'fname' in columns and 'sname' in columns:
    df['fname_sname'] = list(zip(df.fname, df.sname))
    df['fname_sname_mname'] = list(zip(df.fname, df.sname, df.mname))
    columns.append('fname_sname')
    columns.append('fname_sname_mname')

  # Stats of categorical fields
  for field in columns:
    if field in ['coauthor_list', 'author_list']:
      continue
    file_name = os.path.join(settings.stats_dir, '%s-%s.csv' % (field, name))
    df[field].value_counts().to_csv(file_name)


def enumerate_ground_truthlinks(datasetA_abbreviation, datasetB_abbreviation):
  filename = settings.gt_file
  if os.path.isfile(filename):
    gt_link_dict = pickle.load(open(filename, 'rb'))
  else:
    # Generate a dictionary to keep the mapping of origin id to pub id
    pub_origin_id_id_dict_A = dict()
    pub_origin_id_id_dict_B = dict()
    pub_dict = pickle.load(open(settings.pub_file, 'rb'))
    for pub in pub_dict.itervalues():
      if pub[c.I_ID].startswith(datasetA_abbreviation):
        pub_origin_id_id_dict_A[pub[c.BB_I_ORIGIN_ID]] = pub[c.I_ID]
      elif pub[c.I_ID].startswith(datasetB_abbreviation):
        pub_origin_id_id_dict_B[pub[c.BB_I_ORIGIN_ID]] = pub[c.I_ID]
      else:
        raise Exception('Wrong records')

    # Extracting ground truth links
    gt_link_dict = defaultdict(list)
    link = 'P-P'
    for filename in [GT_TEST_FILE, GT_TRAIN_FILE, GT_VALID_FILE]:
      with open(filename, 'r') as gtf:
        reader = csv.DictReader(gtf)
        for row in reader:
          if row['label'] == '1':
            key = tuple(sorted({pub_origin_id_id_dict_A[row['ltable_id']],
                                pub_origin_id_id_dict_B[row['rtable_id']]}))
            gt_link_dict[link].append(key)

    pickle.dump(gt_link_dict, open(settings.gt_file, 'wb'))
  return gt_link_dict


def analyse_authors():
  author_dict = pickle.load(open(settings.authors_file, 'rb'))

  none_dict = defaultdict(int)
  singleton_dict = defaultdict(int)
  for author in author_dict.itervalues():
    if author[c.I_ID].split('-')[0] == 'DD':
      for i in [c.BB_I_FNAME, c.BB_I_MNAME, c.BB_I_SNAME]:
        if author[i] is None:
          none_dict[i] += 1
        else:
          name = author[i].replace('.', '').replace(',', '')
          if len(name) == 1:
            singleton_dict[i] += 1

  for i, name in [(c.BB_I_FNAME, 'fname'), (c.BB_I_MNAME, 'mname'),
                  (c.BB_I_SNAME, 'sname')]:
    print 'Attribute %s ----------' % name
    print 'None count %s' % none_dict[i]
    print 'None percentage %s' % (none_dict[i] * 100.0 / len(author_dict))
    print 'Singleton count %s' % singleton_dict[i]
    print 'Singleton percentage %s' % (singleton_dict[i] * 100.0 / len(
      author_dict))


if __name__ == '__main__':
  c.data_set = 'DBLP-ACM'
  settings.__init_settings__()

  enumerate_ground_truthlinks()
  # enumerate_records()
  # enumerate_ground_truthlinks()

  # analyse_authors()
