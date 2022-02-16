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

"""
Module to load songs data of 'Million Songs Data' and 'Music Brainz' data.

We already have separated data.

Million Songs Data  : Separated to msd1 and msd2
Music Brainz        : Separated to mb1 and mb2

IDs of data set x1 starts with '10'
IDs of data set x2 starts with '20'
"""
import csv
import os
import pickle
from collections import defaultdict

import networkx as nx

from common import settings, constants as c
from data import model


def enumerate_records(dataset, songs_file):
  songs_dict = {}
  for ds_id in [1, 2]:
    # Retrieve data set
    songs_rec_filename = settings.data_set_dir + '/{}{}.csv'.format(dataset,
                                                                    ds_id)
    songs_rec_list = [r for r in csv.DictReader(open(songs_rec_filename, 'r'))]

    for rec in songs_rec_list:
      sid = int(rec[c.SG_ID])
      song = model.new_song(sid, rec[c.SG_TITLE], rec[c.SG_ALBUM],
                            rec[c.SG_ARTIST], rec[c.SG_YEAR], rec[c.SG_LENGTH])
      songs_dict[sid] = song

  nx.write_gpickle(songs_dict, songs_file)


# ---------------------------------------------------------------------------
# Ground Truth Record Pair Enumeration
# ---------------------------------------------------------------------------

def retrieve_ground_truth_links(pairs_file):
  filename = settings.gt_file
  if os.path.isfile(filename):
    gt_link_dict = pickle.load(open(filename, 'rb'))
  else:
    # Extracting ground truth links
    gt_link_dict = defaultdict(list)
    link = 'S-S'
    with open(pairs_file, 'r') as gtf:
      reader = csv.DictReader(gtf)
      for row in reader:
        if row['label'] == '1':
          key = tuple(sorted({int(row['table1.{}'.format(c.SG_ID)]),
                              int(row['table2.{}'.format(c.SG_ID)])}))
          gt_link_dict[link].append(key)
    pickle.dump(gt_link_dict, open(settings.gt_file, 'wb'))
  return gt_link_dict
