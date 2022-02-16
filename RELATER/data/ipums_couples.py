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
from collections import defaultdict, Counter

from common import constants as c, settings


def summarize_link_types():
  hhpernum_linktype_dict = defaultdict(list)
  for filename in [settings.couples_file]:
    couples_csv = csv.DictReader(open(filename, 'r'))
    for row in couples_csv:
      hh1 = '%s-%s' % (row['serial_1'], row['pernum_1'])
      hh2 = '%s-%s' % (row['serial_2'], row['pernum_2'])

      # Check if the person in the household
      if row['namelast_1'] == '' and row['namefrst_1'] == '':
        pass
      elif row['relate_1'] in ['1.0', '2.0', '3.0']:
        hhpernum_linktype_dict[hh1].append(row['linktype'])

      if row['namelast_2'] == '' and row['namefrst_2'] == '':
        pass
      elif row['relate_2'] in ['1.0', '2.0', '3.0']:
        hhpernum_linktype_dict[hh2].append(row['linktype'])

  linked_people_count = 0
  non_linked_people_count = 0

  hh_size_dict = Counter()
  for person, link_list in hhpernum_linktype_dict.iteritems():
    if '0' in link_list or '1' in link_list or '5' in link_list:
      linked_people_count += 1
      hh = person.split('-')[0]
      hh_size_dict.update([hh])
    elif '9' in link_list:
      non_linked_people_count += 1
    else:
      raise Exception('Wrong link type')

  print 'Linked %s' % linked_people_count
  print 'Non linked %s' % non_linked_people_count

  print 'Linked hh average size %s' % (sum(hh_size_dict.values()) * 1.0 / len(
    hh_size_dict.values()))


if __name__ == '__main__':
  c.data_set = 'IPUMS'
  settings.__init_settings__(0.0, 0.0)
  summarize_link_types()
