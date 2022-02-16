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

import sys
import time

from common import hyperparams, constants as c, attributes_meta

# -----------------------------------------------------------------------------
# Basic Settings
# -----------------------------------------------------------------------------

role_type = 'Dd'


enable_analysis = False

parallel_process_count = 30

today_str = time.strftime("%Y%m%d", time.localtime())

# Atomic similarity calculation quotient for core attributes and extra
# attributes
beta = 0.7

numerical_comparison_dmax = 10.0
geographical_comparison_dmax = 20.0

year_gap = 2
marriage_year_gap = 20

eval_start_year = 1881
eval_end_year = 1890

attribute_init_function = ''

# -----------------------------------------------------------------------------
# Set data directories
# -----------------------------------------------------------------------------
data_set_dir = '../data/'
output_home_directory = '../out/{}/'

# -----------------------------------------------------------------------------
# Settings for raw input data sets, input files, output files
# -----------------------------------------------------------------------------

data_set = sys.argv[1]
in_atomic_t = float(sys.argv[2])
output_home_directory = output_home_directory.format(data_set)


# -----------------------------------------------------------------------------
# Settings for input files
# -----------------------------------------------------------------------------

if data_set == 'ios':
  data_set_dir = data_set_dir + 'isle-of-skye/'
  birth_file = data_set_dir + \
               'nov2016-with-ios-id/births-with-ios-ids-20161114.csv'
  marriage_file = data_set_dir + \
                  'nov2016-with-ios-id/marriages-with-ios-ids-20161122.csv'
  death_file = data_set_dir + \
               'nov2016-with-ios-id/deaths-with-ios-ids-20161122.csv'
  life_segments_file = data_set_dir + 'longitudinal-20160314.csv'
  attribute_init_function = attributes_meta.def_ios

elif data_set == 'kil':
  data_set_dir = data_set_dir + 'kilmarnock/'
  birth_file = data_set_dir + 'nov2016/births.csv'
  marriage_file = data_set_dir + 'nov2016/marriages.csv'
  death_file = data_set_dir + 'nov2016/deaths.csv'
  graph_file = '{}graph-{}.gpickle'.format(output_home_directory,
                                           str(in_atomic_t))
  attribute_init_function = attributes_meta.def_kil

elif data_set.startswith('bhic'):
  data_set = 'bhic'
  data_set_dir = data_set_dir + 'bhic/'
  birth_file = data_set_dir + 'births-{}.csv'
  marriage_file = data_set_dir + 'marriages-{}.csv'
  death_file = data_set_dir + 'deaths-{}.csv'
  attribute_init_function = attributes_meta.def15_bhic
elif data_set == 'ipums':
  role_type = 'F'
  data_set_dir = data_set_dir + 'ipums/'
  couples_file = data_set_dir + 'linked_1870-1880_couples.csv'
  females_file = data_set_dir + 'linked_1870-1880_females.csv'
  males_file = data_set_dir + 'linked_1870-1880_males.csv'
  census_file = data_set_dir + 'census_1870-1880_couples.csv'
  attribute_init_function = attributes_meta.def_hh
elif data_set == 'dblp-acm1' or data_set == 'dblp-scholar1':
  data_set_dir = data_set_dir + data_set + '/'
  authors_file = output_home_directory + '/authors.gpickle'
  pub_file = output_home_directory + '/publications.gpickle'
  attribute_init_function = ''
elif data_set == 'mb':
  c.SG_ID = 'TID'
  data_set_dir = data_set_dir + 'musicbrainz/'
  songs_file = output_home_directory + 'songs-S.gpickle'
  attribute_init_function = attributes_meta.def_sg
elif data_set == 'msd':
  data_set_dir = data_set_dir + 'msd/'
  songs_file = output_home_directory + 'songs-S.gpickle'
  attribute_init_function = attributes_meta.def_sg
else:
  raise Exception('Invalid data set name')

people_file = '{}people-{}-{}.gpickle'.format(output_home_directory,
                                              data_set, role_type)
atomic_node_file = '{}atomic-node-{}.gpickle'.format(output_home_directory,
                                                     data_set)

# -----------------------------------------------------------------------------
# Settings for output files
# -----------------------------------------------------------------------------
gt_file = '{}gt-{}.pickle'.format(output_home_directory,
                                  data_set)
graph_file = '{}graph-{}.gpickle'.format(output_home_directory,
                                           str(in_atomic_t))
results_dir = '{}er/'.format(output_home_directory)
result_file = '{}results.csv'.format(results_dir)
entity_file = '{}entities-{}.gpickle'.format(results_dir, hyperparams.scenario)
stats_dir = '{}stats/'.format(output_home_directory)
