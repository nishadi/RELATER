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
""" Module with class ATTRIBUTES that holds the different types of
    attributes and their properties.
"""
import logging

from common import constants as c

if c.data_set == 'ios':
  people_sim_func = {
    c.I_FNAME: 'JW',
    c.I_SNAME: 'JW',
    c.I_ADDR1: 'JW',
    c.I_PARISH: 'JW',
    # c.I_ADDR2: 'JW',
    # c.I_OCC: 'JW',
    c.I_MARRIAGE_YEAR: 'MAD',
    # c.I_MARRIAGE_PLACE1: 'JW',
    c.I_MARRIAGE_PLACE2: 'JW',
    c.I_BIRTH_YEAR: 'MAD',
    c.I_BIRTH_PLACE: 'JW'
  }
elif c.data_set == 'kil':
  people_sim_func = {
    c.I_FNAME: 'JW',
    c.I_SNAME: 'JW',
    # c.I_ADDR1: 'JW',
    # c.I_PARISH: 'JW',
    # c.I_ADDR2: 'JW',
    # c.I_OCC: 'JW',
    c.I_MARRIAGE_YEAR: 'MAD',
    c.I_MARRIAGE_PLACE1: 'JW',
    c.I_MARRIAGE_PLACE2: 'JW',
    c.I_BIRTH_YEAR: 'MAD',
    c.I_BIRTH_PLACE: 'JW'
  }
elif c.data_set.startswith('bhic'):
  people_sim_func = {
    c.I_FNAME: 'JW',
    c.I_SNAME: 'JW',
    c.I_MARRIAGE_YEAR: 'MAD',
    c.I_MARRIAGE_PLACE1: 'JW',
    c.I_MARRIAGE_PLACE2: 'JW',
    c.I_BIRTH_YEAR: 'MAD'
  }

hh_sim_func = {
  c.HH_I_RACE: 'JW',
  c.HH_I_BPL: 'JW',
  c.HH_I_FNAME: 'JW',
  c.HH_I_SNAME: 'JW',
  c.HH_I_BYEAR: 'MAD',
  c.HH_I_STATE: 'JW',
  # c.HH_I_CITY: 'JW',
  c.HH_I_OCCUPATION: 'JW'
}

if c.data_set == 'dblp-acm1':
  authors_sim_func = {c.BB_I_YEAR: 'MAD',
                      # c.BB_I_VENUE: 'JW',
                      c.BB_I_PUBNAME: 'JW',
                      c.BB_I_FNAME: 'JW',
                      c.BB_I_SNAME: 'JW'
                      # c.BB_I_MNAME: 'JW',
                      # c.BB_I_FULLNAME: 'JW'
                      }
  pubs_sim_func = {c.BB_I_YEAR: 'MAD',
                   c.BB_I_VENUE: 'JW',
                   c.BB_I_PUBNAME: 'JW'}
elif c.data_set == 'dblp-scholar1':
  authors_sim_func = {
    c.BB_I_YEAR: 'MAD',
    # c.BB_I_VENUE: 'JW',
    # c.BB_I_PUBNAME: 'JW',
    c.BB_I_FNAME: 'JW',
    c.BB_I_SNAME: 'JW'
    # c.BB_I_MNAME: 'JW',
    # c.BB_I_FULLNAME: 'JW'
  }
  pubs_sim_func = {c.BB_I_PUBNAME: 'JW',
                   c.BB_I_YEAR: 'MAD'
                   # c.BB_I_VENUE: 'JW'
                   }
  # authors_sim_func = {c.BB_I_FNAME: 'JW',
  #                     c.BB_I_SNAME: 'JW'}
  # pubs_sim_func = {c.BB_I_PUBNAME: 'JW',
  #                  c.BB_I_YEAR: 'MAD'}
songs_sim_func = {
  c.SG_I_ALBUM: 'JW',
  c.SG_I_TITLE: 'JW',
  c.SG_I_ARTIST: 'JW',
  c.SG_I_LENGTH: 'MAD',
  c.SG_I_YEAR: 'MAD'
}


class ATTRIBUTE_CATEGORY:

  def __init__(self, attributes_index_list, min_sim=0.0, least_count=0.0):
    """
    :param attributes_sim_func: Dictionary of attributes where key is the
    attribute index and value is the sim_func
    :param min_sim:  Threshold for least number of atomic nodes that should
    be available for a relationship node to exist
    :param least_count: The least number of attributes that should exist for
    the existence of a relationship node
    """
    self.attributes = attributes_index_list
    self.min_sim = min_sim
    self.least_count = least_count


class ATTRIBUTES_META:

  def __init__(self, must_attributes, core_attributes, extra_attributes,
               sim_func_dict):
    self.must_attr = must_attributes
    self.core_attr = core_attributes
    self.extra_attr = extra_attributes

    self.attr_index_list = must_attributes.attributes + \
                           core_attributes.attributes + \
                           extra_attributes.attributes
    self.sim_func_dict = sim_func_dict


# -----------------------------------------------------------------------------
# Initialization methods to initialize attributes with different combinations
# -----------------------------------------------------------------------------

def def1_core3_extra1(atomic_node_similarity_t):
  # Core attributes
  core_attributes_index_list = [c.I_FNAME, c.I_SNAME, c.I_ADDR1, c.I_PARISH]
  core_min_sim = atomic_node_similarity_t
  core_least_count = 3

  # Extra attributes
  extra_attributes_index_list = [c.I_OCC, c.I_MARRIAGE_YEAR,
                                 c.I_MARRIAGE_PLACE1, c.I_MARRIAGE_PLACE2,
                                 c.I_BIRTH_YEAR]
  extra_min_sim = atomic_node_similarity_t - 0.1
  extra_least_count = 1

  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY([])
  core_attributes = ATTRIBUTE_CATEGORY(core_attributes_index_list,
                                       core_min_sim, core_least_count)
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, people_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes


def def2_core3_extra0(atomic_node_similarity_t):
  # Core attributes
  core_attributes_index_list = [c.I_FNAME, c.I_SNAME, c.I_ADDR1, c.I_PARISH]
  core_min_sim = atomic_node_similarity_t
  core_least_count = 3

  # Extra attributes
  extra_attributes_index_list = [c.I_OCC, c.I_MARRIAGE_YEAR,
                                 c.I_MARRIAGE_PLACE1, c.I_MARRIAGE_PLACE2,
                                 c.I_BIRTH_YEAR]
  extra_min_sim = atomic_node_similarity_t - 0.1
  extra_least_count = 0

  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY([])
  core_attributes = ATTRIBUTE_CATEGORY(core_attributes_index_list,
                                       core_min_sim, core_least_count)
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, people_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes


def def3_must1_core2_extra1(atomic_node_similarity_t):
  must_attributes_index_list = [c.I_FNAME]
  must_min_sim = atomic_node_similarity_t
  must_least_count = 1

  core_attributes_index_list = [c.I_SNAME, c.I_ADDR1, c.I_PARISH]
  core_min_sim = atomic_node_similarity_t
  core_least_count = 2

  extra_attributes_index_list = [c.I_OCC, c.I_MARRIAGE_YEAR,
                                 c.I_MARRIAGE_PLACE1, c.I_MARRIAGE_PLACE2,
                                 c.I_BIRTH_YEAR]
  extra_min_sim = atomic_node_similarity_t - 0.1
  extra_least_count = 1
  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY([])
  core_attributes = ATTRIBUTE_CATEGORY(core_attributes_index_list,
                                       core_min_sim, core_least_count)
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, people_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes


def def4_must1_core2_extra0(atomic_node_similarity_t):
  # Must attributes
  must_attributes_index_list = [c.I_FNAME]
  must_min_sim = atomic_node_similarity_t
  must_least_count = 1

  # Core attributes
  core_attributes_index_list = [c.I_SNAME, c.I_ADDR1, c.I_PARISH]
  core_min_sim = atomic_node_similarity_t
  core_least_count = 2

  # Extra attributes
  extra_attributes_index_list = [c.I_OCC, c.I_MARRIAGE_YEAR,
                                 c.I_MARRIAGE_PLACE1, c.I_MARRIAGE_PLACE2,
                                 c.I_BIRTH_YEAR]
  extra_min_sim = atomic_node_similarity_t - 0.1
  extra_least_count = 0

  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY(must_attributes_index_list,
                                       must_min_sim, must_least_count)
  core_attributes = ATTRIBUTE_CATEGORY(core_attributes_index_list,
                                       core_min_sim, core_least_count)
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, people_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes


def def5_must4(atomic_node_similarity_t):
  # Must attributes
  must_attributes_index_list = [c.I_FNAME, c.I_SNAME, c.I_ADDR1, c.I_PARISH]
  must_min_sim = atomic_node_similarity_t
  must_least_count = 4

  # Extra attributes
  extra_attributes_index_list = [c.I_OCC, c.I_MARRIAGE_YEAR,
                                 c.I_MARRIAGE_PLACE1, c.I_MARRIAGE_PLACE2,
                                 c.I_BIRTH_YEAR]
  extra_min_sim = atomic_node_similarity_t - 0.1
  extra_least_count = 0

  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY(must_attributes_index_list,
                                       must_min_sim, must_least_count)
  core_attributes = ATTRIBUTE_CATEGORY([])
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, people_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes


def def6_must3(atomic_node_similarity_t):
  # Must attributes
  must_attributes_index_list = [c.I_FNAME, c.I_ADDR1, c.I_PARISH]
  must_min_sim = atomic_node_similarity_t
  must_least_count = 3

  # Extra attributes
  extra_attributes_index_list = [c.I_SNAME, c.I_OCC, c.I_MARRIAGE_YEAR,
                                 c.I_MARRIAGE_PLACE1, c.I_MARRIAGE_PLACE2,
                                 c.I_BIRTH_YEAR]
  extra_min_sim = atomic_node_similarity_t - 0.1
  extra_least_count = 0

  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY(must_attributes_index_list,
                                       must_min_sim, must_least_count)
  core_attributes = ATTRIBUTE_CATEGORY([])
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, people_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes


def def7_must2(atomic_node_similarity_t):
  # Must attributes
  must_attributes_index_list = [c.I_FNAME, c.I_SNAME]
  must_min_sim = atomic_node_similarity_t
  must_least_count = 2

  # Extra attributes
  extra_attributes_index_list = [c.I_ADDR1, c.I_PARISH, c.I_OCC,
                                 c.I_MARRIAGE_YEAR, c.I_MARRIAGE_PLACE1,
                                 c.I_MARRIAGE_PLACE2, c.I_BIRTH_YEAR]
  extra_min_sim = atomic_node_similarity_t - 0.1
  extra_least_count = 0

  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY(must_attributes_index_list,
                                       must_min_sim, must_least_count)
  core_attributes = ATTRIBUTE_CATEGORY([])
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, people_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes


def def8_must3(atomic_node_similarity_t):
  # Must attributes
  must_attributes_index_list = [c.I_FNAME, c.I_SNAME, c.I_PARISH]
  must_min_sim = atomic_node_similarity_t
  must_least_count = 3

  # Extra attributes
  extra_attributes_index_list = [c.I_ADDR1, c.I_OCC, c.I_MARRIAGE_YEAR,
                                 c.I_MARRIAGE_PLACE1, c.I_MARRIAGE_PLACE2,
                                 c.I_BIRTH_YEAR]
  extra_min_sim = atomic_node_similarity_t - 0.1
  extra_least_count = 0

  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY(must_attributes_index_list,
                                       must_min_sim, must_least_count)
  core_attributes = ATTRIBUTE_CATEGORY([])
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, people_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes


def def9_must3(atomic_node_similarity_t):
  # Must attributes
  must_attributes_index_list = [c.I_FNAME, c.I_SNAME, c.I_ADDR1]
  must_min_sim = atomic_node_similarity_t
  must_least_count = 3

  # Extra attributes
  extra_attributes_index_list = [c.I_PARISH, c.I_OCC, c.I_MARRIAGE_YEAR,
                                 c.I_MARRIAGE_PLACE1, c.I_MARRIAGE_PLACE2,
                                 c.I_BIRTH_YEAR]
  extra_min_sim = atomic_node_similarity_t - 0.1
  extra_least_count = 0

  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY(must_attributes_index_list,
                                       must_min_sim, must_least_count)
  core_attributes = ATTRIBUTE_CATEGORY([])
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, people_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes


def def10_must2_extra2(atomic_node_similarity_t):
  # Must attributes
  must_attributes_index_list = [c.I_FNAME, c.I_SNAME]
  must_min_sim = atomic_node_similarity_t
  must_least_count = 2

  # Extra attributes
  extra_attributes_index_list = [c.I_PARISH, c.I_ADDR1, c.I_OCC,
                                 c.I_MARRIAGE_YEAR,
                                 c.I_MARRIAGE_PLACE1, c.I_MARRIAGE_PLACE2,
                                 c.I_BIRTH_YEAR]
  extra_min_sim = atomic_node_similarity_t - 0.1
  extra_least_count = 2

  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY(must_attributes_index_list,
                                       must_min_sim, must_least_count)
  core_attributes = ATTRIBUTE_CATEGORY([])
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, people_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes


def def11_must2_extra2(atomic_node_similarity_t):
  # Must attributes
  must_attributes_index_list = [c.I_FNAME, c.I_ADDR1]
  must_min_sim = atomic_node_similarity_t
  must_least_count = 2

  # Extra attributes
  extra_attributes_index_list = [c.I_PARISH, c.I_SNAME, c.I_OCC,
                                 c.I_MARRIAGE_YEAR, c.I_MARRIAGE_PLACE1,
                                 c.I_MARRIAGE_PLACE2, c.I_BIRTH_YEAR]
  extra_min_sim = atomic_node_similarity_t - 0.1
  extra_least_count = 2

  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY(must_attributes_index_list,
                                       must_min_sim, must_least_count)
  core_attributes = ATTRIBUTE_CATEGORY([])
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, people_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes


def def12_must2_extra2(atomic_node_similarity_t):
  # Must attributes
  must_attributes_index_list = [c.I_FNAME, c.I_PARISH]
  must_min_sim = atomic_node_similarity_t
  must_least_count = 2

  # Extra attributes
  extra_attributes_index_list = [c.I_SNAME, c.I_ADDR1, c.I_OCC,
                                 c.I_MARRIAGE_YEAR, c.I_MARRIAGE_PLACE1,
                                 c.I_MARRIAGE_PLACE2, c.I_BIRTH_YEAR]
  extra_min_sim = atomic_node_similarity_t - 0.1
  extra_least_count = 2

  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY(must_attributes_index_list,
                                       must_min_sim, must_least_count)
  core_attributes = ATTRIBUTE_CATEGORY([])
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, people_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes


def def13_must2_extra2(atomic_node_similarity_t):
  # Must attributes
  must_attributes_index_list = [c.I_SNAME, c.I_ADDR1]
  must_min_sim = atomic_node_similarity_t
  must_least_count = 2

  # Extra attributes
  extra_attributes_index_list = [c.I_FNAME, c.I_PARISH, c.I_OCC,
                                 c.I_MARRIAGE_YEAR, c.I_MARRIAGE_PLACE1,
                                 c.I_MARRIAGE_PLACE2, c.I_BIRTH_YEAR]
  extra_min_sim = atomic_node_similarity_t - 0.1
  extra_least_count = 2

  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY(must_attributes_index_list,
                                       must_min_sim, must_least_count)
  core_attributes = ATTRIBUTE_CATEGORY([])
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, people_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes


def def14_core2_extra0(atomic_node_similarity_t):
  # Must attributes
  must_attributes_index_list = [c.I_FNAME, c.I_SNAME]
  must_min_sim = atomic_node_similarity_t
  must_least_count = 0

  core_attributes_index_list = [c.I_ADDR1, c.I_PARISH]
  core_min_sim = atomic_node_similarity_t
  core_least_count = 0

  # Extra attributes
  extra_attributes_index_list = [c.I_OCC, c.I_MARRIAGE_YEAR,
                                 c.I_MARRIAGE_PLACE1, c.I_MARRIAGE_PLACE2,
                                 c.I_BIRTH_YEAR, c.I_BIRTH_PLACE]
  extra_min_sim = atomic_node_similarity_t - 0.1
  extra_least_count = 0

  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY(must_attributes_index_list,
                                       must_min_sim, must_least_count)
  core_attributes = ATTRIBUTE_CATEGORY(core_attributes_index_list,
                                       core_min_sim, core_least_count)
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, people_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes


def def_ios(atomic_node_similarity_t):
  # Must attributes
  must_attributes_index_list = [c.I_FNAME, c.I_SNAME]
  must_min_sim = atomic_node_similarity_t
  must_least_count = 0

  core_attributes_index_list = [c.I_ADDR1, c.I_MARRIAGE_YEAR]
  core_min_sim = atomic_node_similarity_t
  core_least_count = 0

  # Extra attributes
  extra_attributes_index_list = [c.I_MARRIAGE_PLACE2, c.I_PARISH,
                                 c.I_BIRTH_YEAR, c.I_BIRTH_PLACE]
  # extra_attributes_index_list = [c.I_OCC,
  #                                c.I_MARRIAGE_PLACE1, c.I_MARRIAGE_PLACE2,
  #                                c.I_BIRTH_YEAR, c.I_BIRTH_PLACE, c.I_PARISH]
  extra_min_sim = atomic_node_similarity_t - 0.1
  extra_least_count = 0

  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY(must_attributes_index_list,
                                       must_min_sim, must_least_count)
  core_attributes = ATTRIBUTE_CATEGORY(core_attributes_index_list,
                                       core_min_sim, core_least_count)
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, people_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes


def def_kil(atomic_node_similarity_t):
  # Must attributes
  # must_attributes_index_list = [c.I_FNAME, c.I_SNAME]
  # must_min_sim = atomic_node_similarity_t
  # must_least_count = 0
  #
  # core_attributes_index_list = []
  # core_min_sim = atomic_node_similarity_t
  # core_least_count = 0
  #
  # # Extra attributes
  # extra_attributes_index_list = [c.I_MARRIAGE_YEAR, c.I_MARRIAGE_PLACE1,
  #                                c.I_MARRIAGE_PLACE2,
  #                                c.I_BIRTH_YEAR, c.I_BIRTH_PLACE]
  # extra_min_sim = atomic_node_similarity_t - 0.1
  # extra_least_count = 0

  must_attributes_index_list = [c.I_SNAME]
  must_min_sim = atomic_node_similarity_t
  must_least_count = 0

  core_attributes_index_list = [c.I_FNAME]
  core_min_sim = atomic_node_similarity_t
  core_least_count = 0

  # Extra attributes
  extra_attributes_index_list = [c.I_MARRIAGE_YEAR,
                                 c.I_MARRIAGE_PLACE1, c.I_MARRIAGE_PLACE2,
                                 c.I_BIRTH_YEAR, c.I_BIRTH_PLACE]
  extra_min_sim = atomic_node_similarity_t - 0.1
  extra_least_count = 0

  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY(must_attributes_index_list,
                                       must_min_sim, must_least_count)
  core_attributes = ATTRIBUTE_CATEGORY(core_attributes_index_list,
                                       core_min_sim, core_least_count)
  # core_attributes = ATTRIBUTE_CATEGORY([])
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, people_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes


def def15_bhic(atomic_node_similarity_t):
  # Must attributes
  core_attributes_index_list = [c.I_FNAME, c.I_SNAME]
  core_min_sim = atomic_node_similarity_t
  core_least_count = 2

  # Extra attributes
  extra_attributes_index_list = [c.I_BIRTH_YEAR, c.I_MARRIAGE_YEAR,
                                 c.I_MARRIAGE_PLACE1, c.I_BIRTH_PLACE]
  extra_min_sim = atomic_node_similarity_t
  extra_least_count = 0

  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY([])
  core_attributes = ATTRIBUTE_CATEGORY(core_attributes_index_list,
                                       core_min_sim, core_least_count)
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, people_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes


def def_authors(atomic_node_similarity_t):
  must_attributes_index_list = []
  must_min_sim = atomic_node_similarity_t
  must_least_count = 0

  # Core attributes
  # core_attributes_index_list = [c.BB_I_FULLNAME, c.BB_I_FNAME, c.BB_I_SNAME]
  core_attributes_index_list = [c.BB_I_SNAME, c.BB_I_FNAME, c.BB_I_PUBNAME]
  core_min_sim = atomic_node_similarity_t
  core_least_count = 1

  # Extra attributes
  extra_attributes_index_list = [c.BB_I_YEAR]
  # extra_attributes_index_list = [c.BB_I_MNAME, c.BB_I_YEAR, c.BB_I_VENUE,
  #                                c.BB_I_FULLNAME, c.BB_I_PUBNAME]
  extra_min_sim = atomic_node_similarity_t - 0.1
  extra_least_count = 0

  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY(must_attributes_index_list,
                                       must_min_sim, must_least_count)
  core_attributes = ATTRIBUTE_CATEGORY(core_attributes_index_list,
                                       core_min_sim, core_least_count)
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, authors_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes


def def_pubs(atomic_node_similarity_t):
  # Core attributes
  must_attributes_index_list = [c.BB_I_PUBNAME]
  must_min_sim = atomic_node_similarity_t
  must_least_count = 1

  # Core attributes
  core_attributes_index_list = [c.BB_I_YEAR]
  core_min_sim = atomic_node_similarity_t - 0.1
  core_least_count = 0

  # Extra attributes
  extra_attributes_index_list = [c.BB_I_VENUE]
  # extra_attributes_index_list = []
  extra_min_sim = atomic_node_similarity_t - 0.1
  extra_least_count = 0

  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY(must_attributes_index_list,
                                       must_min_sim, must_least_count)
  core_attributes = ATTRIBUTE_CATEGORY(core_attributes_index_list,
                                       core_min_sim, core_least_count)
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, pubs_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes


def def_authors_scholar(atomic_node_similarity_t):
  must_attributes_index_list = [c.BB_I_SNAME]
  must_min_sim = atomic_node_similarity_t
  must_least_count = 0

  # Core attributes
  # core_attributes_index_list = [c.BB_I_FULLNAME, c.BB_I_FNAME, c.BB_I_SNAME]
  core_attributes_index_list = [c.BB_I_FNAME]
  core_min_sim = atomic_node_similarity_t
  core_least_count = 1

  # Extra attributes
  extra_attributes_index_list = [c.BB_I_YEAR]
  # extra_attributes_index_list = [c.BB_I_MNAME, c.BB_I_YEAR, c.BB_I_VENUE,
  #                                c.BB_I_FULLNAME, c.BB_I_PUBNAME]
  extra_min_sim = atomic_node_similarity_t - 0.1
  extra_least_count = 0

  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY(must_attributes_index_list,
                                       must_min_sim, must_least_count)
  core_attributes = ATTRIBUTE_CATEGORY(core_attributes_index_list,
                                       core_min_sim, core_least_count)
  # extra_attributes = ATTRIBUTE_CATEGORY([])
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, authors_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes


def def_pubs_scholar(atomic_node_similarity_t):
  # Core attributes
  must_attributes_index_list = [c.BB_I_PUBNAME]
  must_min_sim = atomic_node_similarity_t
  must_least_count = 1

  # Core attributes
  core_attributes_index_list = []
  core_min_sim = atomic_node_similarity_t - 0.1
  core_least_count = 0

  # Extra attributes
  extra_attributes_index_list = [c.BB_I_YEAR]
  # extra_attributes_index_list = []
  extra_min_sim = atomic_node_similarity_t - 0.1
  extra_least_count = 0

  # # Core attributes
  # must_attributes_index_list = [c.BB_I_PUBNAME]
  # must_min_sim = atomic_node_similarity_t
  # must_least_count = 1
  #
  # # # Core attributes
  # core_attributes_index_list = []
  # core_min_sim = atomic_node_similarity_t - 0.1
  # core_least_count = 0
  #
  # # Extra attributes
  # # extra_attributes_index_list = [c.BB_I_VENUE]
  # extra_attributes_index_list = [c.BB_I_YEAR]
  # extra_min_sim = atomic_node_similarity_t - 0.1
  # extra_least_count = 0

  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY(must_attributes_index_list,
                                       must_min_sim, must_least_count)
  core_attributes = ATTRIBUTE_CATEGORY(core_attributes_index_list,
                                       core_min_sim, core_least_count)
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, pubs_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes


def def_hh(atomic_node_similarity_t):
  # Must attributes
  must_attributes_index_list = [c.HH_I_FNAME, c.HH_I_RACE, c.HH_I_BPL]
  must_min_sim = atomic_node_similarity_t
  must_least_count = 2

  # Core attributes
  core_attributes_index_list = [c.HH_I_SNAME, c.HH_I_BYEAR]
  core_min_sim = atomic_node_similarity_t
  core_least_count = 0

  # Extra attributes
  extra_attributes_index_list = [c.HH_I_STATE, c.HH_I_CITY, c.HH_I_OCCUPATION]
  extra_min_sim = atomic_node_similarity_t - 0.1
  extra_least_count = 0

  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY(must_attributes_index_list,
                                       must_min_sim, must_least_count)
  core_attributes = ATTRIBUTE_CATEGORY(core_attributes_index_list,
                                       core_min_sim, core_least_count)
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, hh_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes


def def_hh_link(atomic_node_similarity_t):
  # Must attributes
  must_attributes_index_list = [c.HH_I_SNAME, c.HH_I_BPL, c.HH_I_FNAME,
                                c.HH_I_RACE]
  must_min_sim = atomic_node_similarity_t
  must_least_count = 2

  # Core attributes
  core_attributes_index_list = [c.HH_I_BYEAR]
  core_min_sim = atomic_node_similarity_t
  core_least_count = 0

  # Extra attributes
  # extra_attributes_index_list = [c.HH_I_STATE, c.HH_I_CITY, c.HH_I_OCCUPATION]
  extra_attributes_index_list = [c.HH_I_STATE, c.HH_I_OCCUPATION]
  extra_min_sim = atomic_node_similarity_t - 0.1
  extra_least_count = 0

  # must_attributes_index_list = [c.HH_I_FNAME, c.HH_I_BPL, c.HH_I_RACE,
  #                               c.HH_I_SNAME]
  # must_min_sim = atomic_node_similarity_t
  # must_least_count = 2
  #
  # # Core attributes
  # core_attributes_index_list = [c.HH_I_BYEAR]
  # core_min_sim = atomic_node_similarity_t
  # core_least_count = 1
  #
  # # Extra attributes
  # extra_attributes_index_list = [c.HH_I_STATE, c.HH_I_CITY,
  #                                c.HH_I_OCCUPATION]
  # extra_min_sim = atomic_node_similarity_t - 0.1
  # extra_least_count = 0

  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY(must_attributes_index_list,
                                       must_min_sim, must_least_count)
  core_attributes = ATTRIBUTE_CATEGORY(core_attributes_index_list,
                                       core_min_sim, core_least_count)
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, hh_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes


def def_sg(atomic_node_similarity_t):
  # Must attributes
  must_attributes_index_list = [c.SG_I_TITLE]
  must_min_sim = atomic_node_similarity_t
  must_least_count = 1

  # Core attributes
  core_attributes_index_list = [c.SG_I_ARTIST, c.SG_I_YEAR]
  core_min_sim = atomic_node_similarity_t
  core_least_count = 0

  # Extra attributes
  extra_attributes_index_list = [c.SG_I_LENGTH, c.SG_I_ALBUM]
  extra_min_sim = atomic_node_similarity_t - 0.1
  extra_least_count = 0

  # Attributes
  must_attributes = ATTRIBUTE_CATEGORY(must_attributes_index_list,
                                       must_min_sim, must_least_count)
  core_attributes = ATTRIBUTE_CATEGORY(core_attributes_index_list,
                                       core_min_sim, core_least_count)
  extra_attributes = ATTRIBUTE_CATEGORY(extra_attributes_index_list,
                                        extra_min_sim, extra_least_count)
  attributes = ATTRIBUTES_META(must_attributes, core_attributes,
                               extra_attributes, hh_sim_func)

  logging.info('Must attributes {}'.format(attributes.must_attr.attributes))
  logging.info('Core attributes {}'.format(attributes.core_attr.attributes))
  logging.info('Extra attributes {}'.format(attributes.extra_attr.attributes))

  return attributes
