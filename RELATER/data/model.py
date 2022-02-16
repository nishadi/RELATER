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
import string
import networkx as nx
import re

from common import constants as c


def get_standard_numerical_value(s):
  if type(s) == str:
    if s is '' or s is None:
      return None
    if s.isdigit():
      return int(s)
    if ':' in s and len(s.split(':')) == 2:
      # Time values in the format of 'x:y'
      str_min, str_sec = s.split(':')
      num_min = int(str_min) if str_min.isdigit() else 0
      num_sec = int(str_sec) if str_sec.isdigit() else 0
      return num_min * 60 + num_sec
    try:
      return int(float(s))
    except ValueError:
      return 0
  return s


def get_standard_string(str):
  if str is None or str == 'n/e' or str == 'unknown' or str == 'no entry' or \
      str == 'ng' or str == 'nk' or str == 'na':  # Added for Kilmarnock data set
    return None
  str = str.lower()
  for punct_char in string.punctuation:
    str = str.replace(punct_char, ' ')
  str = re.sub('\s+', ' ', str)
  str = str.strip()
  if str == '' or str == 'anonymous' or str == 'n e':
    return None
  return str


def new_hh_person(id, gender, role, race, bpl, fname, sname, byear, state,
                  city, occ, year, serial, pernum):
  race = get_standard_string(race)
  bpl = get_standard_string(bpl)
  fname = get_standard_string(fname)
  sname = get_standard_string(sname)
  state = get_standard_string(state)
  city = get_standard_string(city)
  occ = get_standard_string(occ)

  if year is not None:
    year = int(year)
  if byear is not None:
    byear = int(byear)

  # I_ID = 0
  # I_ENTITY_ID = 1
  # I_FREQ = 2
  # I_SEX = 3
  # I_ROLE = 4
  # HH_I_RACE = 5
  # HH_I_BPL = 6
  # HH_I_FNAME = 7
  # HH_I_SNAME = 8
  # HH_I_BYEAR = 9
  # HH_I_STATE = 10
  # HH_I_CITY = 11
  # HH_I_OCCUPATION = 12
  # HH_I_YEAR = 13
  # HH_I_SERIAL = 14
  # HH_I_PERNUM = 15
  # HH_I_MOTHER = 16
  # HH_I_FATHER = 17
  # HH_I_CHILDREN = 18
  # HH_I_SPOUSE = 19
  # HH_I_SIBLINGS = 20

  return [int(id), None, None, gender, role, race, bpl, fname, sname, byear,
          state, city, occ, year, int(serial), int(pernum), None, None, None,
          None, None]


def new_author(id, origin_id, year, venue, pubname, fname, sname, mname,
               fullname):
  fname = get_standard_string(fname)
  sname = get_standard_string(sname)
  mname = get_standard_string(mname)
  fullname = get_standard_string(fullname)
  pubname = get_standard_string(pubname)
  venue = get_standard_string(venue)
  year = get_standard_numerical_value(year)

  # Attribute indexes
  # I_ID = 0
  # I_ENTITY_ID = 1
  # I_FREQ = 2
  # I_SEX = 3
  # I_ROLE = 4
  # BB_I_ORIGIN_ID = 5
  # BB_I_YEAR = 6
  # BB_I_VENUE = 7
  # BB_I_PUBNAME = 8
  # BB_I_FNAME = 9
  # BB_I_SNAME = 10
  # BB_I_MNAME = 11
  # BB_I_FULLNAME = 12
  # BB_I_COAUTHOR_LIST = 13
  # BB_I_PUBID = 14

  return [id, None, None, None, 'A', origin_id, year, venue, pubname, fname,
          sname, mname, fullname, None, None]


def new_publication(id, origin_id, year, venue, pubname, author_id_list):
  pubname = get_standard_string(pubname)
  venue = get_standard_string(venue)
  year = get_standard_numerical_value(year)
  # Attribute indexes
  # I_ID = 0
  # I_ENTITY_ID = 1
  # I_FREQ = 2
  # I_SEX = 3
  # I_ROLE = 4
  # BB_I_ORIGIN_ID = 5
  # BB_I_YEAR = 6
  # BB_I_VENUE = 7
  # BB_I_PUBNAME = 8
  # BB_I_AUTHOR_ID_LIST = 9

  return [id, None, None, None, 'P', origin_id, year, venue, pubname,
          author_id_list]


def new_song(id, title, album, artist, year, length):
  title = get_standard_string(title)
  album = get_standard_string(album)
  artist = get_standard_string(artist)
  year = get_standard_numerical_value(year)
  length = get_standard_numerical_value(length)
  # Attribute indexes
  # I_ID = 0
  # I_ENTITY_ID = 1
  # I_FREQ = 2
  # I_SEX = 3
  # I_ROLE = 4
  # SG_I_TITLE = 5
  # SG_I_ALBUM = 6
  # SG_I_ARTIST = 7
  # SG_I_YEAR = 8
  # SG_I_LENGTH = 9

  return [id, None, None, None, 'S', title, album, artist, year, length]


def new_record(pid, fname, sname, sex, occ, addr1, addr2, p_type, timestamp,
               event_date, certificate_id, parish, birth_year=None,
               marriage_year=None, marriage_place1=None,
               marriage_place2=None, marrital_status=None, birth_place=None):
  fname = get_standard_string(fname)
  sname = get_standard_string(sname)
  occ = get_standard_string(occ)
  addr1 = get_standard_string(addr1)
  addr2 = get_standard_string(addr2)
  parish = get_standard_string(parish)
  birth_place = get_standard_string(birth_place)
  marriage_place1 = get_standard_string(marriage_place1)
  marriage_place2 = get_standard_string(marriage_place2)
  birth_year = get_standard_numerical_value(birth_year)
  marriage_year = get_standard_numerical_value(marriage_year)

  # Attribute indexes
  # I_ID = 0
  # I_ENTITY_ID = 1
  # I_FREQ = 2
  # I_SEX = 3
  # I_ROLE = 4
  #
  # I_TIMESTAMP = 5
  # I_EDATE = 6
  # I_CERT_ID = 7
  # I_FNAME = 8
  # I_SNAME = 9
  # I_ADDR1 = 10
  # I_PARISH = 11
  # I_OCC = 12
  # I_ADDR2 = 13
  # I_BIRTH_YEAR = 14
  # I_BIRTH_PLACE = 15
  # I_MARRIAGE_YEAR = 16
  # I_MARRIAGE_PLACE1 = 17
  # I_MARRIAGE_PLACE2 = 18
  # I_MARRITAL_STATUS = 19
  #
  # I_MOTHER = 20
  # I_FATHER = 21
  # I_CHILDREN = 22
  # I_SPOUSE = 23

  record = [pid, None, None, sex, p_type, timestamp, event_date, certificate_id,
            fname, sname, addr1, parish, occ, addr2, birth_year,
            birth_place, marriage_year, marriage_place1, marriage_place2,
            marrital_status, None, None, None, None]

  return record


def new_entity(entity_id, sex):
  # Atomic node attribute lists are to be added later based on the sim_att
  # list provided
  merged_person = {
    c.ENTITY_ID: entity_id,
    c.SEX: sex,
    c.ROLES: None,

    c.MOTHER: None,
    c.FATHER: None,
    c.CHILDREN: None,
    c.SPOUSE: None,

    c.AUTHOR: None,

    c.DEATH_YEAR: None,
    c.EXACT_BIRTH_YEAR: None,  # Birth year directly coming from Bb record
    c.MIN_EY: None,
    c.MAX_EY: None,

    c.TRACE: list(),
    c.GRAPH: nx.Graph()
  }

  return merged_person


if __name__ == '__main__':
  print get_standard_string('Nishadi')
  print get_standard_string('Nish\' adi')
  print get_standard_string('  Ni   s  h adi')
  print get_standard_string('\'\'')
  print get_standard_numerical_value('03:09')
