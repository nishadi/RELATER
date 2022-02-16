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
data_set = ''

# Node attributes
R1 = 'r1'
R2 = 'r2'
TYPE1 = 't1'
TYPE2 = 't2'
SIM = 's'
SIM_ATOMIC = 'sa'
SIM_ATTR = 'sat'
SIM_AMB = 'sam'
SIM_REL = 'sr'
STATE = 'st'
AMB = 'a'

# Node values
N_RELTV = 'R'  # Relative_node
N_ATOMIC = 'A'  # Atomic node
E_REAL = 'R'  # Real valued edges
E_BOOL_S = 'BS'  # Boolean strong valued edges
E_BOOL_W = 'BW'  # Boolean weak valued edges

## Record indexes

# Common
I_ID = 0
I_ENTITY_ID = 1
I_FREQ = 2
I_SEX = 3
I_ROLE = 4

# Record indexes for civil registries
I_TIMESTAMP = 5
I_EDATE = 6
I_CERT_ID = 7
I_FNAME = 8
I_SNAME = 9
I_ADDR1 = 10
I_PARISH = 11
I_OCC = 12
I_ADDR2 = 13
I_BIRTH_YEAR = 14
I_BIRTH_PLACE = 15
I_MARRIAGE_YEAR = 16
I_MARRIAGE_PLACE1 = 17
I_MARRIAGE_PLACE2 = 18
I_MARRITAL_STATUS = 19

I_MOTHER = 20
I_FATHER = 21
I_CHILDREN = 22
I_SPOUSE = 23

CIVIL_I_NAME_DICT = {
  8: 'fname',
  9: 'sname',
  10: 'addr1',
  11: 'parish',
  12: 'occ',
  14: 'byear',
  15: 'bplace',
  16: 'myear',
  17: 'mplace1',
  18: 'mplace2'
}

# # Frequencies of attribute combination for the use of ambiguity calculation
# I_ADDR1_F = 22  # Frequency of fname-sname-address1
# I_PARISH_F = 23  # Frequency of fname-sname-parish
#
# I_DATE = 24

# Record indexes for HH

HH_I_RACE = 5
HH_I_BPL = 6
HH_I_FNAME = 7
HH_I_SNAME = 8
HH_I_BYEAR = 9
HH_I_STATE = 10
HH_I_CITY = 11
HH_I_OCCUPATION = 12
HH_I_YEAR = 13
HH_I_SERIAL = 14
HH_I_PERNUM = 15

HH_I_MOTHER = 16
HH_I_FATHER = 17
HH_I_CHILDREN = 18
HH_I_SPOUSE = 19
HH_I_SIBLINGS = 20

HH_I_NAME_DICT = {
  5: 'race',
  6: 'bplace',
  7: 'fname',
  8: 'sname',
  9: 'byear',
  10: 'state',
  11: 'city',
  12: 'occ'
}

# Record indexes for BIB

# Common for authors and publications
BB_I_ORIGIN_ID = 5
BB_I_YEAR = 6
BB_I_VENUE = 7
BB_I_PUBNAME = 8

# Author specific
BB_I_FNAME = 9
BB_I_SNAME = 10
BB_I_MNAME = 11
BB_I_FULLNAME = 12
BB_I_COAUTHOR_LIST = 13
BB_I_PUBID = 14

# Publication specific
BB_I_AUTHOR_ID_LIST = 9

BB_I_NAME_DICT = {
  6: 'year',
  7: 'venue',
  9: 'fname',
  10: 'sname',
  11: 'mname',
  8: 'pubname',
  12 : 'authorname'
}

# Record indexes for Songs
SG_I_TITLE = 5
SG_I_ALBUM = 6
SG_I_ARTIST = 7
SG_I_YEAR = 8
SG_I_LENGTH = 9

SG_I_NAME_DICT = {
  5: 'title',
  6: 'album',
  7: 'artist',
  8: 'year',
  9: 'length'
}

# Merged person
ENTITY_ID = 'entity_id'
ROLES = 'roles'
SEX = 'sex'
MOTHER = 'm'
FATHER = 'f'
CHILDREN = 'c'
SPOUSE = 's'
SIBLING = 'sb'
AUTHOR = 'au'

DEATH_YEAR = 'dy'
EXACT_BIRTH_YEAR = 'eby'
MIN_EY = 'min_ey'
MAX_EY = 'max_ey'

TRACE = 't'  # type: str
GRAPH = 'g'

# Causes for merging or invalidation
LINKAGE_REASONS_LIST = ['NEW', 'NG', 'LSIM', 'SIN', 'S_B', 'RER-SELF-NE',
                        'RER-SELF-E', 'RER-F-E', 'RER-M-E', 'RER-M-NE', 'RP_S',
                        'RP_G', 'RE2_BY', 'RE2_MY', 'RE2_BMY', 'RE2_SDd',
                        'RE2_SBb',
                        'RE1_BMY', 'RE1_MY', 'RE1_BMY', 'RE1_BY', 'RE1_SBb',
                        'RE1_SDd']
RP_S = 'P - singleton Bb/Dd'
RP_G = 'P - gender'
RP_BY = 'P - Birth years'

RE2_G = 'E2 - gender'
RE2_SBb = 'E2 - singleton Bb'
RE2_SDd = 'E2 - singleton Dd'
RE2_BY = 'E2 - birth years'
RE2_DY = 'E2 - death years'
RE2_MY = 'E2 - marriage years'
RE2_BMY = 'E2 - birth and marriage years'

RE1_G = 'E1 - gender'
RE1_SBb = 'E1 - singleton Bb'
RE1_SDd = 'E1 - singleton Dd'
RE1_BY = 'E1 - birth years'
RE1_DY = 'E1 - death years'
RE1_MY = 'E1 - marriage years'
RE1_BMY = 'E1 - birth and marriage years'

RM_E = 'New entity'
RM_M = 'Already merged'
RM_EE = 'Entity - entity merge'
RM_EP = 'Entity - person merge'

# BIB Column Names
BB_ID = 'id'
BB_TITLE = 'title'
BB_AUTHORS = 'authors'
BB_VENUE = 'venue'
BB_YEAR = 'year'

# HH Column Names
HH_ID = 'id'
HH_YEAR = 'census_year'
HH_SERIAL = 'hh_serial'
HH_PER_NUM = 'pernum'
HH_RACE = 'race'
HH_BPL = 'bpl'
HH_FNAME = 'fname'
HH_SNAME = 'sname'
HH_BYEAR = 'birth_year'
HH_STATE = 'state'
HH_CITY = 'city'
HH_OCC = 'occ'
HH_GENDER = 'gender'
HH_ROLE = 'role'

# Songs Column Names

SG_ID = 'id'
SG_TITLE = 'title'
SG_ALBUM = 'album'
SG_ARTIST = 'artist'
SG_YEAR = 'year'
SG_LENGTH = 'length'

# BIRTH Column Names
B_ID = ''

B_N = ''
B_S = ''
B_DATE = ''
B_A1 = ''
B_A2 = ''
B_SEX = ''
B_FN = ''
B_FS = ''
B_FO = ''
B_MN = ''
B_MS = ''
B_MO = ''
B_PM_D = ''
B_PM_M = ''
B_PM_Y = ''
B_PM_A1 = ''
B_PM_A2 = ''
B_PLACE = ''

# MARRIAGE Column Names
M_ID = ''
M_DATE = ''
M_PLACE1 = ''
M_PLACE2 = ''
M_D = ''
M_M = ''
M_Y = ''
M_GN = ''
M_GS = ''
M_GO = ''
M_G_MARRITAL = ''
M_G_AGE = ''
M_G_A1 = ''
M_G_A2 = ''
M_G_BP = ''
M_BN = ''
M_BS = ''
M_BO = ''
M_B_MARRITAL = ''
M_B_A1 = ''
M_B_A2 = ''
M_B_BP = ''
M_G_FN = ''
M_G_FS = ''
M_G_FO = ''
M_G_MN = ''
M_G_MS = ''
M_B_FN = ''
M_B_FS = ''
M_B_FO = ''
M_B_MN = ''
M_B_MS = ''
M_B_A = ''
M_G_A = ''
M_B_BY = ''
M_G_BY = ''

# COMMON
PARISH = ''
M_PARISH = ''

# DEATH Column Names
D_ID = ''
D_N = ''
D_S = ''
D_O = ''
D_MARRITAL = ''
D_SEX = ''
D_A = ''
D_SN = ''
D_SS = ''
D_SO = ''
D_DATE = ''
D_D = ''
D_M = ''
D_Y = ''
D_A1 = ''
D_A2 = ''
D_FN = ''
D_FS = ''
D_FO = ''
D_MN = ''
D_MS = ''
D_MO = ''


def __init_constants__(ds):
  # Setting the data set
  global data_set
  data_set = ds

  global B_ID, B_N, B_S, B_DATE, B_A1, B_A2, B_SEX, B_FN, B_FS, B_FO, B_MN, \
    B_MS, B_MO, B_PM_D, B_PM_M, B_PM_Y, B_PM_A1, B_PM_A2, M_ID, M_DATE, \
    M_PLACE1, M_PLACE2, M_D, M_M, M_Y, M_GN, M_GS, M_GO, M_G_MARRITAL, \
    M_G_AGE, M_G_A1, M_G_A2, M_BN, M_BS, M_BO, M_B_MARRITAL, M_B_A1, M_B_A2, \
    M_G_FN, M_G_FS, M_G_FO, M_G_MN, M_G_MS, M_B_FN, M_B_FS, M_B_FO, M_B_MN, \
    M_B_MS, M_B_A, M_G_A, PARISH, M_PARISH, D_ID, D_N, D_S, D_O, D_MARRITAL, \
    D_SEX, D_A, D_SN, D_SS, D_SO, D_DATE, D_D, D_M, D_Y, D_A1, D_A2, D_FN, \
    D_FS, D_FO, D_MN, D_MS, D_MO, M_B_BY, M_G_BY, B_PLACE, M_B_BP, M_G_BP

  if data_set == 'ios':

    # BIRTH RECORD DETAILS
    B_ID = 'IOSBIRTH_Identifier'

    B_N = 'child\'s forname(s)'
    B_S = 'child\'s surname'
    B_DATE = 'birth date'
    B_A1 = 'address 1'
    B_A2 = 'address 2'
    B_SEX = 'sex'
    B_FN = 'father\'s forename'
    B_FS = 'father\'s surname'
    B_FO = 'father\'s occupation'
    B_MN = 'mother\'s forename'
    B_MS = 'mother\'s maiden surname'
    B_MO = 'mother\'s occupation'
    B_PM_D = 'day of parents\' marriage'
    B_PM_M = 'month of parents\' marriage'
    B_PM_Y = 'year of parents\' marriage'
    B_PM_A1 = 'place of parent\'s marriage 1'
    B_PM_A2 = 'place of parent\'s marriage 2'

    # MARRIAGE RECORD DETAILS
    M_ID = 'IOS_identifier'
    M_DATE = 'date of marriage'
    M_PLACE1 = 'place of marriage 1'
    M_PLACE2 = 'place of marriage 2'
    M_D = 'day'
    M_M = 'month'
    M_Y = 'year'
    M_GN = 'forename of groom'
    M_GS = 'surname of groom'
    M_GO = 'occupation of groom'
    M_G_MARRITAL = 'marital status of groom'
    M_G_AGE = 'age of groom'
    M_G_A1 = 'address of groom 1'
    M_G_A2 = 'address of groom 2'
    M_BN = 'forename of bride'
    M_BS = 'surname of bride'
    M_BO = 'occupation of bride'
    M_B_MARRITAL = 'marital status of bride'
    M_B_A1 = 'address of bride 1'
    M_B_A2 = 'address of bride 2'
    M_G_FN = 'groom\'s father\'s forename'
    M_G_FS = 'groom\'s father\'s surname'
    M_G_FO = 'groom\'s father\'s occupation'
    M_G_MN = 'groom\'s mother\'s forename'
    M_G_MS = 'groom\'s mother\'s maiden surname'
    M_B_FN = 'bride\'s father\'s forename'
    M_B_FS = 'bride\'s father\'s surname'
    M_B_FO = 'bride\'s father\'s occupation'
    M_B_MN = 'bride\'s mother\'s forename'
    M_B_MS = 'bride\'s mother\'s maiden surname'
    M_B_A = 'brideagey'
    M_G_A = 'groomagey'

    # COMMON
    PARISH = 'source'
    M_PARISH = 'source'

    # DEATH RECORD DETAILS
    D_ID = 'IOSidentifier'
    D_N = 'forename(s) of deceased'
    D_S = 'surname of deceased'
    D_O = 'occupation'
    D_MARRITAL = 'marital status'  # todo use it
    D_SEX = 'sex'
    D_A = 'agey'
    D_SN = 'forename of spouse'
    D_SS = 'surname of spouse'
    D_SO = 'spouse\'s occ'
    D_DATE = 'death date'
    D_D = 'day'
    D_M = 'month'
    D_Y = 'year'
    D_A1 = 'address 1'
    D_A2 = 'address 2'
    D_FN = 'father\'s forename'
    D_FS = 'father\'s surname'
    D_FO = 'father\'s occupation'
    D_MN = 'mother\'s forename'
    D_MS = 'mother\'s maiden surname'
    D_MO = 'mother\'s occupation'

  elif data_set == 'kil':
    # BIRTH RECORD DETAILS
    B_ID = 'ID'

    B_N = 'child\'s forname(s)'
    B_S = 'child\'s surname'
    B_DATE = 'date of birth'
    B_A1 = 'address 2'
    B_A2 = 'address 3'
    B_SEX = 'sex'
    B_FN = 'father\'s forename'
    B_FS = 'father\'s surname'
    B_FO = 'father\'s occupation'
    B_MN = 'mother\'s forename'
    B_MS = 'mother\'s maiden surname'
    B_MO = 'mother\'s occupation'
    B_PM_D = 'day of parents\' marriage'
    B_PM_M = 'month of parents\' marriage'
    B_PM_Y = 'parmaryear'
    B_PM_A1 = 'place of parent\'s marriage 1'
    B_PM_A2 = 'place of parent\'s marriage 2'

    # MARRIAGE RECORD DETAILS
    M_ID = 'ID'
    M_DATE = 'date of marriage'
    M_PLACE1 = 'place of marriage 2'
    M_PLACE2 = 'place of marriage 3'
    M_D = 'day'
    M_M = 'month'
    M_Y = 'year'
    M_GN = 'forename of groom'
    M_GS = 'surname of groom'
    M_GO = 'occupation of groom'
    M_G_MARRITAL = 'marital status of groom'
    M_G_AGE = 'age of groom'
    M_G_A1 = 'address of groom 2'
    M_G_A2 = 'address of groom 3'
    M_BN = 'forename of bride'
    M_BS = 'surname of bride'
    M_BO = 'occupation of bride'
    M_B_MARRITAL = 'marital status of bride'
    M_B_A1 = 'address of bride 2'
    M_B_A2 = 'address of bride 3'
    M_G_FN = 'groom\'s father\'s forename'
    M_G_FS = 'groom\'s father\'s surname'
    M_G_FO = 'groom\'s father\'s occupation'
    M_G_MN = 'groom\'s mother\'s forename'
    M_G_MS = 'groom\'s mother\'s maiden surname'
    M_B_FN = 'bride\'s father\'s forename'
    M_B_FS = 'bride\'s father\'s surname'
    M_B_FO = 'bride\'s father\'s occupation'
    M_B_MN = 'bride\'s mother\'s forename'
    M_B_MS = 'bride\'s mother\'s maiden surname'
    M_B_A = 'age of bride'
    M_G_A = 'age of groom'

    # COMMON
    PARISH = 'address 3'
    M_PARISH = 'place of marriage 3'

    # DEATH RECORD DETAILS
    D_ID = 'ID'
    D_N = 'forename(s) of deceased'
    D_S = 'surname of deceased'
    D_O = 'occupation'
    D_MARRITAL = 'marital status'  # todo use it
    D_SEX = 'sex'
    D_A = 'age at death in years (decimal)'
    D_SN = 'forename of spouse'
    D_SS = 'surname of spouse'
    D_SO = 'spouse\'s occ'
    D_DATE = 'date of death'
    D_D = 'day'
    D_M = 'month'
    D_Y = 'year'
    D_A1 = 'tidy address2'
    D_A2 = 'address 3'
    D_FN = 'father\'s forename'
    D_FS = 'father\'s surname'
    D_FO = 'father\'s occupation'
    D_MN = 'mother\'s forename'
    D_MS = 'mother\'s maiden surname'
    D_MO = 'mother\'s occupation'

  elif data_set == 'bhic' or data_set.startswith('bhic'):
    # BIRTH RECORD DETAILS
    B_ID = 'b_id'

    B_N = 'child_fname'
    B_S = 'child_sname'
    B_DATE = 'event_date'
    B_PLACE = 'source_place'
    B_A1 = ''
    B_A2 = ''
    B_SEX = 'child_gender'
    B_FN = 'father_fname'
    B_FS = 'father_sname'
    B_FO = ''
    B_MN = 'mother_fname'
    B_MS = 'mother_sname'
    B_MO = ''
    B_PM_D = ''
    B_PM_M = ''
    B_PM_Y = ''
    B_PM_A1 = ''
    B_PM_A2 = ''

    # MARRIAGE RECORD DETAILS
    M_ID = 'm_id'
    M_DATE = 'event_date'
    M_PLACE1 = 'event_place'
    M_PLACE2 = ''
    M_D = 'event_day'
    M_M = 'event_month'
    M_Y = 'event_year'
    M_GN = 'groom_fname'
    M_GS = 'groom_sname'
    M_GO = ''
    M_G_MARRITAL = ''
    M_G_AGE = 'groom_age'
    M_G_A1 = ''
    M_G_A2 = ''
    M_G_BP = 'groom_birth_place'
    M_BN = 'bride_fname'
    M_BS = 'bride_sname'
    M_BO = ''
    M_B_MARRITAL = ''
    M_B_A1 = ''
    M_B_A2 = ''
    M_B_BP = 'bride_birth_place'
    M_G_FN = 'groom_father_fname'
    M_G_FS = 'groom_father_sname'
    M_G_FO = ''
    M_G_MN = 'groom_mother_fname'
    M_G_MS = 'groom_mother_sname'
    M_B_FN = 'bride_father_fname'
    M_B_FS = 'bride_father_sname'
    M_B_FO = ''
    M_B_MN = 'bride_mother_fname'
    M_B_MS = 'bride_mother_sname'
    M_B_A = 'bride_age'
    M_G_A = 'groom_age'
    M_B_BY = 'bride_birth_year'
    M_G_BY = 'groom_birth_year'

    # DEATH RECORD DETAILS
    D_ID = 'd_id'
    D_N = 'deceased_fname'
    D_S = 'deceased_sname'
    D_O = ''
    D_MARRITAL = ''
    D_SEX = 'deceased_gender'
    D_A = 'deceased_age'
    D_SN = ''
    D_SS = ''
    D_SO = ''
    D_DATE = 'event_date'
    D_D = 'event_day'
    D_M = 'event_month'
    D_Y = 'event_year'
    D_A1 = ''
    D_A2 = ''
    D_FN = 'father_fname'
    D_FS = 'father_sname'
    D_FO = ''
    D_MN = 'mother_fname'
    D_MS = 'mother_sname'
    D_MO = ''

  elif data_set in ['dblp-acm1', 'dblp-scholar1', 'ipums', 'mb', 'msd'] :
    pass
  else:
    raise Exception("Invalid data set")
