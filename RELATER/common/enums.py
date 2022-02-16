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
class Role():
  Bb = 'Bb'
  Bp = 'Bp'
  Mm = 'Mm'
  Mbp = 'Mbp'
  Mgp = 'Mgp'
  Dd = 'Dd'
  Ds = 'Ds'
  Dp = 'Dp'

  @classmethod
  def list(cls):
    return ['Bb', 'Bp', 'Mm', 'Mbp', 'Mgp', 'Dd', 'Ds', 'Dp']


class HH_Role():
  C = 'C'
  M = 'M'
  F = 'F'
  S = 'S'

  @classmethod
  def list(cls):
    # Manually listed to preserve the order
    return ['C', 'M', 'F', 'S']

class Sex():
  F = 'F'
  M = 'M'


class State():
  ACTIVE = 'A'
  INACTIVE = 'I'
  MERGED = 'M'
  NONMERGE = 'N'


class Bib_Edge_Type():
  AUTHOR = 'a'
  COAUTHOR = 'ca'
  PUBLICATION = 'p'

  @classmethod
  def list(cls):
    return ['a', 'ca', 'p']

class HH_Edge_Type():
  MOTHER = 'm'
  FATHER = 'f'
  CHILDREN = 'c'
  SPOUSE = 's'
  SIBLING = 'sb'

  @classmethod
  def list(cls):
    # Manually listed to preserve the order
    # return ['m', 'f', 'c', 's', 'sb']
    return ['m', 'f', 'c', 's']


class Edge_Type():
  MOTHER = 'm'
  FATHER = 'f'
  CHILDREN = 'c'
  SPOUSE = 's'

  @classmethod
  def list(cls):
    return ['m', 'f', 'c', 's']

  @classmethod
  def parents(cls):
    return ['m', 'f']

class HH_Links():
  FF = 'F-F'
  MM = 'M-M'
  CC = 'C-C'
  # CF = 'C-F'
  # CM = 'C-M'

  @classmethod
  def list(cls):
    # Manually listed to preserve the order
    return ['F-F', 'M-M', 'C-C']

class Links():
  # Birth baby links
  BbBp = 'Bb-Bp'
  BbMm = 'Bb-Mm'
  BbMbp = 'Bb-Mbp'
  BbMgp = 'Bb-Mgp'
  BbDd = 'Bb-Dd'
  BbDs = 'Bb-Ds'
  BbDp = 'Bb-Dp'

  # Birth parent Links
  BpBp = 'Bp-Bp'
  BpMm = 'Bp-Mm'
  BpMbp = 'Bp-Mbp'
  BpMgp = 'Bp-Mgp'
  BpDd = 'Bp-Dd'
  BpDs = 'Bp-Ds'
  BpDp = 'Bp-Dp'

  # Marriage bride/groom links
  MmMm = 'Mm-Mm'
  MmMbp = 'Mm-Mbp'
  MmMgp = 'Mm-Mgp'
  MmDd = 'Mm-Dd'
  MmDs = 'Mm-Ds'
  MmDp = 'Mm-Dp'

  # Marriage brideparent links
  MbpMbp = 'Mbp-Mbp'
  MbpMgp = 'Mbp-Mgp'
  MbpDd = 'Mbp-Dd'
  MbpDs = 'Mbp-Ds'
  MbpDp = 'Mbp-Dp'

  # Marriage groomparent links
  MgpMgp = 'Mgp-Mgp'
  MgpDd = 'Mgp-Dd'
  MgpDs = 'Mgp-Ds'
  MgpDp = 'Mgp-Dp'

  # Deceased person links
  DdDs = 'Dd-Ds'
  DdDp = 'Dd-Dp'

  # Deceased spouse links
  DsDs = 'Ds-Ds'
  DsDp = 'Ds-Dp'

  # Deceased parent links
  DpDp = 'Dp-Dp'

  @classmethod
  def list(cls):
    # Manually listed to preserve the order
    return ['Bb-Bp', 'Bb-Mm', 'Bb-Mbp', 'Bb-Mgp', 'Bb-Dd', 'Bb-Ds', 'Bb-Dp',
            'Bp-Bp',
            'Bp-Mm', 'Bp-Mbp', 'Bp-Mgp', 'Bp-Dd', 'Bp-Ds', 'Bp-Dp', 'Mm-Mm',
            'Mm-Mbp',
            'Mm-Mgp', 'Mm-Dd', 'Mm-Ds', 'Mm-Dp', 'Mbp-Mbp', 'Mbp-Mgp', 'Mbp-Dd',
            'Mbp-Ds', 'Mbp-Dp', 'Mgp-Mgp', 'Mgp-Dd', 'Mgp-Ds', 'Mgp-Dp',
            'Dd-Ds',
            'Dd-Dp', 'Ds-Ds', 'Ds-Dp', 'Dp-Dp']

  @classmethod
  def temporally_adjacent_list(cls):
    # Manually listed to preserve the order
    return ['Bb-Dd',
            'Bp-Bp', 'Bp-Mm', 'Bp-Dd', 'Bp-Ds', 'Bp-Dp',
            'Mm-Mm', 'Mm-Dd', 'Mm-Ds', 'Mm-Dp',
            'Mbp-Mbp', 'Mbp-Mgp', 'Mbp-Dd', 'Mbp-Ds', 'Mbp-Dp',
            'Mgp-Mgp', 'Mgp-Dd', 'Mgp-Ds', 'Mgp-Dp',
            'Dd-Ds', 'Dd-Dp', 'Ds-Ds', 'Ds-Dp', 'Dp-Dp']

  @classmethod
  def get_singleton_links(cls):
    """
    Class method to provide the one to one and one to many links.
    """
    # Manually listed to preserve the order
    return ['Bb-Dd', 'Bb-Bp', 'Bb-Mm', 'Bb-Mbp', 'Bb-Mgp', 'Bb-Ds', 'Bb-Dp',
            'Bp-Dd', 'Mm-Dd', 'Mbp-Dd', 'Mgp-Dd', 'Dd-Ds', 'Dd-Dp']

