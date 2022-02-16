import calendar
import csv
import logging
import os
import pickle
import re
from collections import defaultdict, Counter
from datetime import datetime
from data import model
from common import constants as c
from common import settings

from common.enums import Role, Sex
from common import util


# ---------------------------------------------------------------------------
# Preprocess data
# ---------------------------------------------------------------------------

def __preprocess_address__(data_set, address_index):
  pass
  # std = model.get_standard_string
  # for r in data_set.itervalues():
  #   if address_index != '':
  #     address = std(r[address_index])
  #     if address is not None:
  #       if 'street' in address:
  #         r[address_index] = address.replace('street', 'st')
  #       if 'place' is address:
  #         r[address_index] = address.replace('place', 'pl')
  #       if 'road' is address:
  #         r[address_index] = address.replace('road', 'rd')


def __preprocess_birth_data_kil__(p_births):
  std = model.get_standard_string
  for r in p_births.itervalues():
    parent_my = std(r[c.B_PM_Y])
    parent_my = int(parent_my) \
      if std(r[c.B_PM_Y]) is not None and parent_my.isdigit() else None

    if parent_my < 1800 or parent_my > 1901:
      parent_my = None
    r[c.B_PM_Y] = str(parent_my)


# ---------------------------------------------------------------------------
# Person Record Enumeration
# ---------------------------------------------------------------------------

def __retrieve_certificate_data__(filename, cert_id):
  """
  Method to retrieve data from the original certifcate CSV files.

  :param filename: File name
  :param cert_id: Certificate ID
  :return: A dictionary with IDs as keys
  """
  record_dict = {}

  # if cert_id == 'IOSBIRTH_Identifier':
  #   rows = 17614
  # elif cert_id == 'IOS_identifier':
  #   rows = 2668
  # else:
  #   rows = 12285
  # Generate random line numbers to read based on the sample size
  # row_count = int(rows * settings.dataset_sample_size)
  # lines_to_grab = random.sample(xrange(rows), row_count)

  # Read file
  with open(filename, 'r') as data:
    reader = csv.DictReader(data)
    i = 0
    for record in reader:
      # if i in lines_to_grab:
      record_dict[record[cert_id]] = record
      i += 1
  logging.info(
    'Retrieved records of {} : {}'.format(cert_id, len(record_dict.keys())))
  return record_dict


def enumerate_hh_records():
  census_list = csv.DictReader((open(settings.census_file, 'r')))
  people_dict = {}
  hh_role_id_dict = defaultdict(dict)
  for row in census_list:
    p = model.new_hh_person(row[c.HH_ID], row[c.HH_GENDER], row[c.HH_ROLE],
                            row[c.HH_RACE], row[c.HH_BPL],
                            row[c.HH_FNAME], row[c.HH_SNAME],
                            row[c.HH_BYEAR], row[c.HH_STATE],
                            row[c.HH_CITY], row[c.HH_OCC],
                            row[c.HH_YEAR], row[c.HH_SERIAL],
                            row[c.HH_PER_NUM])
    if row[c.HH_ROLE] not in hh_role_id_dict[row[c.HH_SERIAL]]:
      hh_role_id_dict[row[c.HH_SERIAL]][row[c.HH_ROLE]] = list()
    hh_role_id_dict[row[c.HH_SERIAL]][row[c.HH_ROLE]].append(p[c.I_ID])
    people_dict[p[c.I_ID]] = p

  for hh in hh_role_id_dict.itervalues():

    # Add mother related relations
    for m_id in hh.get('M', list()):
      m = people_dict[m_id]

      # Mother-father relationship
      for f_id in hh.get('F', list()):
        f = people_dict[f_id]
        m[c.HH_I_SPOUSE] = f_id
        f[c.HH_I_SPOUSE] = m_id

      # Mother-children relationship
      if 'C' in hh:
        m[c.HH_I_CHILDREN] = hh['C']

        for c_id in hh['C']:
          child = people_dict[c_id]
          child[c.HH_I_MOTHER] = m_id

    # Add father related relations
    for f_id in hh.get('F', list()):
      f = people_dict[f_id]

      # Father-children relationship
      if 'C' in hh:
        f[c.HH_I_CHILDREN] = hh['C']

        for c_id in hh['C']:
          child = people_dict[c_id]
          child[c.HH_I_FATHER] = f_id

    # Add sibling related relations
    if 'C' in hh:
      for c_id in hh['C']:
        child = people_dict[c_id]
        sibling_id_list = [sib_id for sib_id in hh['C'] if sib_id != c_id]
        child[c.HH_I_SIBLINGS] = sibling_id_list

  return people_dict


def enumerate_records():
  """
  Method to enumerate people records from certificates

  :return: A dictionary of people with person ID as keys.
  """
  people_dict = {}
  p_id = 0

  std = model.get_standard_string

  # Birth records
  p_births = __retrieve_certificate_data__(settings.birth_file, c.B_ID)
  __preprocess_address__(p_births, c.B_A1)
  if c.data_set == 'kil':
    __preprocess_birth_data_kil__(p_births)

  for r in p_births.itervalues():
    r = defaultdict(lambda: None, r)
    mother, father, baby = None, None, None

    event_date = __retrieve_date__(r[c.B_DATE].strip())
    birth_year = event_date.year if event_date is not None else None
    timestamp = calendar.timegm(event_date.timetuple()) if event_date is not None else None
    cert_id = r[c.B_ID]
    parent_my = std(r[c.B_PM_Y])
    parent_marriage_year = int(parent_my) \
      if std(r[c.B_PM_Y]) is not None and parent_my.isdigit() else None
    birth_place = std(r[c.B_PLACE])

    if std(r[c.B_MN]) is not None or std(r[c.B_MS]) is not None:
      p_id += 1
      mother = model.new_record(p_id, r[c.B_MN], r[c.B_MS], Sex.F,
                                r[c.B_MO], r[c.B_A1], r[c.B_A2], Role.Bp,
                                timestamp, event_date, cert_id,
                                r[c.PARISH], None, parent_marriage_year,
                                r[c.B_PM_A1], r[c.B_PM_A2])
    if std(r[c.B_FN]) is not None or std(r[c.B_FS]) is not None:
      p_id += 1
      father = model.new_record(p_id, r[c.B_FN], r[c.B_FS], Sex.M,
                                r[c.B_FO], r[c.B_A1], r[c.B_A2], Role.Bp,
                                timestamp, event_date, cert_id,
                                r[c.PARISH], None, parent_marriage_year,
                                r[c.B_PM_A1], r[c.B_PM_A2])
    p_id += 1
    sex = __retrieve_gender__(r[c.B_SEX])
    baby = model.new_record(p_id, r[c.B_N], r[c.B_S], sex, None, r[c.B_A1],
                            r[c.B_A2], Role.Bb, timestamp, event_date,
                            cert_id, r[c.PARISH], birth_year, None, None,
                            None, None, birth_place)

    # Adding relationships
    baby[c.I_MOTHER] = __retrieve_pid__(mother)
    baby[c.I_FATHER] = __retrieve_pid__(father)
    people_dict[baby[c.I_ID]] = baby

    if mother is not None:
      mother[c.I_CHILDREN] = __retrieve_pid__(baby)
      mother[c.I_SPOUSE] = __retrieve_pid__(father)
      people_dict[mother[c.I_ID]] = mother

    if father is not None:
      father[c.I_CHILDREN] = __retrieve_pid__(baby)
      father[c.I_SPOUSE] = __retrieve_pid__(mother)
      people_dict[father[c.I_ID]] = father

  b_count = p_id

  # Marriage records
  p_marriages = __retrieve_certificate_data__(settings.marriage_file,
                                              c.M_ID)
  __preprocess_address__(p_marriages, c.M_B_A1)
  __preprocess_address__(p_marriages, c.M_G_A1)
  for r in p_marriages.itervalues():

    r = defaultdict(lambda: None, r)
    b_mother, b_father, bride, g_mother, g_father, groom = None, None, None, \
                                                           None, None, None

    event_date = __retrieve_date__(r[c.M_DATE].strip())
    marriage_y = event_date.year if event_date is not None else None
    timestamp = calendar.timegm(event_date.timetuple()) if event_date is not \
                                                           None else None
    cert_id = r[c.M_ID]

    p_id += 1
    bride_birth_year = r[c.M_B_BY] if r[c.M_B_BY] is not None \
      else __get_birth_year__(marriage_y, r[c.M_B_A])
    bride_birth_place = std(r[c.M_B_BP])
    bride = model.new_record(p_id, r[c.M_BN], r[c.M_BS], Sex.F, r[c.M_BO],
                             r[c.M_B_A1], r[c.M_B_A2], Role.Mm, timestamp,
                             event_date, cert_id, r[c.M_PARISH],
                             bride_birth_year, marriage_y, r[c.M_PLACE1],
                             r[c.M_PLACE2], r[c.M_B_MARRITAL],
                             bride_birth_place)
    p_id += 1
    groom_birth_year = r[c.M_G_BY] if r[c.M_G_BY] is not None \
      else __get_birth_year__(marriage_y, r[c.M_G_A])
    groom_birth_place = std(r[c.M_G_BP])
    groom = model.new_record(p_id, r[c.M_GN], r[c.M_GS], Sex.M, r[c.M_GO],
                             r[c.M_G_A1], r[c.M_G_A2], Role.Mm, timestamp,
                             event_date, cert_id, r[c.M_PARISH],
                             groom_birth_year, marriage_y, r[c.M_PLACE1],
                             r[c.M_PLACE2], r[c.M_G_MARRITAL],
                             groom_birth_place)

    if std(r[c.M_G_FN]) is not None or std(r[c.M_G_FS]) is not None:
      p_id += 1
      g_father = model.new_record(p_id, r[c.M_G_FN], r[c.M_G_FS], Sex.M,
                                  r[c.M_G_FO], r[c.M_G_A1], r[c.M_G_A2],
                                  Role.Mgp, timestamp, event_date, cert_id,
                                  r[c.M_PARISH])

    if std(r[c.M_G_MN]) is not None or std(r[c.M_G_MS]) is not None:
      p_id += 1
      g_mother = model.new_record(p_id, r[c.M_G_MN], r[c.M_G_MS], Sex.F,
                                  '',
                                  r[c.M_G_A1], r[c.M_G_A2], Role.Mgp, timestamp,
                                  event_date, cert_id, r[c.M_PARISH])

    if std(r[c.M_B_FN]) is not None or std(r[c.M_B_FS]) is not None:
      p_id += 1
      b_father = model.new_record(p_id, r[c.M_B_FN], r[c.M_B_FS], Sex.M,
                                  r[c.M_B_FO], r[c.M_B_A1], r[c.M_B_A2],
                                  Role.Mbp, timestamp, event_date, cert_id,
                                  r[c.M_PARISH])
    if std(r[c.M_B_MN]) is not None or std(r[c.M_B_MS]) is not None:
      p_id += 1
      b_mother = model.new_record(p_id, r[c.M_B_MN], r[c.M_B_MS], Sex.F,
                                  '',
                                  r[c.M_B_A1], r[c.M_B_A2], Role.Mbp, timestamp,
                                  event_date, cert_id, r[c.M_PARISH])

    # Relationships
    bride[c.I_SPOUSE] = __retrieve_pid__(groom)
    bride[c.I_MOTHER] = __retrieve_pid__(b_mother)
    bride[c.I_FATHER] = __retrieve_pid__(b_father)

    if b_mother is not None:
      b_mother[c.I_SPOUSE] = __retrieve_pid__(b_father)
      b_mother[c.I_CHILDREN] = __retrieve_pid__(bride)
      people_dict[b_mother[c.I_ID]] = b_mother

    if b_father is not None:
      b_father[c.I_SPOUSE] = __retrieve_pid__(b_mother)
      b_father[c.I_CHILDREN] = __retrieve_pid__(bride)
      people_dict[b_father[c.I_ID]] = b_father

    groom[c.I_SPOUSE] = __retrieve_pid__(bride)
    groom[c.I_MOTHER] = __retrieve_pid__(g_mother)
    groom[c.I_FATHER] = __retrieve_pid__(g_father)

    if g_mother is not None:
      g_mother[c.I_SPOUSE] = __retrieve_pid__(g_father)
      g_mother[c.I_CHILDREN] = __retrieve_pid__(groom)
      people_dict[g_mother[c.I_ID]] = g_mother

    if g_father is not None:
      g_father[c.I_SPOUSE] = __retrieve_pid__(g_mother)
      g_father[c.I_CHILDREN] = __retrieve_pid__(groom)
      people_dict[g_father[c.I_ID]] = g_father

    people_dict[bride[c.I_ID]] = bride
    people_dict[groom[c.I_ID]] = groom

  m_count = p_id - b_count

  # Death records
  p_deaths = __retrieve_certificate_data__(settings.death_file, c.D_ID)
  __preprocess_address__(p_deaths, c.D_A1)
  for r in p_deaths.itervalues():

    r = defaultdict(lambda: None, r)
    d_person, d_spouse, d_mother, d_father = None, None, None, None
    p_id += 1
    sex = __retrieve_gender__(r[c.D_SEX])

    event_date = __retrieve_date__(r[c.D_DATE].strip())
    death_year = event_date.year if event_date is not None else None
    timestamp = calendar.timegm(event_date.timetuple()) if event_date is not None else None
    cert_id = r[c.D_ID]

    d_person = model.new_record(p_id, r[c.D_N], r[c.D_S], sex, r[c.D_O],
                                r[c.D_A1], r[c.D_A2], Role.Dd, timestamp,
                                event_date, cert_id, r[c.PARISH],
                                __get_birth_year__(death_year, r[c.D_A]),
                                None, None, None, r[c.D_MARRITAL])

    if std(r[c.D_SN]) is not None or std(r[c.D_SS]) is not None:
      p_id += 1

      sex = Sex.F if d_person[c.I_SEX] == Sex.M else \
        (Sex.M if d_person[c.I_SEX] == Sex.F else '')
      d_spouse = model.new_record(p_id, r[c.D_SN], r[c.D_SS], sex, r[c.D_SO],
                                  r[c.D_A1], r[c.D_A2], Role.Ds, timestamp,
                                  event_date, cert_id, r[c.PARISH])

    if std(r[c.D_FN]) is not None or std(r[c.D_FS]) is not None:
      p_id += 1
      d_father = model.new_record(p_id, r[c.D_FN], r[c.D_FS], Sex.M,
                                  r[c.D_FO], r[c.D_A1], r[c.D_A2], Role.Dp,
                                  timestamp, event_date, cert_id, r[c.PARISH])

    if std(r[c.D_MN]) is not None or std(r[c.D_MS]) is not None:
      p_id += 1
      d_mother = model.new_record(p_id, r[c.D_MN], r[c.D_MS], Sex.F,
                                  r[c.D_MO], r[c.D_A1], r[c.D_A2], Role.Dp,
                                  timestamp, event_date, cert_id, r[c.PARISH])

    # relationships
    d_person[c.I_SPOUSE] = __retrieve_pid__(d_spouse)
    d_person[c.I_MOTHER] = __retrieve_pid__(d_mother)
    d_person[c.I_FATHER] = __retrieve_pid__(d_father)
    people_dict[d_person[c.I_ID]] = d_person

    if d_spouse is not None:
      d_spouse[c.I_SPOUSE] = __retrieve_pid__(d_person)
      people_dict[d_spouse[c.I_ID]] = d_spouse

    if d_mother is not None:
      d_mother[c.I_CHILDREN] = __retrieve_pid__(d_person)
      d_mother[c.I_SPOUSE] = __retrieve_pid__(d_father)
      people_dict[d_mother[c.I_ID]] = d_mother

    if d_father is not None:
      d_father[c.I_CHILDREN] = __retrieve_pid__(d_person)
      d_father[c.I_SPOUSE] = __retrieve_pid__(d_mother)
      people_dict[d_father[c.I_ID]] = d_father

  d_count = p_id - m_count - b_count

  print 'B Count {}'.format(b_count)
  print 'M count {}'.format(m_count)
  print 'D count {}'.format(d_count)

  return people_dict


# ---------------------------------------------------------------------------
# Ground Truth Record Pair Enumeration
# ---------------------------------------------------------------------------

def retrieve_ground_truth_links(graph=None):
  filename = settings.gt_file
  if os.path.isfile(filename):
    links_dict = pickle.load(open(filename, 'rb'))
  else:
    if c.data_set == 'ios':
      links_dict = retrieve_ground_truth_links_ios()
    elif c.data_set == 'kil':
      links_dict = retrieve_ground_truth_links_kil()
    elif c.data_set == 'bhic':
      links_dict = retrieve_ground_truth_links_bhic(graph)
    else:
      raise Exception('Unsupported dataset')
  return links_dict


birth_cert_id_record_dict = {}


def __get_r1_r2__(record_dict, node, link):
  role1, role2 = link.split('-')[0], link.split('-')[1]

  p1 = record_dict[node[c.R1]]
  p2 = record_dict[node[c.R2]]
  if p1[c.I_ROLE] == role1:
    r1, r2 = p1, p2
  elif p2[c.I_ROLE] == role2:
    r1, r2 = p2, p1
  else:
    raise Exception('Wait what')

  return r1, r2


def __get_parent_link__(link, m_link, f_link, r2):
  if link == 'Bb-Mm':
    person_sex = r2[c.I_SEX]
    if person_sex == Sex.M:
      p_link = m_link
    elif person_sex == Sex.F:
      p_link = f_link
    else:
      raise Exception('Unknown sex')
  else:
    p_link = m_link
  return p_link


def retrieve_ground_truth_links_bhic(graph):
  N_ATOMIC, TYPE1, TYPE2, SIM = c.N_ATOMIC, c.TYPE1, c.TYPE2, c.SIM

  I_FNAME, I_SNAME, I_BIRTH_PLACE, I_BIRTH_YEAR, I_ROLE, I_MOTHER, I_FATHER, \
  I_SEX, I_ID = c.I_FNAME, c.I_SNAME, c.I_BIRTH_PLACE, c.I_BIRTH_YEAR, \
                c.I_ROLE, c.I_MOTHER, c.I_FATHER, c.I_SEX, c.I_ID

  role_counter = defaultdict(Counter)
  role_t_counter = defaultdict(Counter)
  parent_none_counter = Counter()

  counted_bb_set = set()

  # Counting co-occurences of bb and mm
  for node_id, node in graph.G.nodes(data=True):
    if node[TYPE2] == 'Bb-Mm':

      p1 = graph.record_dict[node[c.R1]]
      p2 = graph.record_dict[node[c.R2]]
      if p1[I_ROLE] == 'Bb':
        r1, r2 = p1, p2
      elif p1[I_ROLE] == 'Mm':
        r1, r2 = p2, p1
      else:
        raise Exception('Wait what')

      # Birth babies
      #
      if r1[I_ID] not in counted_bb_set:
        # Getting parents of bb and mm
        bm = None if r1[I_MOTHER] is None else graph.record_dict[r1[I_MOTHER]]
        bm_str = 'None' if bm is None else str(bm[I_FNAME]) + str(bm[I_SNAME])
        bf = None if r1[I_FATHER] is None else graph.record_dict[r1[I_FATHER]]
        bf_str = 'None' if bf is None else str(bf[I_FNAME]) + str(bf[I_SNAME])

        bb_key = str(r1[I_FNAME]) + str(r1[I_SNAME]) + str(r1[I_BIRTH_PLACE])
        r1.append(bb_key)
        role_counter['Bb'].update([bb_key])

        bb_t_key = bb_key + bm_str + bf_str
        r1.append(bb_t_key)
        role_t_counter['Bb'].update([bb_t_key])

        if bm is None and bf is None:
          parent_none_counter.update(['Bb'])

        counted_bb_set.add(r1[I_ID])

      # Marriage bride/grooms
      mmm = None if r2[I_MOTHER] is None else graph.record_dict[r2[I_MOTHER]]
      mmm_str = 'None' if mmm is None else str(mmm[I_FNAME]) + str(mmm[I_SNAME])
      mf = None if r2[I_FATHER] is None else graph.record_dict[r2[I_FATHER]]
      mf_str = 'None' if mf is None else str(mf[I_FNAME]) + str(mf[I_SNAME])

      mm_key = str(r2[I_FNAME]) + str(r2[I_SNAME]) + str(r2[I_BIRTH_PLACE])
      r2.append(mm_key)
      role_counter['Mm'].update([mm_key])

      mm_t_key = mm_key + mmm_str + mf_str
      r2.append(mm_t_key)
      role_t_counter['Mm'].update([mm_t_key])

      if mmm is None and mf is None:
        parent_none_counter.update(['Mm'])

    if node[TYPE2] == 'Bb-Dd':

      p1 = graph.record_dict[node[c.R1]]
      p2 = graph.record_dict[node[c.R2]]
      if p1[I_ROLE] == 'Bb':
        r1, dd = p1, p2
      elif p1[I_ROLE] == 'Dd':
        r1, dd = p2, p1
      else:
        raise Exception('Wait what')

      # Birth babies
      #
      if r1[I_ID] not in counted_bb_set:
        # Getting parents of bb
        bm = None if r1[I_MOTHER] is None else graph.record_dict[r1[I_MOTHER]]
        bm_str = 'None' if bm is None else str(bm[I_FNAME]) + str(bm[I_SNAME])
        bf = None if r1[I_FATHER] is None else graph.record_dict[r1[I_FATHER]]
        bf_str = 'None' if bf is None else str(bf[I_FNAME]) + str(bf[I_SNAME])

        bb_key = str(r1[I_FNAME]) + str(r1[I_SNAME]) + str(r1[I_BIRTH_PLACE])
        r1.append(bb_key)
        role_counter['Bb'].update([bb_key])

        bb_t_key = bb_key + bm_str + bf_str
        r1.append(bb_t_key)
        role_t_counter['Bb'].update([bb_t_key])

        counted_bb_set.add(r1[I_ID])

        if bm is None and bf is None:
          parent_none_counter.update(['Bb'])

      # Death deceased
      #
      dm = None if dd[I_MOTHER] is None else graph.record_dict[dd[I_MOTHER]]
      dm_str = 'None' if dm is None else str(dm[I_FNAME]) + str(dm[I_SNAME])
      df = None if dd[I_FATHER] is None else graph.record_dict[dd[I_FATHER]]
      df_str = 'None' if df is None else str(df[I_FNAME]) + str(df[I_SNAME])

      dd_key = str(dd[I_FNAME]) + str(dd[I_SNAME]) + str(dd[I_BIRTH_PLACE])
      dd.append(dd_key)
      role_counter['Dd'].update([dd_key])

      dd_t_key = dd_key + dm_str + df_str
      dd.append(dd_t_key)
      role_t_counter['Dd'].update([dd_t_key])

      if dm is None and df is None:
        parent_none_counter.update(['Dd'])

  bb_list = filter(lambda p: p[I_ROLE] == 'Bb', graph.record_dict.itervalues())
  mm_list = filter(lambda p: p[I_ROLE] == 'Mm', graph.record_dict.itervalues())
  dd_list = filter(lambda p: p[I_ROLE] == 'Dd', graph.record_dict.itervalues())
  logging.info('Bb count {}, unique t count {}'
               .format(len(bb_list), len(role_t_counter['Bb'])))
  logging.info('Mm count {}, unique t count {}'
               .format(len(mm_list), len(role_t_counter['Mm'])))
  logging.info('Dd count {}, unique t count {}'
               .format(len(dd_list), len(role_t_counter['Dd'])))
  for role in ['Bb', 'Dd', 'Mm']:
    logging.info('{} parent none count {}'.format(role,
                                                  parent_none_counter.get(
                                                    role)))
    filename = '{}unique-t-{}.csv'.format(settings.output_home_directory, role)

    with open(filename, 'w') as f:
      fieldnames = ['key', 'count']
      writer = csv.writer(f)
      writer.writerow(fieldnames)
      for key, value in role_t_counter[role].items():
        writer.writerow([key, value])

  # Generating gt links
  #
  gt_links_dict = defaultdict(list)
  byear_diff = defaultdict(Counter)
  unique_triangle_match_count = defaultdict(int)
  unique_bday_triangle_match_count = defaultdict(int)

  # Keep track of duplicates
  linked_people = defaultdict(lambda: defaultdict(set))
  duplicates = defaultdict(lambda: defaultdict(set))

  # Attributes used to
  person_attributes = [I_FNAME, I_SNAME]
  mother_attributes = [I_FNAME, I_SNAME]
  father_attributes = [I_FNAME, I_SNAME]

  for node_id, node in graph.G.nodes(data=True):
    for link, m_link, f_link in [('Bb-Mm', 'Bp-Mgp', 'Bp-Mbp'),
                                 ('Bb-Dd', 'Bp-Dp', 'Bp-Dp')]:
      if node[TYPE2] == link:

        r1, r2 = __get_r1_r2__(graph.record_dict, node, link)
        role1, role2 = link.split('-')[0], link.split('-')[1]

        # Check if the triangles are unique
        if role_t_counter[role1].get(r1[25]) == 1 and \
            role_t_counter[role2].get(r2[25]) == 1:

          valid = True

          # Check if the attributes are exact
          for i_attr in person_attributes:
            if r1[i_attr] != r2[i_attr]:
              valid = False
          if not valid:
            continue

          if link == 'Bb-Mm' and r1[I_BIRTH_PLACE] != r2[I_BIRTH_PLACE]:
            continue

          # Validate mother attributes
          mother_exists = True if (r1[I_MOTHER] is not None and r2[I_MOTHER] is
                                   not None) else False
          if mother_exists:
            m1 = graph.record_dict[r1[I_MOTHER]]
            m2 = graph.record_dict[r2[I_MOTHER]]
            for i_attr in mother_attributes:
              if m1[i_attr] != m2[i_attr] or m1[i_attr] is None:
                valid = False
          elif (r1[I_MOTHER] is None and r2[I_MOTHER] is None):
            # If both records has no mother, validity is True
            pass
          else:
            # If only one record has mother, validity is False
            valid = False

          # Validate father attributes
          father_exists = True if (r1[I_FATHER] is not None and r2[I_FATHER] is
                                   not None) else False
          if father_exists:
            f1 = graph.record_dict[r1[I_FATHER]]
            f2 = graph.record_dict[r2[I_FATHER]]
            for i_attr in father_attributes:
              if f1[i_attr] != f2[i_attr] or f1[i_attr] is None:
                valid = False
          elif (r1[I_FATHER] is None and r2[I_FATHER] is None):
            # If both records has no father, validity is True
            pass
          else:
            # If only one record has father, validity is False
            valid = False

          if not valid:
            continue

          unique_triangle_match_count[link] += 1

          key = tuple(sorted({node[c.R1], node[c.R2]}))
          gt_links_dict[link].append(key)

          # Checks for duplicates
          if r1[I_ID] in linked_people[link][role1]:
            duplicates[link][role1].add(r1[I_ID])
          if r2[I_ID] in linked_people[link][role2]:
            duplicates[link][role2].add(r1[I_ID])
          linked_people[link][role1].add(r1[I_ID])
          linked_people[link][role2].add(r2[I_ID])

          # Birth year based validation
          if r1[I_BIRTH_YEAR] is None or r2[I_BIRTH_YEAR] is None:
            byear_diff[link].update(['None'])
          else:
            if r1[I_BIRTH_YEAR] == r2[I_BIRTH_YEAR]:
              unique_bday_triangle_match_count[link] += 1
            diff = str(abs(r1[I_BIRTH_YEAR] - r2[I_BIRTH_YEAR]))
            byear_diff[link].update([diff])

          # Adding parental ground truth links
          p_link = __get_parent_link__(link, m_link, f_link, r2)
          if mother_exists:
            m_key = tuple(sorted({r1[I_MOTHER], r2[I_MOTHER]}))
            # assert m_key in graph.relationship_nodes
            gt_links_dict[p_link].append(m_key)
          if father_exists:
            f_key = tuple(sorted({r1[I_FATHER], r2[I_FATHER]}))
            # assert f_key in graph.relationship_nodes
            gt_links_dict[p_link].append(f_key)

  for link in ['Bb-Mm', 'Bb-Dd']:
    logging.info('Unique records info {}: '.format(link))
    logging.info(
      '\tunique triangle match: {}'.format(unique_triangle_match_count[link]))
    logging.info(
      '\tunique bday triangle match: {}'
        .format(unique_bday_triangle_match_count[link]))
    for role, id_set in duplicates[link].iteritems():
      logging.info(
        '\tduplicates of {} in {} : {}'.format(role, link, len(id_set)))

      # Remove duplicates
      #
      for id in id_set:
        logging.info('\t\tDuplicate id {}'.format(id))

        candidate_gt_links = [gt_pair for gt_pair in gt_links_dict[link] if
                              id in gt_pair]
        # Find incorrect gt links
        incorrect_gt_links = list()
        for gt_pair in candidate_gt_links:
          p1 = graph.record_dict[gt_pair[0]]
          p2 = graph.record_dict[gt_pair[1]]
          logging.info('\t\t\t {}'.format(p1))
          logging.info('\t\t\t {} \n'.format(p2))

          # If the triangular string is not matching, consider it as an
          # incorrect gt pair
          if p1[25] != p2[25]:
            incorrect_gt_links.append(gt_pair)

        # Validate either a single pair or none, is correct
        assert len(candidate_gt_links) <= len(incorrect_gt_links) + 1

        # Remove incorrect gt links
        for gt_pair in incorrect_gt_links:
          gt_links_dict[link].remove(gt_pair)

  logging.info('Ground truth links count')
  for key, value in gt_links_dict.iteritems():
    logging.info('{} : {}'.format(key, len(value)))
    logging.info('\t byear diff {} : {}'.format(key, byear_diff[key]))

  # settings.gt_file = '{}gt-{}.pickle'.format(settings.output_home_directory,
  #                         c.data_set)
  pickle.dump(gt_links_dict, open(settings.gt_file, 'wb'))
  return gt_links_dict


# def retrieve_ground_truth_links_bhic(nodes_list, record_dict):
#   # counter_dict = defaultdict(Counter)
#   # for role in ['Bb', 'Bp', 'Mm', 'Mbp', 'Mgp']:
#   #   people_list = filter(lambda r: r[c.I_ROLE] == role,
#   #                        record_dict.itervalues())
#   #   for p in people_list:
#   #     key = str(p[I_FNAME]) + str(p[I_SNAME])
#   #     counter_dict[role].update([key])
#   #
#   gt_links_dict = defaultdict(list)
#   #
#   # f_list = list(filter((lambda p: p[c.I_SEX] in [Sex.F, '']),
#   #                      record_dict.itervalues()))
#   # m_list = list(filter((lambda p: p[c.I_SEX] in [Sex.M, '']),
#   #                      record_dict.itervalues()))
#   #
#   # female_blocks = defaultdict(set)
#   # for female in f_list:
#   #   fname_key = encode.dmetaphone(female[c.I_FNAME])
#   #   sname_key = encode.dmetaphone(female[c.I_SNAME])
#   #
#   #   female_blocks[fname_key].add(female[c.I_ID])
#   #   female_blocks[sname_key].add(female[c.I_ID])
#   #
#   # male_blocks = defaultdict(set)
#   # for male in m_list:
#   #   fname_key = encode.dmetaphone(male[c.I_FNAME])
#   #   sname_key = encode.dmetaphone(male[c.I_SNAME])
#   #
#   #   male_blocks[fname_key].add(male[c.I_ID])
#   #   male_blocks[sname_key].add(male[c.I_ID])
#   #
#   # for block in [female_blocks, male_blocks]:
#   #   for blocking_key, people_id_list in block.iteritems():
#   #     # People are categorized to each role based on gender
#   #     #
#   #     people_list = map(lambda p_id: record_dict[p_id], people_id_list)
#   #     people = {
#   #       'Bb': filter(lambda p: p[c.I_ROLE] == Role.Bb, people_list),
#   #       'Bp': filter(lambda p: p[c.I_ROLE] == Role.Bp, people_list),
#   #       'Mm': filter(lambda p: p[c.I_ROLE] == Role.Mm, people_list),
#   #       'Mbp': filter(lambda p: p[c.I_ROLE] == Role.Mbp, people_list),
#   #       'Mgp': filter(lambda p: p[c.I_ROLE] == Role.Mgp, people_list)
#   #     }
#   #
#   #     for role1, role2 in [('Bb', 'Bp'), ('Bb', 'Mm'), ('Bb', 'Mbp'),
#   #                          ('Bb', 'Mgp'),
#   #                          ('Mm', 'Mbp'), ('Mm', 'Mgp'),
#   #                          ('Bp', 'Mm'), ('Bp', 'Bp'), ('Bp', 'Mbp'),
#   #                          ('Bp', 'Mgp'),
#   #                          ('Mbp', 'Mbp'), ('Mbp', 'Mgp'),
#   #                          ('Mgp', 'Mgp')]:
#   #       people_list1 = people[role1]
#   #       people_list2 = people[role2]
#   #
#   #       link = '%s-%s' % (role1, role2)
#   #
#   #       for p1 in people_list1:
#   #         key1 = str(p1[I_FNAME]) + str(p1[I_SNAME])
#   #         if counter_dict[role1].get(key1) == 1:
#   #           for p2 in people_list2:
#   #             key2 = str(p2[I_FNAME]) + str(p2[I_SNAME])
#   #             if counter_dict[role2].get(key2) == 1:
#   #               if p1[I_FNAME] == p2[I_FNAME] and p1[I_SNAME] == p2[I_SNAME]:
#   #                 key = tuple(sorted({p1[c.I_ID], p2[c.I_ID]}))
#   #                 gt_links_dict[link].append(key)
#
#   I_FNAME, I_SNAME, I_BIRTH_PLACE, I_BIRTH_YEAR = \
#     c.I_FNAME, c.I_SNAME, c.I_BIRTH_PLACE, c.I_BIRTH_YEAR
#   for gender in [Sex.F, Sex.M]:
#
#     logging.info('Gender %s' % gender)
#
#     bb_list = filter(lambda r: r[c.I_ROLE] == 'Bb' and r[c.I_SEX] in
#                                [gender, ''], record_dict.itervalues())
#     mm_list = filter(lambda r: r[c.I_ROLE] == 'Mm' and r[c.I_SEX] in
#                                [gender, ''], record_dict.itervalues())
#
#     # Finding unique values
#     bb_counter = Counter()
#     mm_counter = Counter()
#     for bb in bb_list:
#       key = str(bb[I_FNAME]) + str(bb[I_SNAME]) + str(bb[I_BIRTH_PLACE]) + \
#             str(bb[I_BIRTH_YEAR])
#       bb.append(key)
#       bb_counter.update([key])
#     for mm in mm_list:
#       key = str(mm[I_FNAME]) + str(mm[I_SNAME]) + str(mm[I_BIRTH_PLACE]) + \
#             str(mm[I_BIRTH_YEAR])
#       mm.append(key)
#       mm_counter.update([key])
#
#     bb_unique_count = len(filter(lambda x: x == 1, bb_counter.itervalues()))
#     logging.info('Bb count %s' % len(bb_list))
#     logging.info('Bb unique count %s' % bb_unique_count)
#
#     mm_unique_count = len(filter(lambda x: x == 1, mm_counter.itervalues()))
#     logging.info('Mm count %s' % len(mm_list))
#     logging.info('Mm unique count %s' % mm_unique_count)
#
#     # Blocking
#     bb_blocks = defaultdict(set)
#     mm_blocks = defaultdict(set)
#     for bb in bb_list:
#       fname_key = encode.dmetaphone(bb[c.I_FNAME])
#       sname_key = encode.dmetaphone(bb[c.I_SNAME])
#       bb_blocks[fname_key + sname_key].add(bb[c.I_ID])
#     for mm in mm_list:
#       fname_key = encode.dmetaphone(mm[c.I_FNAME])
#       sname_key = encode.dmetaphone(mm[c.I_SNAME])
#       mm_blocks[fname_key + sname_key].add(mm[c.I_ID])
#
#     for key, bb_blocked_list in bb_blocks.iteritems():
#       for bb_id in bb_blocked_list:
#
#         bb = record_dict[bb_id]
#         if bb_counter.get(bb[24]) == 1:
#
#           for mm_id in mm_blocks[key]:
#             mm = record_dict[mm_id]
#             if mm_counter.get(mm[24]) == 1:
#
#               if bb[I_FNAME] == mm[I_FNAME] \
#                   and bb[I_SNAME] == mm[I_SNAME] \
#                   and bb[I_BIRTH_PLACE] == mm[I_BIRTH_PLACE] \
#                   and bb[I_BIRTH_YEAR] == mm[I_BIRTH_YEAR]:
#                 key = tuple(sorted({bb[c.I_ID], mm[c.I_ID]}))
#                 gt_links_dict['Bb-Mm'].append(key)
#
#   logging.info('Ground truth values unique fname-sname-bpl-byear')
#   for link, value_set in gt_links_dict.iteritems():
#     logging.info('\t%s - %s' % (link, len(value_set)))
#
#   pickle.dump(gt_links_dict, open(settings.gt_file, 'wb'))
#   return gt_links_dict


def __enrich_lifeseg_dict__(p1_lifesegid, p1_role, child1_cert_id,
                            p2_lifesegid, p2_role, child2_cert_id,
                            life_seg_dict, life_seg_id_index, sex,
                            parent_role_certid_lifesegid_dict):
  if p1_lifesegid is None and p2_lifesegid is None:
    entity = defaultdict(set)
    entity['S'] = sex
    entity[p1_role].add(child1_cert_id)
    entity[p2_role].add(child2_cert_id)
    life_seg_id_index += 1
    life_seg_dict[life_seg_id_index] = entity

    parent_role_certid_lifesegid_dict[p1_role][
      child1_cert_id] = life_seg_id_index
    parent_role_certid_lifesegid_dict[p2_role][
      child2_cert_id] = life_seg_id_index

  elif p1_lifesegid is not None and p2_lifesegid is not None:
    if p1_lifesegid != p2_lifesegid:
      p1_lifeseg = life_seg_dict[p1_lifesegid]
      p2_lifeseg = life_seg_dict[p2_lifesegid]
      assert p1_lifeseg['S'] == p2_lifeseg['S']
      for role in Role.list():
        p1_lifeseg[role] = p1_lifeseg[role].union(p2_lifeseg[role])
  elif p1_lifesegid is not None:
    p1_lifeseg = life_seg_dict[p1_lifesegid]
    assert child1_cert_id in p1_lifeseg[p1_role]
    p1_lifeseg[p2_role].add(child2_cert_id)

    # if p1_role == 'Bp' and p2_role == 'Bp':
    #   p1_fname = birth_cert_id_record_dict[list(p1_lifeseg['Bp'])[0]][
    #     "mother's forename"]
    #   if p1_fname != birth_cert_id_record_dict[child2_cert_id]["mother's " \
    #                                                            "forename"]:
    #     print'o'
  elif p2_lifesegid is not None:
    p2_lifeseg = life_seg_dict[p2_lifesegid]
    assert child2_cert_id in p2_lifeseg[p2_role]
    p2_lifeseg[p1_role].add(child1_cert_id)

    # if p1_role == 'Bp' and p2_role == 'Bp':
    #   p2_fname = birth_cert_id_record_dict[list(p2_lifeseg['Bp'])[0]][
    #     "mother's forename"]
    #   if p2_fname != birth_cert_id_record_dict[child1_cert_id]["mother's " \
    #                                                            "forename"]:
    #     print'o'

  return life_seg_id_index


def retrieve_ground_truth_links_ios():
  # Dictionary to hold certificate ID to person records dictionary
  cert_id_person_dict = {
    'Bb': {},
    'Dd': {},
    'Mm': {}
  }

  # Getting birth ID to certificate ID dictionary
  #
  birth_id_cert_id_dict = {}
  with open(settings.birth_file, 'r') as data:
    reader = csv.DictReader(data)
    for record in reader:
      birth_id_cert_id_dict[record['ID']] = record[c.B_ID]
      cert_id_person_dict['Bb'][record[c.B_ID]] = record
      birth_cert_id_record_dict[record[c.B_ID]] = record

  # Getting marriage ID to certificate ID dictionary
  #
  marriage_id_cert_id_dict = {}
  with open(settings.marriage_file, 'r') as data:
    reader = csv.DictReader(data)
    for record in reader:
      marriage_id_cert_id_dict[record['ID']] = record[c.M_ID]
      cert_id_person_dict['Mm'][record[c.M_ID]] = record

  # Getting death ID to certificate ID dictionary
  #
  death_id_cert_id_dict = {}
  with open(settings.death_file, 'r') as data:
    reader = csv.DictReader(data)
    for record in reader:
      death_id_cert_id_dict[record['ID']] = record[c.D_ID]
      cert_id_person_dict['Dd'][record[c.D_ID]] = record

  # Getting life segment dictionary
  #

  # Dictionary to hold entities with life segment IDs as keys
  life_seg_dict = {}
  life_seg_id_index = 0

  # Dictionary to hold entity families with sibling IDs as keys
  sib_list_dict = {}

  # Dictionaries to index some relationships to ease building up
  # of indirect relationships
  m_id_b_id_dict = {}
  b_id_life_seg_id_dict = {}
  m_id_life_seg_id_dict = {}

  with open(settings.life_segments_file, 'r') as data:

    reader = csv.DictReader(data)
    for record in reader:

      entity = defaultdict(set)
      life_seg_id = record['lifesegmentID']

      # Get sex
      entity['S'] = __retrieve_gender__(record['sex'])

      # Get sibling group
      entity['F'] = record['sibsetID']
      if entity['F'] != '':
        if entity['F'] not in sib_list_dict:
          sib_list_dict[entity['F']] = list()
        sib_list_dict[entity['F']].append(life_seg_id)

      # Get parent marriage
      entity['P'] = marriage_id_cert_id_dict.get(record['parentmarriageID'],
                                                 None)

      # Get birth ID
      #
      b_cert_id = birth_id_cert_id_dict.get(record['BirthID'], None)
      if b_cert_id is not None:
        entity['Bb'].add(b_cert_id)
        # Generate birth ID to life segment ID dictionary
        b_id_life_seg_id_dict[b_cert_id] = life_seg_id

      # Get death ID
      #
      d_cert_id = death_id_cert_id_dict.get(record['DeathID'], None)
      if d_cert_id is not None:
        entity['Dd'].add(d_cert_id)

      # Get marriage IDs
      #
      for male in ['marriageID1', 'marriageID2', 'marriageID3',
                   'marriageID4']:
        m_cert_id = marriage_id_cert_id_dict.get(record[male], None)
        if m_cert_id is not None:
          entity['Mm'].add(m_cert_id)

          # Generate marriage ID to birth ID dictionary
          if m_cert_id not in m_id_b_id_dict:
            m_id_b_id_dict[m_cert_id] = list()
          if b_cert_id is not None:
            m_id_b_id_dict[m_cert_id].append(b_cert_id)

          # Generate marriage ID to life segment dictionary
          if m_cert_id not in m_id_life_seg_id_dict:
            m_id_life_seg_id_dict[m_cert_id] = list()
          m_id_life_seg_id_dict[m_cert_id].append(life_seg_id)

      life_seg_dict[int(life_seg_id)] = entity
      life_seg_id_index = max(life_seg_id_index, int(life_seg_id))

  # Optimizations
  female = Sex.F
  male = Sex.M

  # Iterate through the life segments to identify birth parents, marriage
  # parents, death parents and death spouses.
  #
  for life_seg_id, life_seg in life_seg_dict.iteritems():

    # For each life segment, which has a parent marriage associated with,
    # update the parent's life segments with Bp, Mp and Dp roles.
    #
    if life_seg['P'] is not None:

      # Each parents life segment is updated with their roles
      for parent_life_seg_id in m_id_life_seg_id_dict[life_seg['P']]:

        parent_life_seg_id = int(parent_life_seg_id)

        # Get the parent
        parent = life_seg_dict[parent_life_seg_id]

        # If this life segment has a birth record, update parent's Bp role
        if len(life_seg['Bb']) == 1:
          bb = cert_id_person_dict['Bb'][next(iter(life_seg['Bb']))]
          if parent['S'] == female and bb[c.B_MN] == '' and bb[c.B_MS] \
              == '':
            print 'skipping, no data'
          elif parent['S'] == male and bb[c.B_FN] == '' and bb[c.B_FS] \
              == '':
            print 'skipping, no data'
          else:
            baby_birth_id = list(life_seg['Bb'])[0]
            parent['Bp'].add(baby_birth_id)

        # If this life segment has a death record, update parent's Dp role
        if len(life_seg['Dd']) == 1:
          dd = cert_id_person_dict['Dd'][next(iter(life_seg['Dd']))]
          if parent['S'] == female and dd[c.D_MN] == '' and dd[c.D_MS] \
              == '':
            print 'wrong deceased parent link'
          elif parent['S'] == male and dd[c.D_FN] == '' and dd[c.D_FS] \
              == '':
            print 'wrong deceased parent link'
          else:
            deceased_id = list(life_seg['Dd'])[0]
            parent['Dp'].add(deceased_id)

        # If this life segment has marriage records, update parent's Mp role
        # No ground truth here
        for marriage_id in life_seg['Mm']:
          mm = cert_id_person_dict['Mm'][marriage_id]

          if life_seg['S'] == female:
            if parent['S'] == female and mm[c.M_B_MN] == '' and mm[
              c.M_B_MS] == '':
              print 'wrong mb parent link'
            elif parent['S'] == male and mm[c.M_B_FN] == '' and mm[
              c.M_B_FS] == '':
              print 'wrong deceased parent link'
            else:
              parent['Mbp'].add(marriage_id)

          elif life_seg['S'] == male:
            if parent['S'] == female and mm[c.M_G_MN] == '' and mm[
              c.M_G_MS] == '':
              print 'wrong deceased parent link'
            elif parent['S'] == male and mm[c.M_G_FN] == '' and mm[
              c.M_G_FS] == '':
              print 'wrong deceased parent link'
            else:
              parent['Mgp'].add(marriage_id)
          else:
            raise Exception('No sex')

    # For each life segment, which has a death and marriage(s) associated with,
    # update the marriage spouse's Ds role with the death certificate ID.
    # NOTE: WRONG
    # Death spouse can be one of the marriage spouses/ not all, this can be
    # wrong and can be correct.
    #
    # if len(life_seg['Dd']) == 1:
    #
    #   # Check if the death record has a spouse
    #   dd = cert_id_person_dict['Dd'][next(iter(life_seg['Dd']))]
    #   if dd[c.D_SN] == '' and dd[c.D_SS] == '':
    #     continue
    #
    #   # Iterate through the marriages
    #   for marriage_id in life_seg['Mm']:
    #     life_seg_id_list = m_id_life_seg_id_dict[marriage_id]
    #
    #     assert len(life_seg_id_list) == 2
    #     spouse_life_seg_id = life_seg_id_list[0] \
    #       if life_seg_id_list[1] == life_seg_id else life_seg_id_list[1]
    #     life_seg_dict[spouse_life_seg_id]['Ds'].add(list(life_seg['Dd'])[0])

  __enumerate_indirect_gt_links__(life_seg_dict, sib_list_dict,
                                  cert_id_person_dict, life_seg_id_index)

  links_dict = util.enumerate_links(life_seg_dict.values())
  # persist_gt_entities(life_seg_dict)
  # for key in sorted(links_dict.iterkeys()):
  #   print key, len(links_dict[key])
  pickle.dump(links_dict, open(settings.gt_file, 'wb'))
  return links_dict


def __enumerate_indirect_gt_links__(life_seg_dict, sib_list_dict,
                                    cert_id_person_dict, life_seg_id_index):
  """


  :param life_seg_dict:
  :param sib_list_dict:
  :param cert_id_person_dict:
  :param life_seg_id_index:
  :return:
  """
  std = model.get_standard_string

  mother_role_certid_lifesegid_dict = defaultdict(dict)
  father_role_certid_lifesegid_dict = defaultdict(dict)
  for life_seg_id, life_seg in life_seg_dict.iteritems():
    for role in ['Bp', 'Mbp', 'Mgp', 'Dp']:
      for certid in life_seg[role]:
        if life_seg['S'] == 'F':
          mother_role_certid_lifesegid_dict[role][certid] = life_seg_id
        elif life_seg['S'] == 'M':
          father_role_certid_lifesegid_dict[role][certid] = life_seg_id
        else:
          raise Exception('Missing gender')

  # Optimizations
  is_parent_link_exists = __is_parent_link_exists__

  # Iterate through sibling sets to retrieve their parent links.
  # From each sibling groups we can enumerate Bp-Bp links, Mbp-Mbp links,
  # Mgp-Mgp links and
  # Dp-Dp links considering the parents in births, marriages and deaths of
  # siblings. Additionally, Bp-Dp, Bp-Mp and and Mp-Dp Links can be retrieved.
  #
  for sib_set in sib_list_dict.itervalues():

    for i in xrange(0, len(sib_set)):

      sib1_life_seg = life_seg_dict[int(sib_set[i])]
      sib1_roles_set = util.retrieve_child_parent_role_set(sib1_life_seg)

      for j in xrange(i + 1, len(sib_set)):

        sib2_life_seg = life_seg_dict[int(sib_set[j])]
        sib2_roles_set = util.retrieve_child_parent_role_set(sib2_life_seg)

        # For both sibling 1 and sibling 2, we need to iterate through their
        # main roles (Bb, Mm and Dd) and compare their parents in each role (
        # Bp, Mp and Dp).

        # Iterate through sibling 1's main roles list
        for role1, p1_role in sib1_roles_set:
          sib1_cert_id_set = sib1_life_seg[role1]

          # Iterate through sibling 2's main roles list
          for role2, p2_role in sib2_roles_set:
            sib2_cert_id_set = sib2_life_seg[role2]

            # For both sib1_role in sibling 1 and sib2_role in sibling 2,
            # we need to iterate through their certificates and consider the
            # parent links.

            # Iterate through sib 1 cert IDs
            for sib1_cert_id in sib1_cert_id_set:
              # Iterate through sib 2 cert IDs
              for sib2_cert_id in sib2_cert_id_set:
                # Get the person records for sib 1 and sib 2
                sib1 = cert_id_person_dict[role1][sib1_cert_id]
                sib2 = cert_id_person_dict[role2][sib2_cert_id]

                # Check if the mother links and father links exists. If both
                # sibling's mothers are available, mother link exists.
                # Similarly, if both siblings fathers are available, father
                # link exists.
                link_exists_dict = is_parent_link_exists(sib1, p1_role,
                                                         sib2, p2_role)
                for role, sex, role_dict in [('mother', Sex.F,
                                              mother_role_certid_lifesegid_dict),
                                             ('father', Sex.M,
                                              father_role_certid_lifesegid_dict)]:
                  if link_exists_dict[role]:
                    p1_lifesegid = role_dict[p1_role].get(sib1_cert_id, None)
                    p2_lifesegid = role_dict[p2_role].get(sib2_cert_id, None)

                    life_seg_id_index = \
                      __enrich_lifeseg_dict__(p1_lifesegid, p1_role,
                                              sib1_cert_id,
                                              p2_lifesegid, p2_role,
                                              sib2_cert_id, life_seg_dict,
                                              life_seg_id_index, sex, role_dict)

  # Enrich the ground truth links, if a person to person link is there,
  # the parents also should be the same
  for life_seg_id, life_seg in life_seg_dict.copy().iteritems():

    # Enumerate Bp-Dp links from Bb-Dd links
    if len(life_seg['Bb']) > 0 and len(life_seg['Dd']) > 0:

      b_certid = list(life_seg['Bb'])[0]
      d_certid = list(life_seg['Dd'])[0]

      b_record = cert_id_person_dict['Bb'][b_certid]
      d_record = cert_id_person_dict['Dd'][d_certid]

      if (std(b_record[c.B_MN]) is not None or std(b_record[c.B_MS]) is not
          None) and \
          (std(d_record[c.D_MN]) is not None or std(d_record[c.D_MS]) is not
           None):
        bp_lifesegid = mother_role_certid_lifesegid_dict['Bp'].get(b_certid,
                                                                   None)
        dp_lifesegid = mother_role_certid_lifesegid_dict['Dp'].get(d_certid,
                                                                   None)
        life_seg_id_index = \
          __enrich_lifeseg_dict__(bp_lifesegid, 'Bp', b_certid,
                                  dp_lifesegid, 'Dp', d_certid, life_seg_dict,
                                  life_seg_id_index, Sex.F,
                                  mother_role_certid_lifesegid_dict)
      if (std(b_record[c.B_FN]) is not None or std(b_record[c.B_FS]) is not
          None) and \
          (std(d_record[c.D_FN]) is not None or std(d_record[c.D_FS]) is not
           None):
        bp_lifesegid = father_role_certid_lifesegid_dict['Bp'].get(b_certid,
                                                                   None)
        dp_lifesegid = father_role_certid_lifesegid_dict['Dp'].get(d_certid,
                                                                   None)
        life_seg_id_index = \
          __enrich_lifeseg_dict__(bp_lifesegid, 'Bp', b_certid,
                                  dp_lifesegid, 'Dp', d_certid, life_seg_dict,
                                  life_seg_id_index, Sex.M,
                                  father_role_certid_lifesegid_dict)

    # Enumerate Bp-Mbp and Bp-Mgp links from Bb-Mm links
    if len(life_seg['Bb']) > 0 and len(life_seg['Mm']) > 0:

      b_certid = list(life_seg['Bb'])[0]
      b_record = cert_id_person_dict['Bb'][b_certid]

      b_m_exists = (std(b_record[c.B_MN]) is not None or std(
        b_record[c.B_MS]) is not None)
      b_f_exists = (std(b_record[c.B_FN]) is not None or std(
        b_record[c.B_FS]) is not None)

      assert life_seg['S'] in ['M', 'F']
      bride_groom_type = 'b' if life_seg['S'] == 'F' else 'g'
      for m_certid in life_seg['Mm']:

        m_record = cert_id_person_dict['Mm'][m_certid]
        m_m_exists, m_f_exists = __check_MP_availability(bride_groom_type,
                                                         m_record)
        m_parent_role = 'M{}p'.format(bride_groom_type)

        if b_m_exists and m_m_exists:
          bp_lifesegid = \
            mother_role_certid_lifesegid_dict['Bp'].get(b_certid, None)
          mp_lifesegid = \
            mother_role_certid_lifesegid_dict[m_parent_role].get(m_certid,
                                                                 None)

          life_seg_id_index = \
            __enrich_lifeseg_dict__(bp_lifesegid, 'Bp', b_certid,
                                    mp_lifesegid, m_parent_role, m_certid,
                                    life_seg_dict, life_seg_id_index, Sex.F,
                                    mother_role_certid_lifesegid_dict)
        if b_f_exists and m_f_exists:
          bp_lifesegid = \
            father_role_certid_lifesegid_dict['Bp'].get(b_certid, None)
          mp_lifesegid = \
            father_role_certid_lifesegid_dict[m_parent_role].get(m_certid,
                                                                 None)

          life_seg_id_index = \
            __enrich_lifeseg_dict__(bp_lifesegid, 'Bp', b_certid,
                                    mp_lifesegid, m_parent_role, m_certid,
                                    life_seg_dict, life_seg_id_index, Sex.M,
                                    father_role_certid_lifesegid_dict)

    # Enumerate Dp-Mbp and Dp-Mgp links from Dd-Mm links
    if len(life_seg['Dd']) > 0 and len(life_seg['Mm']) > 0:

      d_certid = list(life_seg['Dd'])[0]
      d_record = cert_id_person_dict['Dd'][d_certid]

      d_m_exists = std(d_record[c.D_MN]) is not None or std(
        d_record[c.D_MS]) is not None
      d_f_exists = std(d_record[c.D_FN]) is not None or std(
        d_record[c.D_FS]) is not None

      assert life_seg['S'] in ['M', 'F']
      bride_groom_type = 'b' if life_seg['S'] == 'F' else 'g'
      for m_certid in life_seg['Mm']:

        m_record = cert_id_person_dict['Mm'][m_certid]
        m_m_exists, m_f_exists = __check_MP_availability(bride_groom_type,
                                                         m_record)
        m_parent_role = 'M{}p'.format(bride_groom_type)

        if d_m_exists and m_m_exists:
          dp_lifesegid = \
            mother_role_certid_lifesegid_dict['Dp'].get(d_certid, None)
          mp_lifesegid = \
            mother_role_certid_lifesegid_dict[m_parent_role].get(m_certid,
                                                                 None)

          life_seg_id_index = \
            __enrich_lifeseg_dict__(dp_lifesegid, 'Dp', d_certid,
                                    mp_lifesegid, m_parent_role, m_certid,
                                    life_seg_dict, life_seg_id_index, Sex.F,
                                    mother_role_certid_lifesegid_dict)
        if d_f_exists and m_f_exists:
          dp_lifesegid = \
            father_role_certid_lifesegid_dict['Dp'].get(d_certid, None)
          mp_lifesegid = \
            father_role_certid_lifesegid_dict[m_parent_role].get(m_certid,
                                                                 None)

          life_seg_id_index = \
            __enrich_lifeseg_dict__(dp_lifesegid, 'Dp', d_certid,
                                    mp_lifesegid, m_parent_role, m_certid,
                                    life_seg_dict, life_seg_id_index, Sex.M,
                                    father_role_certid_lifesegid_dict)


def persist_gt_entities(life_seg_dict):
  m_role_certid_person_dict = defaultdict(dict)
  f_role_certid_person_dict = defaultdict(dict)
  people_dict = util.get_people_dict(settings.people_file)
  for person in people_dict.itervalues():
    if person[c.I_SEX] == 'M':
      m_role_certid_person_dict[person[c.I_ROLE]][person[c.I_CERT_ID]] = person
    elif person[c.I_SEX] == 'F':
      f_role_certid_person_dict[person[c.I_ROLE]][person[c.I_CERT_ID]] = person
    else:
      m_role_certid_person_dict[person[c.I_ROLE]][person[c.I_CERT_ID]] = person
      f_role_certid_person_dict[person[c.I_ROLE]][person[c.I_CERT_ID]] = person

  gt_entities_file = '{}gt_entities.csv'.format(settings.output_home_directory)
  with open(gt_entities_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 's', 'role', 'fname', 'sname', 'addr1', 'parish',
                     'occ', 'by', 'my', 'mplace1', 'mlace2', 'certid'])

    role_list = Role.list()
    for id, life_seg in life_seg_dict.iteritems():

      fname_list = list()

      for role in role_list:
        for certid in life_seg[role]:
          if life_seg['S'] == 'M':
            if certid in m_role_certid_person_dict[role]:
              p = m_role_certid_person_dict[role][certid]
            else:
              p = f_role_certid_person_dict[role][certid]
          elif life_seg['S'] == 'F':
            if certid in f_role_certid_person_dict[role]:
              p = f_role_certid_person_dict[role][certid]
            else:
              p = m_role_certid_person_dict[role][certid]
          else:
            if certid in m_role_certid_person_dict[role]:
              p = m_role_certid_person_dict[role][certid]
            else:
              p = f_role_certid_person_dict[role][certid]

          person = [id]
          person.append(p[c.I_SEX])
          person.append(p[c.I_ROLE])
          person.append(p[c.I_FNAME])
          person.append(p[c.I_SNAME])
          person.append(p[c.I_ADDR1])
          person.append(p[c.I_PARISH])
          person.append(p[c.I_OCC])
          person.append(p[c.I_BIRTH_YEAR])
          person.append(p[c.I_MARRIAGE_YEAR])
          person.append(p[c.I_MARRIAGE_PLACE1])
          person.append(p[c.I_MARRIAGE_PLACE2])
          person.append(p[c.I_CERT_ID])
          writer.writerow(person)

      writer.writerow([])


def retrieve_ground_truth_links_kil():
  # Dictionary to hold certificate ID to certificate records dictionary
  cert_id_person_dict = {
    'Bb': {},
    'Dd': {},
    'Mm': {}
  }

  # Getting birth ID to certificate ID dictionary
  #
  with open(settings.birth_file, 'r') as data:
    reader = csv.DictReader(data)
    for record in reader:
      cert_id_person_dict['Bb'][record[c.B_ID]] = record

  # Getting marriage ID to certificate ID dictionary
  #
  with open(settings.marriage_file, 'r') as data:
    reader = csv.DictReader(data)
    for record in reader:
      cert_id_person_dict['Mm'][record[c.M_ID]] = record

  # Getting death ID to certificate ID dictionary
  #
  with open(settings.death_file, 'r') as data:
    reader = csv.DictReader(data)
    for record in reader:
      cert_id_person_dict['Dd'][record[c.D_ID]] = record

  # Dictionary to hold entities with life segment IDs as keys
  life_seg_dict = {}
  life_seg_id = 0

  # Dictionary to hold entity families with sibling IDs as keys
  sib_list_dict = defaultdict(list)

  # Dictionary to hold death id to life seg id
  birth_id_lifeseg_id_dict = {}

  # Dictionary to hold death id to life seg id
  death_id_lifeseg_id_dict = {}

  # Dictionary to hold marriage id to life seg id
  marriage_id_lifeseg_id_dict = {}

  # Retrieve life segments based on births
  with open(settings.birth_file, 'r') as data:
    reader = csv.DictReader(data)

    for record in reader:

      entity = defaultdict(set)
      life_seg_id += 1

      # Get sex
      entity['S'] = __retrieve_gender__(record['sex'])

      # Get sibling group
      entity['F'] = record['family'].strip()
      if entity['F'] != '':
        sib_list_dict[entity['F']].append(life_seg_id)

      # Get parent marriage
      entity['P'] = record['parents\' marriage'].strip()
      if entity['P'] != '':
        assert entity['P'] in cert_id_person_dict['Mm'].iterkeys()

      # Get birth ID
      entity['Bb'].add(record['ID'].strip())
      birth_id_lifeseg_id_dict[record['ID'].strip()] = life_seg_id

      # Get death ID
      death_id = record['death'].strip()
      if death_id != '':
        if death_id in cert_id_person_dict['Dd'].iterkeys():
          entity['Dd'].add(death_id)
          death_id_lifeseg_id_dict[death_id] = life_seg_id
        else:
          print 'wrong death id %s' % (death_id)

      life_seg_dict[life_seg_id] = entity

  # Retrieve life segments based on deaths
  with open(settings.death_file, 'r') as data:
    reader = csv.DictReader(data)

    for record in reader:

      death_id = record['ID'].strip()
      birth_id = record['Birth'].strip()
      if birth_id != '':
        if death_id in death_id_lifeseg_id_dict:
          tmp_life_seg_id = death_id_lifeseg_id_dict[death_id]
          entity = life_seg_dict[tmp_life_seg_id]

          for bb in entity['Bb']:
            assert bb == birth_id
        else:
          if birth_id != '-1':
            entity = life_seg_dict[birth_id_lifeseg_id_dict[birth_id]]
            entity['Dd'].add(death_id)

  # Enrich life segments based on marriages
  with open(settings.marriage_file, 'r') as data:
    reader = csv.DictReader(data)

    for record in reader:

      marriage_id = record['ID']

      # Update groom marriage ID
      g_death_id = record['groom death'].strip()
      if g_death_id != '' and g_death_id in death_id_lifeseg_id_dict:

        tmp_life_seg_id = death_id_lifeseg_id_dict[g_death_id]
        entity = life_seg_dict[tmp_life_seg_id]

        entity['Mm'].add(marriage_id)
        marriage_id_lifeseg_id_dict[marriage_id] = tmp_life_seg_id

        g_e_marriage_id = record['groom earlier marriage'].strip()
        if g_e_marriage_id != '':
          entity['Mm'].add(g_e_marriage_id)
          marriage_id_lifeseg_id_dict[g_e_marriage_id] = tmp_life_seg_id

        g_l_marriage_id = record['groom later marriage'].strip()
        if g_l_marriage_id != '':
          entity['Mm'].add(g_l_marriage_id)
          marriage_id_lifeseg_id_dict[g_l_marriage_id] = tmp_life_seg_id

      # Update bride marriage ID
      b_death_id = record['bride death'].strip()
      if b_death_id != '' and b_death_id in death_id_lifeseg_id_dict:

        tmp_life_seg_id = death_id_lifeseg_id_dict[b_death_id]
        entity = life_seg_dict[tmp_life_seg_id]

        entity['Mm'].add(marriage_id)
        marriage_id_lifeseg_id_dict[marriage_id] = tmp_life_seg_id

        b_e_marriage_id = record['bride earlier marriage'].strip()
        if b_e_marriage_id != '':
          entity['Mm'].add(b_e_marriage_id)
          marriage_id_lifeseg_id_dict[b_e_marriage_id] = tmp_life_seg_id

        b_l_marriage_id = record['bride later marriage'].strip()
        if b_l_marriage_id != '':
          entity['Mm'].add(b_l_marriage_id)
          marriage_id_lifeseg_id_dict[b_l_marriage_id] = tmp_life_seg_id

  __enumerate_indirect_gt_links__(life_seg_dict, sib_list_dict,
                                  cert_id_person_dict, life_seg_id)

  links_dict = util.enumerate_links(life_seg_dict.values())
  # persist_gt_entities(life_seg_dict)
  # for key in sorted(links_dict.iterkeys()):
  #   print key, len(links_dict[key])
  pickle.dump(links_dict, open(settings.gt_file, "wb"))
  return links_dict


def __check_MP_availability(bg_type, m_record):
  std = model.get_standard_string
  if bg_type == 'b':
    mother_available = (std(m_record[c.M_B_MN]) is not None or
                        std(m_record[c.M_B_MS])) is not None
    father_available = (std(m_record[c.M_B_FN]) is not None or
                        std(m_record[c.M_B_FS])) is not None
  else:
    mother_available = (std(m_record[c.M_G_MN]) is not None or
                        std(m_record[c.M_G_MS])) is not None
    father_available = (std(m_record[c.M_G_FN]) is not None or
                        std(m_record[c.M_G_FS])) is not None
  return mother_available, father_available


# ---------------------------------------------------------------------------
# Utility functions for data loading
# ---------------------------------------------------------------------------

def __retrieve_gender__(gender):
  male_gender = 'm'
  female_gender = 'f'
  gender = gender.strip()
  if gender not in (male_gender, female_gender):
    gender = ''
  return Sex.M if gender == male_gender else (
    Sex.F if gender == female_gender else '')


def __retrieve_pid__(person):
  return None if person is None else person[c.I_ID]


def __retrieve_date__(date_str):
  date_str = str(date_str)
  date_format_list = ['%d/%m/%Y', '%d/%m/%Y %H:%M:%S', '%Y',
                      '%d/%m/%Y %H:%M',
                      '%d/%m/%Y%H:%M:%S']

  # Dutch data set specific issue, an invalid date
  print date_str
  if 'None/' in date_str:
    date_str = date_str.replace('None/', '')
  if date_str == '31/4/1905':
    date_str = '30/4/1905'
  elif date_str == '30/2/1853':
    date_str = '28/2/1853'
  elif date_str == '29/2/1849':
    date_str = '28/2/1849'
  elif date_str == '31/6/1857':
    date_str = '30/6/1857'
  elif '22/1814' in date_str:
    date_str = '1814'

  date = None
  for date_format in date_format_list:
    try:
      date = datetime.strptime(date_str, date_format)
      break
    except ValueError:
      continue

  if date is None:
    # return None
    raise Exception('Invalid format {}'.format(date_str))
  return date


def __is_parent_link_exists__(child1, parent1_role, child2, parent2_role):
  """
  Check if the mother links and father links exists. If both sibling's mothers
  are available, mother link exists. Similarly, if both siblings fathers are
  available, father link exists.
  :param child1: Child 1
  :param parent1_role: Role of the parent of child 1
  :param child2: Child 2
  :param parent2_role: Role of the parent of child 2
  :return: A dictionary with keys as mother and father while the values are
  boolean value indicating the existence of the links.
  """
  attribute_dict = {
    'Bp': {
      'mother': [c.B_MN, c.B_MS],
      'father': [c.B_FN, c.B_FS]
    },
    'Dp': {
      'mother': [c.D_MN, c.D_MS],
      'father': [c.D_FN, c.D_FS]
    },
    'Mbp': {
      'mother': [c.M_B_MN, c.M_B_MS],
      'father': [c.M_B_FN, c.M_B_FS]
    },
    'Mgp': {
      'mother': [c.M_G_MN, c.M_G_MS],
      'father': [c.M_G_FN, c.M_G_FS]
    }
  }

  link_exists_dict = {}
  std = model.get_standard_string

  # Check if mother-mother link exists and father-father link exists
  for parent in ['mother', 'father']:
    p1_attributes = attribute_dict[parent1_role][parent]
    p2_attributes = attribute_dict[parent2_role][parent]

    p1_parent_exists = True \
      if (std(child1[p1_attributes[0]]) is not None or
          std(child1[p1_attributes[1]]) is not None) \
      else False
    p2_parent_exists = True \
      if (std(child2[p2_attributes[0]]) is not None or
          std(child2[p2_attributes[1]]) is not None) \
      else False

    link_exists_dict[parent] = p1_parent_exists and p2_parent_exists

  return link_exists_dict


def __get_birth_year__(event_year, age):
  if event_year is None or age is None or event_year == '' or age == '':
    # or (not age.isdigit()): # Removed due to floating point strings in IOS
    return None

  # Check if there are letters in the string
  if type(age) == str:
    age = age.lower()
    if age.islower():
      return None

  return int(float(event_year) - float(age))


def __get_sorted_key__(str1, str2):
  return tuple(sorted({str1, str2}))


if __name__ == '__main__':
  c.__init_constants__('ios')
  settings.__init_settings__(1.0, 1.0)

  people = enumerate_records()
  __retrieve_date__('31/4/1905')
  # gt_links_dict = retrieve_ground_truth_links()

  # c.__init_constants__('KIL')
  # settings.__init_settings__(0.0, 0.0)
  # retrieve_ground_truth_links_ios()
  # retrieve_ground_truth_links()

  # people_dict = common.get_people_dict(settings.get)
  #
  # person[role] = [append_role(cert_id, role, p['S'])
  #                 for cert_id in p[role]]

  # __retrieve_birthyear__('10/1/1861 00:00:00')

  # c.__init_constants__('KIL')
  # settings.__init_settings__(0.0, 0.0)
  # retrieve_ground_truth_links_kil()
