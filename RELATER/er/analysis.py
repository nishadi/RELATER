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
import os
from collections import defaultdict, Counter

import networkx as nx

from data import data_loader
from common import util, constants as c, stats, settings, sim
from common.enums import Links

# Dictionary for aggregating the causes for merge of invalidating
reason_dict = defaultdict(dict)

tmp_set = set()
# Dictionary to keep track of ground truth links
gt_links = {}

fp_dict = defaultdict(list)

bb_dd_tp_list = set()

current_stage = None



## ANALYSIS UTILS
#

def get_trace(node, p1, p2, is_merged, cause, cause2=''):
  """
  Method to get entity trace as a tuple.

  :param node: node
  :param p1: person 1
  :param p2: person 2
  :param is_merged: merged status
  :param cause: the cause for not merging / merging
  :return: (link, (person id tuple), merged status, ground truth status,
   cause for merging, node similarity, node atomic similarity)

  """
  link = node[c.TYPE2]
  # is_gt_link = is_ground_truth_link(p1, p2, link)
  node_key = (node[c.R1], node[c.R2])

  reason_dict[link][node_key] = cause


  # cause = cause + cause2
  # if not is_merged and not is_gt_link:
  #   return ''
  # return (link, (p1[c.I_ID], p2[c.I_ID]), is_merged, is_gt_link, cause,
  #         node[c.SIM], node[c.SIM_ATOMIC])
  return ''


def is_ground_truth_link(person_1, person_2, link):
  cert_id_1 = util.append_role(person_1[c.I_CERT_ID], person_1[c.I_ROLE],
                               person_1[c.I_SEX])
  cert_id_2 = util.append_role(person_2[c.I_CERT_ID], person_2[c.I_ROLE],
                               person_2[c.I_SEX])
  key = tuple(sorted({cert_id_1, cert_id_2}))

  if key in gt_links[link]:
    return True
  return False


def get_atomic_nodes(G, node_id):
  atomic_nodes = list()
  for neighbor_id in G.predecessors(node_id):
    if G.nodes[neighbor_id][c.TYPE1] == c.N_ATOMIC:
      atomic_nodes.append(G.nodes[neighbor_id])
  return atomic_nodes


def get_neighbor(G, node_id, type):
  for edge in list(G.in_edges(node_id, data=True)):
    edge_data = edge[2]
    if edge_data[c.TYPE1] != c.E_REAL and edge_data[c.TYPE2] == type:
      return edge[0]


def print_node_details_fn(node_id_set, G, people_dict):
  for node_id in node_id_set:
    node = G.nodes[node_id]
    p1 = people_dict[node[c.R1]]
    p2 = people_dict[node[c.R2]]
    is_gt_link = is_ground_truth_link(p1, p2, node[c.TYPE2])
    if node[c.TYPE2] == 'Bp-Mbp':
      if is_gt_link:
        print_node_details(G, people_dict, node_id, node, p1, p2)


def print_node_details(G, people_dict, node_id, node, p1, p2):
  print '--------------------------------------------------------'
  print 'Bp-Mbp node sim {}'.format(node[c.SIM])
  util.print_atomic_nodes(get_atomic_nodes(G, node_id), p1, p2)

  child_node_id = get_neighbor(G, node_id, c.CHILDREN)
  if child_node_id is not None:
    child_node = G.nodes[child_node_id]
    print '\nChild node sim {}'.format(child_node[c.SIM])
    util.print_atomic_nodes(get_atomic_nodes(G, child_node_id),
                            people_dict[child_node[c.R1]],
                            people_dict[child_node[c.R2]])

  spouse_node_id = get_neighbor(G, node_id, c.SPOUSE)
  if spouse_node_id is not None:
    spouse_node = G.nodes[spouse_node_id]
    print '\nSpouse node sim {}'.format(spouse_node[c.SIM])
    util.print_atomic_nodes(get_atomic_nodes(G, spouse_node_id),
                            people_dict[spouse_node[c.R1]],
                            people_dict[spouse_node[c.R2]])


def persist_bb_dd_tp_dict(filename):
  nx.write_gpickle(bb_dd_tp_list, filename)


def analyse_fn_fp_reasons(false_positive_link_dict, false_negative_link_dict,
                          people_dict, relationship_node_key_set, nodes):
  similarity = sim.get_pair_sim

  # Generate cert id to pid dictionary
  cert_id_role_person_id_dict = dict()
  for pid, person in people_dict.iteritems():
    cert_id_role = util.append_role(person[c.I_CERT_ID], person[c.I_ROLE],
                                    person[c.I_SEX])
    cert_id_role_person_id_dict[cert_id_role] = pid

  # Generate FP reasons

  # fp_stats_file_name = '{}fp-stats-{}-a{}-m{}-{}.csv'.format(
  #   settings.results_dir,
  #   settings.graph_definition,
  #   settings.atomic_t,
  #   settings.merge_t,
  #   settings.scenario)
  # with open(fp_stats_file_name, 'w') as fp_stats_file:
  #   stat_writer = csv.writer(fp_stats_file)
  #   stat_writer.writerow(['Stat', 'Similar Count', 'Similar %', 'None Count',
  #                         'None %', 'Avg'])
  #
  #   fp_reasons = defaultdict(Counter)
  #   for link in Links.list():
  #
  #     fp_stats_dict = defaultdict(int)
  #     fp_stats_avg_dict = defaultdict(int)
  #     fp_none_dict = defaultdict(int)
  #     for cert_id_1, cert_id_2 in false_positive_link_dict[link]:
  #       pid_1 = cert_id_role_person_id_dict[cert_id_1]
  #       pid_2 = cert_id_role_person_id_dict[cert_id_2]
  #
  #       person1 = people_dict[pid_1]
  #       person2 = people_dict[pid_2]
  #
  #       for stat, func in [(c.I_FNAME, 'JW'), (c.I_SNAME, 'JW'),
  #                          (c.I_ADDR1, 'JW'), (c.I_PARISH, 'JW'),
  #                          (c.I_BIRTH_YEAR, 'MAD'), (c.I_MARRIAGE_YEAR, 'MAD')]:
  #         if person1[stat] is None or person2[stat] is None:
  #           fp_none_dict[stat] += 1
  #         else:
  #           s = similarity((person1[stat], person2[stat]), func)
  #           fp_stats_avg_dict[stat] += s
  #           if s >= 0.9:
  #             fp_stats_dict[stat] += 1
  #
  #       # Event year
  #       year_diff = abs(person1[c.I_DATE].year - person2[c.I_DATE].year)
  #       fp_stats_avg_dict[c.I_DATE] += year_diff
  #       if year_diff <= 15:
  #         fp_stats_dict[c.I_DATE] += 1
  #
  #       if (pid_1, pid_2) in reason_dict[link]:
  #         fp_reasons[link].update([reason_dict[link][(pid_1, pid_2)]])
  #       elif (pid_2, pid_1) in reason_dict[link]:
  #         fp_reasons[link].update([reason_dict[link][(pid_2, pid_1)]])
  #       else:
  #         assert (pid_1, pid_2) not in relationship_node_key_set and \
  #                (pid_2, pid_1) not in relationship_node_key_set
  #         fp_reasons[link].update(['NG'])
  #         # print 'FP %s %s - %s' % (pid_1, pid_2, relationship_node_key_set[(pid_1, pid_2)])
  #
  #     # Persist stats
  #     link_fp_count = len(false_positive_link_dict[link])
  #     stat_writer.writerow([link, link_fp_count, 100.0, 0, 0])
  #     for stat, stat_name in \
  #         [(c.I_FNAME, 'First Name'), (c.I_SNAME, 'Last Name'),
  #          (c.I_ADDR1, 'Address 1'), (c.I_PARISH, 'Parish'),
  #          (c.I_BIRTH_YEAR, 'BY'), (c.I_MARRIAGE_YEAR, 'MY'),
  #          (c.I_DATE, 'Event Year')]:
  #       percentage = 0
  #       none_percentage = 0
  #       avg = 0
  #       if link_fp_count != 0:
  #         percentage = round(fp_stats_dict[stat] * 100.0 / link_fp_count)
  #         none_percentage = round(fp_none_dict[stat] * 100.0 / link_fp_count)
  #         if (link_fp_count - fp_none_dict[stat]) != 0:
  #           avg = fp_stats_avg_dict[stat] * 1.0 / (
  #               link_fp_count - fp_none_dict[stat])
  #       stat_writer.writerow([stat_name, fp_stats_dict[stat], percentage,
  #                             fp_none_dict[stat], none_percentage, avg])
  #
  #   # fp_reason_file_name = '{}fp-reason-{}-a{}-m{}-{}.csv'.format(
  #   #   settings.results_dir,
  #   #   settings.graph_definition,
  #   #   settings.atomic_t,
  #   #   settings.merge_t,
  #   #   settings.scenario)
  #   fp_reason_file_name = '{}fp-reason-a{}-m{}.csv'.format(
  #     settings.results_dir,
  #     settings.atomic_t,
  #     settings.merge_t)
  #   __persist_reasons__(fp_reasons, fp_reason_file_name)

  ## Generate FN reasons
  #

  fn_stats_file_name = '{}fn-stats-{}-a{}-m{}-{}.csv'.format(
    settings.results_dir,
    settings.graph_definition,
    settings.atomic_t,
    settings.merge_t,
    settings.scenario)

  fn_bp_bp_ng_name = '{}fn_bb_dd_ng_{}.csv'.format(settings.results_dir,
                                                   settings.scenario)

  with open(fn_stats_file_name, 'w') as fn_stats_file:
    stat_writer = csv.writer(fn_stats_file)
    stat_writer.writerow(['Stat', 'Similar Count', 'Similar %', 'None Count',
                          'None %', 'Avg'])

    with open(fn_bp_bp_ng_name, 'w') as ng_file:
      ng_writer = csv.writer(ng_file)

      both_entity_count = 0
      one_entity_count = 0
      both_record_count = 0
      total = 0

      fn_reasons = defaultdict(Counter)
      for link in Links.list():

        fn_stats_dict = defaultdict(int)
        fn_stats_avg_dict = defaultdict(int)
        fn_none_dict = defaultdict(int)
        for cert_id_1, cert_id_2 in false_negative_link_dict[link]:
          pid_1 = cert_id_role_person_id_dict[cert_id_1]
          pid_2 = cert_id_role_person_id_dict[cert_id_2]

          person1 = people_dict[pid_1]
          person2 = people_dict[pid_2]

          for stat, func in [(c.I_FNAME, 'JW'), (c.I_SNAME, 'JW'),
                             (c.I_ADDR1, 'JW'), (c.I_PARISH, 'JW'),
                             (c.I_BIRTH_YEAR, 'MAD'), (c.I_MARRIAGE_YEAR, 'MAD')]:
            if person1[stat] is None or person2[stat] is None:
              fn_none_dict[stat] += 1
            else:
              s = similarity((person1[stat], person2[stat]), func)
              fn_stats_avg_dict[stat] += s
              if s >= 0.9:
                fn_stats_dict[stat] += 1

          # Event year
          year_diff = abs(person1[c.I_DATE].year - person2[c.I_DATE].year)
          fn_stats_avg_dict[c.I_DATE] += year_diff
          if year_diff <= 15:
            fn_stats_dict[c.I_DATE] += 1

          if (pid_1, pid_2) in reason_dict[link]:
            fn_reasons[link].update([reason_dict[link][(pid_1, pid_2)]])
          elif (pid_2, pid_1) in reason_dict[link]:
            fn_reasons[link].update([reason_dict[link][(pid_2, pid_1)]])
          else:
            assert (pid_1, pid_2) not in relationship_node_key_set and \
                   (pid_2, pid_1) not in relationship_node_key_set
            fn_reasons[link].update(['NG'])
            if link == 'Bb-Dd':
              total += 1
              if person1[c.I_ENTITY_ID] is not None and person2[c.I_ENTITY_ID] \
                  is not None:
                both_entity_count += 1
              elif person1[c.I_ENTITY_ID] is None and person2[c.I_ENTITY_ID] \
                  is None:
                both_record_count += 1
              else:
                one_entity_count += 1
              print_attributes(person1, person2, ng_writer)
            # print 'FN %s %s - %s' % (pid_1, pid_2, relationship_node_key_set[
            #   (pid_1, pid_2)])

        # Persist stats
        link_fn_count = len(false_negative_link_dict[link])
        stat_writer.writerow([link, link_fn_count, 100.0])
        for stat, stat_name in \
            [(c.I_FNAME, 'First Name'), (c.I_SNAME, 'Last Name'),
             (c.I_ADDR1, 'Address 1'), (c.I_PARISH, 'Parish'),
             (c.I_BIRTH_YEAR, 'BY'), (c.I_MARRIAGE_YEAR, 'MY'),
             (c.I_DATE, 'Event Year')]:
          percentage = 0
          none_percentage = 0
          avg = 0
          if link_fn_count != 0:
            percentage = round(fn_stats_dict[stat] * 100.0 / link_fn_count)
            none_percentage = round(fn_none_dict[stat] * 100.0 / link_fn_count)
            if (link_fn_count - fn_none_dict[stat]) != 0:
              avg = fn_stats_avg_dict[stat] * 1.0 / (
                  link_fn_count - fn_none_dict[stat])
          stat_writer.writerow([stat_name, fn_stats_dict[stat], percentage,
                                fn_none_dict[stat], none_percentage, avg])

  print 'Bp-Bp NG both entities %s' % both_entity_count
  print 'Bp-Bp NG both non-entities %s' % both_record_count
  print 'Bp-Bp NG one entity %s' % one_entity_count
  print 'Bp-Bp NG tot %s' % total

  # fn_reason_file_name = '{}fn-reason-{}-a{}-m{}-{}.csv'.format(
  #   settings.results_dir,
  #   settings.graph_definition,
  #   settings.atomic_t,
  #   settings.merge_t,
  #   settings.scenario)

  fn_reason_file_name = '{}fn-reason-a{}-m{}.csv'.format(
    settings.results_dir,
    settings.atomic_t,
    settings.merge_t)

  __persist_reasons__(fn_reasons, fn_reason_file_name, false_negative_link_dict)


def __persist_reasons__(counter_dict, file_name, false_negative_link_dict):
  file_exists = os.path.isfile(file_name)

  column_list = ['NEW', 'NG', 'LSIM', 'SIN', 'S_B',
                 'RER-SELF-NE', 'RER-SELF-E', 'RER-F-E', 'RER-M-E', 'RER-M-NE',
                 'RP_S', 'RP_G',
                 'RE2_BY', 'RE2_MY', 'RE2_BMY','RE2_SDd', 'RE2_SBb',
                 'RE1_BMY', 'RE1_MY', 'RE1_BMY', 'RE1_BY', 'RE1_SBb', 'RE1_SDd']
  with open(file_name, mode='a') as f:
    w = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    if not file_exists:
      w.writerow(['link', 'scenario', 'total'] + column_list)
    scenario = settings.scenario
    for link, link_counter in counter_dict.iteritems():
      row = [link, scenario, len(false_negative_link_dict[link])]
      for reason in column_list:
        if reason in link_counter.iterkeys():
          # percentage = round((link_counter.get(reason) *100.0 /
          #                     len(false_negative_link_dict[link])), 2)
          # str = '%s (%s%%)' % (link_counter.get(reason), percentage)
          row.append(link_counter.get(reason))
        else:
          row.append(0)
      w.writerow(row)


def analyse_graphs(g):
  atomic_nodes = 0
  rel_nodes = 0
  for node_id, node in g.G.nodes(data=True):

    if node[c.TYPE1] == c.N_RELTV:
      rel_nodes += 1
    elif node[c.TYPE1] == c.N_ATOMIC:
      atomic_nodes += 1

  print 'Atomic nodes : {}'.format(atomic_nodes)
  print 'Rel nodes : {}'.format(rel_nodes)


def print_attributes(p1, p2, csv_writer=None):
  for i in range(6, 17):

    if i == 12:
      continue

    attr_name = util.get_attribute_name(i)

    function = 'MAD' if i in [13, 14] else 'JW'
    if p1[i] is None or p2[i] is None:
      sim_val = 0.0
    else:
      sim_val = sim.get_pair_sim((p1[i], p2[i]), function)

    str1 = '' if p1[i] is None else p1[i]
    str2 = '' if p2[i] is None else p2[i]

    attr_name_padd_str = ' ' * abs(8 - len(attr_name))
    attr_val1_padd_str = ' ' * abs(15 - len(str(str1)))
    attr_val2_padd_str = ' ' * abs(15 - len(str(str2)))

    if csv_writer is None:
      line_str = '   %s%s | %s%s | %s%s | %.2f' % (attr_name_padd_str,
                                                   attr_name,
                                                   attr_val1_padd_str,
                                                   str1,
                                                   attr_val2_padd_str,
                                                   str2, sim_val)
      print (line_str)
    else:
      csv_writer.writerow([attr_name, str1, str2, sim_val])

  if csv_writer is not None:
    csv_writer.writerow(['pid', p1[c.I_PID], p2[c.I_PID], '-'])
    csv_writer.writerow(['cert id', p1[c.I_CERT_ID], p2[c.I_CERT_ID], '-'])
    csv_writer.writerow([])