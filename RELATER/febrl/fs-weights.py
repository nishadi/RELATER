import math
import random
import sys
import time


def calculate_agree_disagree_weights(enc_data_dict, plain_data_dict,
                                     enc_unique_val_dict, plain_unique_val_dict,
                                     consider_attr_num_list, sample_size=None):
  start_time = time.time()

  u_prob_approx = False

  enc_rec_id_set = set(enc_data_dict.keys())
  plain_rec_id_set = set(plain_data_dict.keys())

  if (sample_size != None):
    random.seed(random.randint(0, sys.maxint))

    enc_rec_id_sample = set(random.sample(enc_rec_id_set, sample_size))
    plain_rec_id_sample = set(random.sample(plain_rec_id_set, sample_size))

    common_rec_id_set = enc_rec_id_sample.intersection(plain_rec_id_sample)
    common_rec_perc = 100 * float(len(common_rec_id_set)) / len(
      enc_rec_id_sample)

  else:
    common_rec_id_set = enc_rec_id_set.intersection(plain_rec_id_set)
    common_rec_perc = 100 * float(len(common_rec_id_set)) / len(enc_rec_id_set)

  print 'Number of records common to both datasets: %d (%.2f%%)' % (
  len(common_rec_id_set), common_rec_perc)
  print

  m_prob_dict = {}
  u_prob_dict = {}

  m_count_dict = {}
  u_count_dict = {}

  attr_weight_dict = {}

  for common_rec_id in common_rec_id_set:

    enc_rec_val_list = enc_data_dict[common_rec_id]
    plain_rec_val_list = plain_data_dict[common_rec_id]

    for attr_num in consider_attr_num_list:

      if (enc_rec_val_list[attr_num] == plain_rec_val_list[attr_num]):
        m_count_dict[attr_num] = m_count_dict.get(attr_num, 0) + 1

  num_common_rec = len(common_rec_id_set)

  for attr_num, attr_count in m_count_dict.iteritems():
    m_prob_dict[attr_num] = float(attr_count) / num_common_rec

  print 'm-probability values', m_prob_dict

  time_used = time.time() - start_time
  print '  Calculated m-probability values in %d sec' % time_used

  # Calculate u-probabilities
  #
  for attr_index in consider_attr_num_list:

    enc_val_dict = enc_unique_val_dict[attr_index]
    plain_val_dict = plain_unique_val_dict[attr_index]

    if (u_prob_approx):

      num_unique_val = len(enc_val_dict.keys())
      u_porb = float(1) / num_unique_val

      u_prob_dict[attr_index] = u_porb

    else:

      for attr_val in enc_val_dict.keys():

        if (attr_val in plain_val_dict):
          enc_id_set = enc_val_dict[attr_val]
          plain_id_set = plain_val_dict[attr_val]

          comm_id_set = enc_id_set.intersection(plain_id_set)

          num_enc_id = len(enc_id_set)
          num_plain_id = len(plain_id_set)
          num_comm_id = len(comm_id_set)

          non_match_pairs = (num_enc_id * num_plain_id) - num_comm_id

          u_count_dict[attr_index] = u_count_dict.get(attr_index,
                                                      0) + non_match_pairs

  total_non_match_pairs = (len(enc_rec_id_set) * len(plain_rec_id_set)) - len(
    common_rec_id_set)
  total_pairs = len(enc_rec_id_set) * len(plain_rec_id_set)

  if (u_prob_approx == False):
    for attr_num, attr_count in u_count_dict.iteritems():
      joint_prob = float(attr_count) / total_pairs
      marginal_prob = float(total_non_match_pairs) / total_pairs

      cond_prob = joint_prob / marginal_prob
      u_prob_dict[attr_num] = cond_prob
      # u_prob_dict[attr_num] = numpy.mean(attr_prob_list)

  print
  print 'u-probability values', u_prob_dict

  time_used = time.time() - start_time
  print '  Calculated u-probability values in %d sec' % time_used

  print

  for attr_index in consider_attr_num_list:

    if (attr_index in m_prob_dict):
      attr_m_prob = m_prob_dict[attr_index]
    else:
      attr_m_prob = 0.0

    if (attr_index in u_prob_dict):
      attr_u_porb = u_prob_dict[attr_index]
    else:
      attr_u_porb = 0.0

    # Prevent division by 0 errors
    if (attr_u_porb == 0.0):
      agree_val = 0.0
    else:
      agree_val = float(attr_m_prob) / attr_u_porb

    # Prevent division by 0 errors
    if (attr_u_porb == 1.0):
      disagree_val = 0.0
    else:
      disagree_val = float(1 - attr_m_prob) / (1 - attr_u_porb)

    if (agree_val == 0.0):
      attr_agree_weight = 0.0
    else:
      attr_agree_weight = math.log(agree_val, 10)

    if (disagree_val == 0.0):
      attr_disagree_weight = 0.0
    else:
      attr_disagree_weight = math.log(disagree_val, 10)

    attr_weight_dict[attr_index] = [attr_agree_weight, attr_disagree_weight]

  return attr_weight_dict

