"""Module with various smaller functions used in other modules.
"""

# =============================================================================
# Import necessary modules (Febrl modules first, then Python standard modules)

import logging
import os
import sets
import sys
import types

# =============================================================================

def check_is_not_none(variable, value): 
  """Check if the given value is not None, if it is raise an exception.
  """

  if (value == None):
    logging.exception('Value of "%s" is None' % (variable))
    raise Exception

# -----------------------------------------------------------------------------

def check_is_string(variable, value): 
  """Check if the type of the given value is a string, if not raise an
     exception.
  """

  if (not isinstance(value, str)):
    logging.exception('Value of "%s" is not a string: %s (%s)' % \
                      (variable, str(value), type(value)))
    raise Exception

# -----------------------------------------------------------------------------

def check_is_number(variable, value): 
  """Check if the type of the given value is an integer or a floating point
     number, if not raise an exception.
  """

  if ((not isinstance(value, int)) and (not isinstance(value, float))):
    logging.exception('Value of "%s" is not a number: %s (%s)' % \
                      (variable, str(value), type(value)))
    raise Exception

# -----------------------------------------------------------------------------

def check_is_positive(variable, value): 
  """Check if the type of the given value is an integer or a floating point
     number, and is positive, if not raise an exception.
  """

  if ((not isinstance(value, int)) and (not isinstance(value, float)) or \
      (value <= 0.0)):
    logging.exception('Value of "%s" is not a positive number: ' % \
                      (variable) + '%s (%s)' % (str(value), type(value)))
    raise Exception

# -----------------------------------------------------------------------------

def check_is_not_negative(variable, value): 
  """Check if the type of the given value is an integer or a floating point
     number, and is not negative, if not raise an exception.
  """

  if ((not isinstance(value, int)) and (not isinstance(value, float)) or \
      (value < 0.0)):
    logging.exception('Value of "%s" is not a number or it is a ' % \
                      (variable) + 'negative number: %s (%s)' % \
                      (str(value), type(value)))
    raise Exception

# -----------------------------------------------------------------------------

def check_is_normalised(variable, value): 
  """Check if the type of the given value is an integer or a floating point
     number between 0 and 1 (including), if not raise an exception.
  """

  if ((not isinstance(value, int)) and (not isinstance(value, float)) or \
      (value < 0.0) or (value > 1.0)):
    logging.exception('Value of "%s" is not a normalised number ' % \
                      (variable) + '(between 0.0 and 1.0): %s (%s)' % \
                      (str(value), type(value)))
    raise Exception

# -----------------------------------------------------------------------------

def check_is_percentage(variable, value): 
  """Check if the type of the given value is an integer or a floating point
     number between 0 and 100 (including), if not raise an exception.
  """

  if ((not isinstance(value, int)) and (not isinstance(value, float)) or \
      (value < 0.0) or (value > 100.0)):
    logging.exception('Value of "%s" is not a percentage number ' % \
                      (variable) + '(between 0.0 and 100.0): %s (%s)' % \
                      (str(value), type(value)))
    raise Exception

# -----------------------------------------------------------------------------

def check_is_integer(variable, value): 
  """Check if the type of the given value is an integer, if not raise an
     exception.
  """

  if (not isinstance(value, int)):
    logging.exception('Value of "%s" is not an integer: %s (%s)' % \
                      (variable, str(value), type(value)))
    raise Exception

# -----------------------------------------------------------------------------

def check_is_float(variable, value): 
  """Check if the type of the given value is a floating point number, if not
     raise an exception.
  """

  if (not isinstance(value, float)):
    logging.exception('Value of "%s" is not a floating point ' % (variable) + \
                      'number: %s (%s)' % (str(value),type(value)))
    raise Exception

# -----------------------------------------------------------------------------

def check_is_dictionary(variable, value): 
  """Check if the type of the given value is a dictionary, if not raise an
     exception.
  """

  if (not isinstance(value, dict)):
    logging.exception('Value of "%s" is not a dictionary: %s' % \
                      (variable, type(value)))
    raise Exception

# -----------------------------------------------------------------------------

def check_is_list(variable, value): 
  """Check if the type of the given value is a list, if not raise an exception.
  """

  if (not isinstance(value, list)):
    logging.exception('Value of "%s" is not a list: %s' % \
                      (variable, type(value)))
    raise Exception

# -----------------------------------------------------------------------------

def check_is_set(variable, value): 
  """Check if the type of the given value is a set, if not raise an exception.

     Have to check both set module type as well as built in set type.
  """

  if ((not isinstance(value, sets.Set)) and (not isinstance(value, set))):
    logging.exception('Value of "%s" is not a set: %s' % \
                      (variable, type(value)))
    raise Exception

# -----------------------------------------------------------------------------

def check_is_tuple(variable, value): 
  """Check if the type of the given value is a tuple, if not raise an
     exception.
  """

  if (not isinstance(value, tuple)):
    logging.exception('Value of "%s" is not a tuple: %s' % \
                      (variable, type(value)))
    raise Exception

# -----------------------------------------------------------------------------

def check_is_flag(variable, value): 
  """Check if the given value is either True or False, if not raise an
     exception.
  """

  if (value not in [True, False]):
    logging.exception('Value of "%s" is not True or False: %s' % \
                      (variable, str(value)))
    raise Exception

# -----------------------------------------------------------------------------

def check_is_function_or_method(variable, value): 
  """Check if the given value is either a function or an instance method, if
     not raise an exception.
  """

  if (type(value) not in [types.FunctionType, types.MethodType]):
    logging.exception('%s is not a function or method: %s' % \
                      (str(variable), type(value)))
    raise Exception

# =============================================================================

def time_string(seconds):
  """Function which returns a time in hours, minutes or seconds according to
     the value of the argument 'seconds':

     - in milliseconds if less than one second
     - in seconds if the value is less than 60 seconds
     - in minutes and seconds if the value is less than one hour
     - in hours and minutes otherwise

     Returns a string.
  """

  if (seconds < 0.01):  # Less than 10 milli seconds
    stringtime = '%.2f milli sec' % (seconds*1000)
  elif (seconds < 1.0):
    stringtime = '%i milli sec' % (int(seconds*1000))
  elif (seconds < 30):
    stringtime = '%.2f sec' % (seconds)
  elif (seconds < 60):
    stringtime = '%i sec' % (int(seconds))
  elif (seconds < 3600):
    min = int(seconds / 60)
    sec = seconds - min*60
    stringtime = '%i min and %i sec' % (min, sec)
  else:
    hrs = int(seconds / 3600)
    min = int((seconds - hrs *3600) / 60)
    stringtime = '%i hrs and %i min' % (hrs, min)

  return stringtime

# =============================================================================

def get_memory_usage():
  """Function which return the current memory usage as a string.
     Currently only seems to work on Linux!

     Returns None if memory usage cannot be calculated.
  """

  try:
    ps = open('/proc/%d/status' % os.getpid())
    vs = ps.read()
    ps.close()
  except:
    return None  # Likely not on a Linux machine

  if (('VmSize:' in vs) and ('VmRSS:' in vs)):
    memo_index = vs.index('VmSize:')
    resi_index = vs.index('VmRSS:')
  else:
    return None  # Likely not on a Linux machine

  memo_list = vs[memo_index:].split(None,3)[:3]
  resi_list = vs[resi_index:].split(None,3)[:3]

  if ((int(memo_list[1]) > 10240) and (memo_list[2] in ['kB','KB'])):
    memo_list[1] = str(int(memo_list[1])/1024)
    memo_list[2] = 'MB'

  if ((int(resi_list[1]) > 10240) and (resi_list[2] in ['kB','KB'])):
    resi_list[1] = str(int(resi_list[1])/1024)
    resi_list[2] = 'MB'

  if ((len(memo_list) < 3) or (len(resi_list) < 3)):
    return None  # Invalid format in status information

  return 'Memory usage: %s %s (resident: %s %s)' % \
         (memo_list[1], memo_list[2], resi_list[1], resi_list[2])

# =============================================================================

def get_memory_usage_val():
  """Function which return the current memory usage as a value that corresponds
     to the amount of memory used in megabytes.
     Currently only seems to work on Linux!

     Returns -1 if memory usage cannot be calculated.
  """

  try:
    ps = open('/proc/%d/status' % os.getpid())
    vs = ps.read()
    ps.close()
  except:
    return None  # Likely not on a Linux machine

  if (('VmSize:' in vs) and ('VmRSS:' in vs)):
    memo_index = vs.index('VmSize:')
  else:
    return None  # Likely not on a Linux machine

  memo_list = vs[memo_index:].split(None,3)[:3]

  if ((int(memo_list[1]) > 10240) and (memo_list[2] in ['kB','KB'])):
    memo_use = float(memo_list[1])/1024.0
  else:
    memo_use = float(memo_list[1])

  if (len(memo_list) < 3):
    memo_use =  -1  # Invalid format in status information

  return memo_use

# =============================================================================

def check_memory_use(max_mbyte):
  """Stop the program if it uses more than the maximum specified amount of
     memory in Megabytes.
  """

  if (get_memory_usage_val() > max_mbyte):
    print
    print '*** Warning: Program uses more than %d MBytes: %d MBytes ***' % \
          (max_mbyte, get_memory_usage_val())
    print
    sys.exit()  # Stop due to memory overuse

# =============================================================================

def str_vector(vec, num_digits=4, keep_int=True):
  """Function to create a string representation of a vector containing numbers
     rounded to a specified number of digits after the comma (default is 4).

     If 'keep_int' s set to True (default), integer values will printed as
     integers, otherwise a floats.
  """

  vec_str = '['
  for x in vec:

    if (x == int(x)):
      x = int(x)  # Make it an integer

    if ((keep_int == True) and (isinstance(x, int))):
      vec_str = vec_str + '%d, ' % (x)
    else:
      val_str = '%f' % (x)
      vec_str += val_str[:val_str.index('.')+num_digits+1]+', '

  return vec_str[:-2]+']'

# =============================================================================
