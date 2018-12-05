import sys  
import os
import statistics

def main():  
  filepath1 = sys.argv[1]
  filepath2 = sys.argv[2]

  if not os.path.isfile(filepath1):
    print("File path {} does not exist. Exiting...".format(filepath1))
    sys.exit()

  if not os.path.isfile(filepath2):
    print("File path {} does not exist. Exiting...".format(filepath2))
    sys.exit()

  numbs1 = []
  with open(filepath1) as fp:
    for line in fp:
      numbs1.append(float(line))

  numbs2 = []
  with open(filepath2) as fp:
    for line in fp:
      numbs2.append(float(line))

  diff = []
  for f1, f2 in zip(numbs1, numbs2):
    diff.append(f1 - f2)

  print('mean:', statistics.mean(diff))
  print('std:', statistics.stdev(diff))


if __name__ == '__main__':  
  main()
