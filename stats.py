import sys  
import os
import statistics

def main():  
  filepath = sys.argv[1]

  if not os.path.isfile(filepath):
    print("File path {} does not exist. Exiting...".format(filepath))
    sys.exit()

  numbs = []
  with open(filepath) as fp:
    cnt = 0
    for line in fp:
      numbs.append(float(line))

  print('mean:', statistics.mean(numbs))
  print('std:', statistics.stdev(numbs))

if __name__ == '__main__':  
  main()
