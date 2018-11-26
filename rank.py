import sys  
import os
import statistics

def main():  
  filepath = sys.argv[1]
  increase = len(sys.argv) > 2 and sys.argv[2] == '+'

  if not os.path.isfile(filepath):
    print("File path {} does not exist. Exiting...".format(filepath1))
    sys.exit()

  numbs = []
  with open(filepath) as fp:
    for line in fp:
      numbs.append(float(line))

  ret = sorted(range(len(numbs)), key=lambda i: numbs[i], reverse=increase)[-20:]
  print(ret)
  for n in ret:
    print(n, numbs[n])


if __name__ == '__main__':  
  main()
