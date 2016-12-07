from tree import Tree, Node
from collections import defaultdict

import itertools
import math
import sys

class CFG(object):
  def __init__(self):
    self._rules = defaultdict(list)
    self.words_seen = set()

  def add_rule(self, rule):
    if rule not in self._rules[rule.base]:
      # Rule not seen yet
      self._rules[rule.base].append(rule)
    else:
      # Rule seen already, add to count
      index = self._rules[rule.base].index(rule)
      self._rules[rule.base][index].times_seen += 1

  @property
  def rules(self):
    return itertools.chain.from_iterable(self._rules.values())

  @property
  def bases(self):
    return set([rule.base for rule in self.rules])

  def train(self, filename, use_vertical_markov=True):
    with open(filename) as treeFile:
      lines = treeFile.readlines()
      for line in lines:
        tree = Tree.from_str(line)
        for node in tree.bottomup():
          if len(node.children) > 0:

            ###################################################
            # Add Vertical Markovization
            if use_vertical_markov:
              if node.parent is not None:
                node.label += "[parent=" + node.parent.label + "]"
            ###################################################
            
            rule = Rule(node.label, [child.label for child in node.children])
            self.add_rule(rule)
          else:
            # Node is a word
            self.words_seen.add(node.label)

    # Add smoothing by adding X-><unk> for every base
    for base in self.bases:
      """ Check if unary rule that does not go to unk already
      """
      unk_rule = Rule(base, ["<unk>"])
      self.add_rule(unk_rule)
      
  def cky(self, string):
    """ Finds the highest probability parse of a given string
    """
    # Create Chart
    words = string.strip().split(" ")
    n = len(words)

    chart = [[dict() for i in range(n+1)] for i in range(n+1)]
    best = [[defaultdict(float) for i in range(n+1)] for i in range(n+1)]

    # Do top row
    for i in range(1, n+1):
      word = words[i-1]
      if word not in self.words_seen:
        word = "<unk>"
      for rule in self.rules:
        if word in rule.goes_to:
          p = self.conditional_probability(rule)
          if p > best[i-1][i][rule.base]:
            best[i-1][i][rule.base] = p
            chart[i-1][i][rule.base] = [rule, i, None, None]

    # Fill in other rows
    for l in range(2, n + 1):
      for i in range(0, n-l+1):
        j = i + l
        for k in range(i+1, j):
          binary_rules = filter(lambda x: len(x.goes_to) == 2, self.rules)
          for rule in binary_rules:
            Y = rule.goes_to[0]
            Z = rule.goes_to[1]

            y_set = [base for base in chart[i][k]]
            z_set = [base for base in chart[k][j]]

            if Y in y_set and Z in z_set:
              p_prime = self.conditional_probability(rule) * best[i][k][Y] * best[k][j][Z]
              if p_prime > best[i][j][rule.base]:
                best[i][j][rule.base] = p_prime
                chart[i][j][rule.base] = [rule, i, j, k]

    """
    # print chart
    for i in range(n+1):
      for j in range(n+1):
        print(str(i) + ":" + str(j))
        for rule in chart[i][j]:
          print("\t" + str(rule))
    """

    def make_tree(chart, rule, i, j, k, use_vertical_markov=True):
      if j is not None:
        left_rule, left_i, left_j, left_k = chart[i][k][rule.goes_to[0]]
        left_tree = make_tree(chart, left_rule, left_i, left_j, left_k, use_vertical_markov)
    
        right_rule, right_i, right_j, right_k = chart[k][j][rule.goes_to[1]]
        right_tree = make_tree(chart, right_rule, right_i, right_j, right_k, use_vertical_markov)

        ###################################################
        # Remove Vertical Markovization
        if use_vertical_markov:
          return "(" + rule.base.split("[parent")[0] + " " + left_tree + " " + right_tree + ")"
        else:
          return "(" + rule.base + " " + left_tree + " " + right_tree + ")"
        ###################################################

      else:
        ###################################################
        # Remove Vertical Markovization
        if use_vertical_markov:
          return "(" + rule.base.split("[parent")[0] + " " + rule.goes_to[0] + ")"
        else:
          return "(" + rule.base + " " + rule.goes_to[0] + ")"
        ###################################################

    #print("\n" + string.strip())
    if 'TOP' in chart[0][n]:
      # Parse Exists, backtrack to find full parse
      top_rule, i, j, k = chart[0][n]['TOP']
      tree = make_tree(chart, top_rule, i, j, k)
      return tree
    else:
      return None

  def conditional_probability(self, rule):
    rulelist = self._rules[rule.base]
    total = sum(rule.times_seen  for rule in rulelist)
    return float(rule.times_seen) / total

  def __len__(self):  
    """ Returns the number of unique rules
    """
    return len(list(self.rules))

  def __str__(self):
    s = ""
    for rulelist in self._rules.values():
      total = sum(rule.times_seen for rule in rulelist)
      for rule in rulelist:
        s += rule.string_with_percent(total) + "\n"
    return s.strip()

class Rule(object):
  def __init__(self, base, goes_to):
    self.base = base
    self.goes_to = goes_to
    self.times_seen = 1

  def string_with_percent(self, total_rules):
    return self.base + " -> " + " ".join(self.goes_to) + " # " + str(round(float(self.times_seen) / total_rules, 3))

  def __str__(self):
    return self.base + " -> " + " ".join(self.goes_to) + " # " + str(self.times_seen)

  def __hash__(self):
    result = hash(self.base)
    for token in self.goes_to:
      result ^= hash(token)
    return result

  def __eq__(self, other):
    return self.base == other.base and self.goes_to == other.goes_to

def main():
  cfg = CFG()

  cfg.train("train.trees.pre.unk")
  #print cfg
  print("Length: " + str(len(cfg)))

  # Find top 5 most occurring rules
  print("\nTop 5 most occurring rules with counts: ")
  rules = list(cfg.rules)
  rules.sort(key = lambda x: x.times_seen, reverse=True)
  for rule in rules[:5]:
    print(rule)

  # Find top 5 most occurring rules by percent
  print("\nTop 5 most occurring rules with percents: ")
  rules = list(cfg.rules)
  rules.sort(key = lambda x: cfg.conditional_probability(x), reverse=True)
  for rule in rules[:5]:
    print(str(rule).split("#")[0] + "# " + str(cfg.conditional_probability(rule)))

  # Run your parser on dev.strings and save the output to dev.parses.
  # Show the output of your parser on the first five lines of dev.strings,
  # along with their log-probabilities (base 10).
  print("\nCKY Parses")
  with open("dev.strings") as devFile:
    lines = devFile.readlines()[0:5]
    for line in lines:
      prob = cfg.cky(line)
      if prob is not None:
        print("probability: " + str(math.log(prob)))

if __name__ == "__main__":
  #main()
  cfg = CFG()
  no_markovization = CFG()

  cfg.train("train.trees.pre.unk")
  no_markovization.train("train.trees.pre.unk", False)

  #print("\nCKY Parses of all lines in devFile")
  with open(sys.argv[1]) as devFile:
    lines = devFile.readlines()
    #lines = devFile.readlines()[37:38]
    for line in lines:
      tree_string = cfg.cky(line)
      if tree_string is not None:
        print(tree_string)
      else:
        tree_string_without_markovization = no_markovization.cky(line)
        if tree_string_without_markovization is not None:
          print(tree_string_without_markovization)
        else:
          print("")
