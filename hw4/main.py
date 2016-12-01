from tree import Tree
from collections import defaultdict

import itertools

class CFG(object):
  def __init__(self):
    self._rules = defaultdict(list)

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

  def __eq__(self, other):
    return self.base == other.base and self.goes_to == other.goes_to

if __name__ == "__main__":
  cfg = CFG()

  with open("train.trees.pre.unk") as treeFile:
    lines = treeFile.readlines()
    for line in lines:
      tree = Tree.from_str(line)
      for node in tree.bottomup():
        if len(node.children) > 0:
          rule = Rule(node.label, [child.label for child in node.children])
          cfg.add_rule(rule)

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
