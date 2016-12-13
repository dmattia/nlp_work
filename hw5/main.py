from collections import defaultdict

class Rule:
  def __init__(self, line):
    values = line.split("\t")
    self.base = values[0]

    chinese_tokens = values[1].split()
    self.chinese = [token.split("[")[0] for token in chinese_tokens]
    
    """
    en_tokens = values[2].split()
    self.english = [token.split("[")[0] for token in en_tokens]
    """
    self.english = values[2].split()

    self.prob = float(values[3])

class Translator:
  def __init__(self):
    self.rules = []
    self.words_seen = set()

  def train(self, filename):
    print("Beginning training...")

    with open(filename) as ruleFile:
      lines = ruleFile.readlines()

    for line in lines:
      rule = Rule(line)
      self.rules.append(rule)
      if len(rule.chinese) == 1:
        self.words_seen.add(rule.chinese[0])
    self.rules = [Rule(line) for line in lines]

    # Add Glue Rule
    glue_rule = Rule("PHRASE\tPHRASE[0] PHRASE[1]\tPHRASE[0] PHRASE[1]\t1")
    self.rules.append(glue_rule)

    # Add Identity Rules
    for word in self.words_seen:
      rule = Rule("PHRASE\t" + word + "\t" + word + "\t1e-10")
      self.rules.append(rule)

    print("Completed training")

  def test_file(self, filename):
    with open(filename) as testFile:
      #lines = testFile.readlines()[2:3]
      lines = testFile.readlines()

    for line in lines:
      self._test(line.strip())

  def _test(self, line):
    """ Runs Viterbi Algorithm on this line
    """
    print("testing: " + line)

    words = line.split(" ")
    n = len(words)

    chart = [[dict() for i in range(n+1)] for i in range(n+1)]
    best = [[defaultdict(float) for i in range(n+1)] for i in range(n+1)]

    # Do top row
    for i in range(1, n+1):
      word = words[i-1]
      if word not in self.words_seen:
        word = "<unk>"
      for rule in self.rules:
        if word in rule.chinese:
          p = rule.prob
          if p > best[i-1][i][rule.base]:
            best[i-1][i][rule.base] = p
            chart[i-1][i][rule.base] = [rule, i, None, None]

    # Fill in other rows
    for l in range(2, n + 1):
      for i in range(0, n-l+1):
        j = i + l
        for k in range(i+1, j):
          binary_rules = filter(lambda x: len(x.chinese) == 2, self.rules)
          for rule in binary_rules:
            Y = rule.chinese[0]
            Z = rule.chinese[1]

            y_set = [base for base in chart[i][k]]
            z_set = [base for base in chart[k][j]]

            if Y in y_set and Z in z_set:
              p_prime = rule.prob * best[i][k][Y] * best[k][j][Z]
              if p_prime > best[i][j][rule.base]:
                best[i][j][rule.base] = p_prime
                chart[i][j][rule.base] = [rule, i, j, k]

    print(chart[0][n])

    def make_tree(chart, rule, i, j, k):
      if j is not None:
        # This is a binary rule
        left_rule, left_i, left_j, left_k = chart[i][k][rule.chinese[0]]
        left_tree = make_tree(chart, left_rule, left_i, left_j, left_k)

        right_rule, right_i, right_j, right_k = chart[k][j][rule.chinese[1]]
        right_tree = make_tree(chart, right_rule, right_i, right_j, right_k)

        #return "(" + rule.base + " " + left_tree + " " + right_tree + ")"
        s = ""
        english_tokens = rule.english
        for token in english_tokens:
          if "[0]" in token:
            s += left_tree
          elif "[1]" in token:
            s += right_tree
          else:
            s += token
        return s
      else:
        # This is a unary rule. ie) PHRASE -> chinese_char
        #return "[" + rule.rawenglish[0] + "]"
        return ""
        #return "(" + rule.base + " " + rule.chinese[0] + ")"

    if 'PHRASE' in chart[0][n]:
      # Parse Exists, backtrack to find full parse
      top_rule, i, j, k = chart[0][n]['PHRASE']
      tree = make_tree(chart, top_rule, i, j, k)
      print(tree)
      return tree
    else:
      return None

def main():
  translator = Translator()
  translator.train("rules.binary")
  translator.test_file("episode3-100.zh")

if __name__ == "__main__":
  main()
