from collections import defaultdict, Counter

class MarkovModel:
  def __init__(self):
    self.tokens = defaultdict(lambda : defaultdict(int))
    self.mostCommonGuess = ""

    self.guessDict = {}

  def train(self, filename):
    with open(filename) as inputFile:
      words = inputFile.read().split()
      tokens = list(map(Token, words))
      for token in tokens:
        self.tokens[token.word][token.kind] += 1

    # Find the most common pos to use for unk 
    kinds = list(map(lambda x: x.kind, tokens))
    counter = Counter(kinds)
    count, guess = max((count, word) for (word, count) in counter.items())
    self.mostCommonGuess = guess

  def test(self, filename):
    with open(filename) as devFile:
      words = devFile.read().split()
      tokens = list(map(Token, words))

    correct_count = 0
    for token in tokens:
      guess = self.guessKind(token.word) 
      if guess == token.kind:
        correct_count += 1

    return float(correct_count) / len(tokens)

  def guessKind(self, word):
    if word not in self.guessDict:
      kind_dict = self.tokens[word]
      if len(kind_dict) == 0:
        # This word is unk
        return self.mostCommonGuess
      count, guess = max((count, word) for (word, count) in kind_dict.items())
      self.guessDict[word] = guess
    return self.guessDict[word]

class Token:
  def __init__(self, wordPair):
    self.word = wordPair.split("/")[0]
    self.kind = wordPair.split("/")[1]

if __name__ == "__main__":
  model = MarkovModel()
  print("Training")
  model.train("train.txt")

  print("Testing")
  result = model.test("test.txt")
  print("Percent Correct: " + str(result))
