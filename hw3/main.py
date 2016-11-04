from collections import defaultdict, Counter

class MarkovModel:
  def __init__(self):
    self.tokens = defaultdict(list)
    self.mostCommonGuess = ""

    self.guessDict = {}

  def train(self, filename):
    with open(filename) as inputFile:
      words = inputFile.read().split()
      tokens = list(map(Token, words))
      for token in tokens:
        self.tokens[token.word].append(token.part_of_speech)

    # Find the most common pos to use for unk 
    parts_of_speech = list(map(lambda x: x.part_of_speech, tokens))
    counter = Counter(parts_of_speech)
    count, guess = max((count, word) for (word, count) in counter.items())
    self.mostCommonGuess = guess

  def test(self, filename):
    with open(filename) as devFile:
      words = devFile.read().split()
      tokens = list(map(Token, words))

    correct_count = 0
    for token in tokens:
      guess = self.guessPartOfSpeech(token.word) 
      if guess == token.part_of_speech:
        correct_count += 1

    return float(correct_count) / len(tokens)

  def guessPartOfSpeech(self, word):
    if word not in self.guessDict:
      parts_of_speech = self.tokens[word]
      if len(parts_of_speech) == 0:
        # This word is unk
        return self.mostCommonGuess
      counter = Counter(parts_of_speech)
      count, guess = max((count, word) for (word, count) in counter.items())
      self.guessDict[word] = guess
    return self.guessDict[word]

class Token:
  def __init__(self, wordPair):
    self.word = wordPair.split("/")[0]
    self.part_of_speech = wordPair.split("/")[1]

if __name__ == "__main__":
  model = MarkovModel()
  print("Training")
  model.train("train.txt")

  print("Testing")
  result = model.test("test.txt")
  print("Percent Correct: " + str(result))
