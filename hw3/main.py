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
    count, guess = max((count, tag) for (tag, count) in counter.items())
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
  print("1.1) I decided that any unknown word would be predicted to have the most common tag for all words in the training set")

  model = MarkovModel()
  model.train("train.txt")

  # Find P(t | you) for all t
  print("\n1.2)")
  word_counts = model.tokens["you"]
  total_occurrences = sum(word_counts.values())
  for (tag, count) in word_counts.items():
    print("P(" + tag + " | you) = " + str(float(count) / total_occurrences))

  print("\n1.3) Finding accuracy on test.txt")
  result = model.test("test.txt")
  print("Percent Correct: " + str(result))

  print("\n1.4) Second line of test.txt")
  with open("test.txt") as testFile:
    second_line = testFile.readlines()[1]
  tokens = second_line.split()
  words = [token.split("/")[0] for token in tokens]
  guesses = [(word, model.guessKind(word)) for word in words]
  for guess in guesses:
    print(guess[0] + "/" + guess[1], end=" ")

  print("\nWithout N words: ")
  non_n_guesses = filter(lambda x: x[1] != "N", guesses)
  for guess in non_n_guesses:
    print(guess[0] + "/" + guess[1], end=" ")
  print("")
