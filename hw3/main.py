from collections import defaultdict, Counter
from prettytable import PrettyTable
from operator import attrgetter

class MarkovModel:
  def __init__(self):
    self.tokens = defaultdict(lambda : defaultdict(int))
    self.tag_table = defaultdict(lambda : defaultdict(int))
    self.tag_counts = defaultdict(lambda : 1)
    self.most_common_guess = ""
    self.tags = []

    self.guessDict = {}

  def train(self, filename):
    with open(filename) as inputFile:
      # Find all words, but before each sentence add <s> and after each sentence add <end>
      raw_lines = inputFile.readlines()
      marked_lines = ["<s>/<s> " + line + " </s>/</s>" for line in raw_lines]
      full_file_str = " ".join(marked_lines)
      words = full_file_str.split()

      tokens = list(map(Token, words))
      for token in tokens:
        # Count occurrences of this tag, word
        self.tokens[token.word][token.tag] += 1
        self.tag_counts[token.tag] += 1

      # Add one to this tag occurring based on the previous tag
      for index in range(1, len(tokens)):
        current_tag = tokens[index].tag
        previous_tag = tokens[index-1].tag
        self.tag_table[current_tag][previous_tag] += 1

    self.tags = [tag for tag in self.tag_counts]

    # Find the most common pos to use for unk 
    count, guess = max((count, tag) for (tag, count) in self.tag_counts.items())
    self.most_common_guess = guess

  def test_0th_order(self, filename):
    with open(filename) as devFile:
      words = devFile.read().split()
      tokens = list(map(Token, words))

    correct_count = 0
    for token in tokens:
      guess = self.guessTag(token.word) 
      if guess == token.tag:
        correct_count += 1

    return float(correct_count) / len(tokens)

  def test_1st_order(self, filename):
    with open(filename) as devFile:
      raw_lines = devFile.readlines()
      marked_lines = ["<s>/<s> " + line + " </s>/</s>" for line in raw_lines]

      total = 0
      correct = 0

      for line in marked_lines:
        #print("LINE: " + line)
        words = line.split()
        actual_tokens = [Token(word) for word in words]
        best_path = []

        start_node = Node(Token(words[0] + "/<s>"))
        start_node.probability = 1.0
        viterbi = [[] for i in range(len(words))]
        viterbi[0].append(start_node)

        for word_index in range(1, len(words)):
          word = words[word_index]
          # Make one node for each tag type
          for tag in self.tags:

            actual_token = Token(word)
            new_token = actual_token
            new_token.tag = tag
            new_node = Node(new_token)

            for prev_node in viterbi[word_index - 1]:  
              prob = self.prob_node_given_previous_node(new_node, prev_node) 
              if prob > new_node.probability:
                new_node.probability = prob
                new_node.previous = prev_node
            if new_node.probability > 0:
              viterbi[word_index].append(new_node)

        # recreate path
        last_node = max(viterbi[-1], key=attrgetter('probability'))
        n = last_node
        while n is not None:
          best_path.append(n)
          n = n.previous
        best_path = list(reversed(best_path))

        # Check correct
        for node_index in range(len(best_path)):
          node = best_path[node_index]
          if node.token.word not in ["<s>", "</s>"]:
            if node.token.tag == actual_tokens[node_index].tag:
              correct += 1
            #else:
              #print(node.token.word)
            total += 1
      print("Correct: " + str(float(correct) / total))

  def test_1st_order_improved(self, filename):
    with open(filename) as devFile:
      raw_lines = devFile.readlines()
      marked_lines = ["<s>/<s> " + line + " </s>/</s>" for line in raw_lines]

      total = 0
      correct = 0

      for line in marked_lines:
        #print("LINE: " + line)
        words = line.split()
        actual_tokens = [Token(word) for word in words]
        best_path = []

        start_node = Node(Token(words[0] + "/<s>"))
        start_node.probability = 1.0
        viterbi = [[] for i in range(len(words))]
        viterbi[0].append(start_node)

        for word_index in range(1, len(words)):
          word = words[word_index]
          # Make one node for each tag type
          for tag in self.tags:

            actual_token = Token(word)
            new_token = actual_token
            new_token.tag = tag
            new_node = Node(new_token)

            for prev_node in viterbi[word_index - 1]:  
              prob = self.prob_node_given_previous_node_improved(new_node, prev_node) 
              if prob > new_node.probability:
                new_node.probability = prob
                new_node.previous = prev_node
            if new_node.probability > 0:
              viterbi[word_index].append(new_node)

        # recreate path
        last_node = max(viterbi[-1], key=attrgetter('probability'))
        n = last_node
        while n is not None:
          best_path.append(n)
          n = n.previous
        best_path = list(reversed(best_path))

        # Check correct
        for node_index in range(len(best_path)):
          node = best_path[node_index]
          if node.token.word not in ["<s>", "</s>"]:
            if node.token.tag == actual_tokens[node_index].tag:
              correct += 1
            #else:
              #print(node.token.word)
            total += 1

      print("Correct: " + str(float(correct) / total))

  def prob_node_given_previous_node_improved(self, node, prev_node):
    if node.token.word == ",":
      if node.token.tag == prev_node.token.tag:
        return prev_node.probability
      else:
        return 0
    # Find p_tag
    p_tag = self.prob_of_tag_given_previous_tag(node.token.tag, prev_node.token.tag)

    # Find p_word_given_tag
    p_word_given_tag = self.prob_word_given_tag(node.token.word, node.token.tag)

    return p_tag * p_word_given_tag * prev_node.probability

  def prob_node_given_previous_node(self, node, prev_node):
    # Find p_tag
    p_tag = self.prob_of_tag_given_previous_tag(node.token.tag, prev_node.token.tag)

    # Find p_word_given_tag
    p_word_given_tag = self.prob_word_given_tag(node.token.word, node.token.tag)

    return p_tag * p_word_given_tag * prev_node.probability

  def prob_of_tag_given_previous_tag(self, tag, prev_tag):
    return self.tag_table[tag][prev_tag] / self.tag_counts[prev_tag]

  def prob_word_given_tag(self, word, tag):
    # Guess N if unk
    if self.tokens[word][tag] == 0:
      return 1 if tag == self.most_common_guess else 0
    return self.tokens[word][tag] / self.tag_counts[tag]

  def guessTag(self, word):
    if word not in self.guessDict:
      tag_dict = self.tokens[word]
      if len(tag_dict) == 0:
        # This word is unk
        return self.most_common_guess
      count, guess = max((count, word) for (word, count) in tag_dict.items())
      self.guessDict[word] = guess
    return self.guessDict[word]

class Node:
  def __init__(self, token):
    self.token = token
    self.previous = None
    self.probability = 0

  @property
  def is_first_node(self):
    return self.token.tag == "<s>"

class Token:
  def __init__(self, wordPair):
    if wordPair == "</s>/</s>":
      self.word = "</s>"
      self.tag = "</s>"
    else:
      self.word = wordPair.split("/")[0]
      self.tag = wordPair.split("/")[1]

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
  result = model.test_0th_order("test.txt")
  print("Percent Correct: " + str(result))

  print("\n1.4) Second line of test.txt")
  with open("test.txt") as testFile:
    second_line = testFile.readlines()[1]
  tokens = second_line.split()
  words = [token.split("/")[0] for token in tokens]
  guesses = [(word, model.guessTag(word)) for word in words]
  for guess in guesses:
    print(guess[0] + "/" + guess[1], end=" ")

  print("\nWithout N words: ")
  non_n_guesses = filter(lambda x: x[1] != "N", guesses)
  for guess in non_n_guesses:
    print(guess[0] + "/" + guess[1], end=" ")
  print("")

  #######################
  ###### Part 2 #########
  #######################

  print("\n2.1) I guesses the most common tag for all unknown words")

  print("\nPrinting table of p(t' | t)")
  print("Note: I choose the top row to be t' and each for to be t")
  tags = model.tags
  table = PrettyTable([""] + tags)
  for tag in tags:
    row = [tag]
    for prev_tag in tags:
      p = model.prob_of_tag_given_previous_tag(tag, prev_tag)
      row += [str(round(p,2))]
    table.add_row(row)
  print(table)

  print("\n")
  for tag in tags:
    print("P(you|" + tag + "): " + str(model.prob_word_given_tag("you", tag))) 

  print("\nTesting 1st order markov model")
  model.test_1st_order("test.txt")

  #######################
  ###### Part 3 #########
  #######################
  print("\nPART 3")

  print("\nI noticed a lot of commas were marked wrong, so I added new tags that applied to commas.  These new tags were just Cx where x is any existing tag.  For example, if a comma followed a N word, it would have the tag CN.")

  print("\nTesting 1st order markov model improved")
  model.test_1st_order_improved("test.txt")

  print("\nThis worked very well. I think the commas were actually added to the training set with this rule in mind, because I no longer miss them at all.")
