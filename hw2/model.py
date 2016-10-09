from collections import Counter
import math

class CachedAttribute(object):
    '''Computes attribute value and caches it in the instance.
    From the Python Cookbook (Denis Otkidach)
    This decorator allows you to create a CachedAttribute which can be computed once and
    accessed many times. Sort of like memoization.
    
    author: Denis Otkidach
    source: http://code.activestate.com/recipes/276643-caching-and-aliasing-with-descriptors/
    '''
    def __init__(self, method, name=None):
        self.method = method
        self.name = name or method.__name__
        self.__doc__ = method.__doc__
    def __get__(self, inst, cls):
        if inst is None:
            return self
        result = self.method(inst)
        setattr(inst, self.name, result)
        return result

class NGramModel(object):
  def __init__(self, n):
    self.ngrams = [['']]
    self.n = n
    self.chars = set()

    # Caching
    self.c_udots = {}
    self.lambda_func_cache = {}
    self.prob_cache = {}

  def train(self, filename):
    for gram_size in range(1, self.n+1):
      igrams = []
      for line in open(filename):
        characters = [character for character in line]
        for c in characters:
          self.chars.add(c)
        for i in range(len(characters) - gram_size):
          ngram = ""
          for j in range(gram_size):
            ngram += characters[i + j]
          igrams.append(ngram)
      self.ngrams.append(igrams)
      print("Finished finding grams of size: " + str(gram_size))

  def start(self):
    self.history = ''

  def read(self, w):
    self.history += w

  def c_udot(self, u):
    # find c(u.)
    if u in self.c_udots:
      return self.c_udots[u]
    count = 0
    for w in self.chars:
      count += self.c_uw(u + w)
    self.c_udots[u] = count
    return count

  def c_uw(self, uw):
    # find c(uw)
    return self.counts[len(uw)][uw]

  def lambda_func(self, u):
    # find c(u.) and store in var @count
    if u in self.lambda_func_cache:
      return self.lambda_func_cache[u]
    count = 0
    possible_next_chars = 0
    for w in self.chars:
      temp = self.c_uw(u + w)
      count += temp
      if temp > 0:
        possible_next_chars += 1
    if count == 0:
      self.lambda_func_cache[u] = 0
      return 0
    self.lambda_func_cache[u] = count / (count + possible_next_chars)
    return count / (count + possible_next_chars)

  def prob(self, w):
    """ Returns the probability of the next character being w given self.history
    """
    start_of_gram = self.history[-(self.n-1):]
    return self.prob_of_gram(start_of_gram + w)

  @CachedAttribute
  def counts(self):
    return [Counter(ngram) for ngram in self.ngrams]

  def prob_of_gram(self, gram):
    """ Returns the probability of the next character being gram[-1] given gram[:-1]
    """
    if gram in self.prob_cache:
      return self.prob_cache[gram]
    if len(gram) == 1:
      return self.counts[1][gram] / len(self.ngrams[1])
    u = gram[:-1]
    w = gram[-1]
    lambda_u = self.lambda_func(u)
    c_uw = self.c_uw(gram)
    c_udot = self.c_udot(u)
    if c_udot == 0:
      temp = (1 - lambda_u) * self.prob_of_gram(gram[1:])
      self.prob_cache[gram] = temp
      return temp
    temp = lambda_u * c_uw / c_udot + (1 - lambda_u) * self.prob_of_gram(gram[1:])
    self.prob_cache[gram] = temp
    return temp

  def probs(self):
    d = {}
    for w in self.chars:
      d[w] = self.prob(w)
    return d

if __name__ == "__main__":
  gram_size = 10
  m = NGramModel(gram_size)
  m.train("english/train")
  m.start()

  print("\nGetting the most probable char for the first 10 chars of the dev set")
  with open("english/dev") as devFile:
    chars = devFile.readline()[:10]
    for char in chars:
      prob, guess = max((p, w) for (w, p) in m.probs().items())  
      print("Guess: " + guess + " with a probability of: " + str(prob) + ". Actual: " + char)
      m.read(char)

  print("\nFinding Perplexity")
  with open("english/test") as testFile:
    content = testFile.read().replace('\n', '')
    total = 0
    for char_index in range(len(content)):
      gram = content[char_index-gram_size+1 : char_index+1]
      if len(gram) > 0:
        total += math.log(m.prob_of_gram(gram))
    total /= len(content)
    print("Perplexity: " + str(math.exp(-total)))

  # Reset the model's history
  m.start()

  # Test on test set
  print("\nTesting percent correct on the test file")
  with open("english/test") as testFile:
    lines = testFile.readlines()
    count_correct = 0
    count_total = 0
    for line in lines:
      for char_index in range(len(line)):
        _, guess = max((p, w) for (w, p) in m.probs().items())
        m.read(line[char_index])
        if line[char_index] == guess:
          count_correct += 1
        count_total += 1

  print("Percent correct: " + str(count_correct / float(count_total)))
