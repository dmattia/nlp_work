from collections import Counter

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
    self.chars = 'qwertyuiopasdfghjklzxcvbnm,. '

  def train(self, filename):
    for gram_size in range(1, self.n+1):
      print("Finding grams of size: " + str(gram_size))
      igrams = []
      for line in open(filename):
        characters = [character for character in line if character in self.chars]
        for i in range(len(characters) - gram_size):
          ngram = ""
          for j in range(gram_size):
            ngram += characters[i + j]
          igrams.append(ngram)
      self.ngrams.append(igrams)

  def start(self):
    self.history = ''

  def read(self, w):
    print("'" + w + "' was pressed")
    self.history += w

  def c_udot(self, u):
    # find c(u.)
    count = 0
    for gram in self.ngrams[len(u)+1]:
      if gram[:-1] == u:
        count += 1
    return count

  def c_uw(self, uw):
    # find c(u.)
    count = 0
    for gram in self.ngrams[len(uw)]:
      if gram == uw:
        count += 1
    return count

  def lambda_func(self, u):
    # find c(u.) and store in var @count
    count = 0
    possible_next_chars = set()
    for gram in self.ngrams[len(u)+1]:
      if gram[:-1] == u:
        count += 1
        possible_next_chars.add(gram[-1])

    return count / (count + len(possible_next_chars))

  def prob(self, w):
    """ Returns the probability of the next character being w given self.history
    """
    print(self.history)
    start_of_gram = self.history[-(self.n-1):]
    print("Finding probability of " + w + " given the start of a gram: " + start_of_gram)
    return self.prob_of_gram(start_of_gram + w)

  def prob_of_gram(self, gram):
    """ Returns the probability of the next character being gram[-1] given gram[:-1]
    """
    if len(gram) == 1:
      return Counter(self.ngrams[1])[gram] / len(self.ngrams[1])
    u = gram[:-1]
    w = gram[-1]
    lambda_u = self.lambda_func(u)
    c_uw = self.c_uw(gram)
    c_udot = self.c_udot(u)
    return lambda_u * c_uw / c_udot + (1 - lambda_u) * self.prob_of_gram(gram[1:])

  def probs(self):
    #return {gram: self.prob(gram) for gram in self.ngrams[self.n]}
    d = {}
    #for gram in self.ngrams[self.n]:
    for w in self.chars:
      d[w] = self.prob(w)
      print("Probability of " + w + " is " + str(d[w])) 
    return d
