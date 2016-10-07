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
    self.ngrams = []
    self.n = n
    self.chars = 'qwertyuiopasdfghjklzxcvbnm,. '

  def train(self, filename):
    for line in open(filename):
      characters = [character for character in line if character in self.chars]
      for i in range(len(characters) - self.n):
        ngram = ""
        for j in range(self.n):
          ngram += characters[i + j]
        print(ngram)
        self.ngrams += ngram

  def start(self):
    pass

  def read(self, gram):
    pass

  @CachedAttribute
  def counts(self):
    return Counter(self.ngrams)

  @CachedAttribute
  def count(self):
    return len(set(self.ngrams))

  def lambda_func(self, u):
    # find c(u.) and store in var @count
    count = 0
    possible_next_chars = set()
    for gram in self.ngrams:
      if gram[:-1] == u:
        count += 1
        possible_next_chars += gram[-1]

    return count / (count + len(possible_next_chars))

  def prob(self, gram):
    if self.count == 0:
      return 0
    sigma = self.count / self.count + 1
    factor = 0.1
    return (self.counts[gram] + sigma) / (len(self.ngrams) + factor * sigma)

  def probs(self):
    return {gram: self.prob(gram) for gram in self.ngrams}
