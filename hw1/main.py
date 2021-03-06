from collections import Counter, defaultdict
from math import log, exp
import nltk
import random

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

class SpeakerMap:
  """ A map of a speaker's name to a Speaker object.
      @param lines  The raw lines from an input file.
  """
  def __init__(self, statements):
    self._speaker_map = {}
    for statement in statements:
      if statement.speaker not in self._speaker_map:
        self[statement.speaker] = Speaker()
      self[statement.speaker].add_line(statement)

  def __getitem__(self, key):
    """ Overload the [] operator for retreiving a Speaker
    """
    return self._speaker_map[key]

  def __setitem__(self, key, newValue):
    """ Overload the [] operator for setting a Speaker
    """
    self._speaker_map[key] = newValue

  @CachedAttribute
  def statement_count(self):
    """ Returns the total number of statements made by all speakers
    """
    total = 0
    for speaker_name in self._speaker_map:
      total += self[speaker_name].statement_count
    return total

  @CachedAttribute
  def speakers(self):
    """ Returns a list of all speaker names
    """
    return [speaker_name for speaker_name in self._speaker_map]

  @CachedAttribute
  def bigrams(self):
    bigrams = []
    for speaker_name in self.speakers:
      bigrams += self[speaker_name].bigrams
    return bigrams

  @CachedAttribute
  def words(self):
    """ A list containing all words said by any speaker
    """
    words = []
    for speaker_name in self.speakers:
      words += self[speaker_name].words
    return words

  @CachedAttribute
  def unique_word_count(self):
    """ The total number of unique words said by any speaker
    """
    return len(set(self.words))

  @CachedAttribute
  def unique_bigram_count(self):
    return len(set(self.bigrams))

  def statement_count_for_speaker_with_name(self, speaker_name):
    """ c(k)
    """
    return self[speaker_name].statement_count

  def word_count_for_word_and_speaker(self, word, speaker_name):
    """ c(k, w)
    """
    return self[speaker_name].word_counts[word]

  def probability_of_speaker_with_name(self, speaker_name):
    """ p(k)
    """
    return float(self[speaker_name].statement_count) / self.statement_count

  def probability_of_word_given_speaker_with_name(self, word, speaker_name):
    """ p(w | k), using add-one smoothing
    """
    speaker = self[speaker_name]
    word_count = speaker.word_counts[word]
    smoothing_value = 0.1
    possible_word_count = self.unique_word_count + 1 # Add one for a generic, unseen word
    return float(word_count + smoothing_value) / (speaker.word_count + smoothing_value * possible_word_count)

  def probability_of_bigram_given_speaker_with_name(self, bigram, speaker_name):
    """ p(bigram | k), using add-one smoothing
    """
    speaker = self[speaker_name]
    bigram_count = speaker.bigram_counts[bigram]
    smoothing_value = 0.1
    possible_bigram_count = self.unique_bigram_count + 1 # Add one for a generic, unseen bigram
    return float(bigram_count + smoothing_value) / (speaker.bigram_count + smoothing_value * possible_bigram_count)

  def pos_probability_of_word_given_speaker_with_name(self, word, speaker_name):
    """ p(pos | k), using add-one smoothing
    """
    speaker = self[speaker_name]
    word_count = speaker.pos_counts[nltk.pos_tag([word])[0][1]]
    smoothing_value = 0.1
    possible_word_count = self.unique_word_count + 1 # Add one for a generic, unseen word
    return float(word_count + smoothing_value) / (speaker.word_count + smoothing_value * possible_word_count)

  def bigram_prob_of_speakers_given_statement(self, statement):
    def proportional_p_b_given_d(speaker_name):
      """
      total = self.probability_of_speaker_with_name(speaker_name)
      for bigram in statement.bigrams:
        total *= self.probability_of_bigram_given_speaker_with_name(bigram, speaker_name)
      return total
      """
      sum_total = log(self.probability_of_speaker_with_name(speaker_name))
      for bigram in statement.bigrams:
        sum_total += log(self.probability_of_bigram_given_speaker_with_name(bigram, speaker_name))
      return sum_total
      
    proportional_probabilities = [proportional_p_b_given_d(speaker_name) for speaker_name in self.speakers]

    # Make these values larger in order to ensure e^x does not equate to 0 due to roundoff error.
    max_value = max(proportional_probabilities)
    proportional_probabilities = [p - max_value for p in proportional_probabilities]
    proportional_probabilities = [exp(p) for p in proportional_probabilities]

    # Now make these probabilities true probabilities by making them sum to 1
    dividing_factor = sum(proportional_probabilities)
    probabilities = [p / dividing_factor for p in proportional_probabilities]
    return dict(zip(self.speakers, probabilities))

  def pos_probability_of_speakers_given_statement(self, statement):
    def proportional_p_k_given_d(speaker_name):
      sum_total = log(self.probability_of_speaker_with_name(speaker_name))
      for word in statement.words:
        sum_total += log(self.pos_probability_of_word_given_speaker_with_name(word, speaker_name))
      return sum_total
    proportional_probabilities = [proportional_p_k_given_d(speaker_name) for speaker_name in self.speakers]

    # Make these values larger in order to ensure e^x does not equate to 0 due to roundoff error.
    max_value = max(proportional_probabilities)
    proportional_probabilities = [exp(p - max_value) for p in proportional_probabilities]

    # Now make these probabilities true probabilities by making them sum to 1
    dividing_factor = sum(proportional_probabilities)
    probabilities = [p / dividing_factor for p in proportional_probabilities]
    return dict(zip(self.speakers, probabilities))

  def probability_of_speakers_given_statement(self, statement):
    """ p(k | d) for all k.
        Takes in a statement and returns a dictionary containing all speakers as keys,
        with the cooresponding value being a float representing the percent chance that
        this statement came from the speaker.
    """
    def proportional_p_k_given_d(speaker_name):
      """ Returns log(p(k)) + sum(log(p(w | k))).
          This is a useful value for determining the proportional probability of a speaker.
          The result does not include the final exponential call in order to preserve small values.
      """
      sum_total = log(self.probability_of_speaker_with_name(speaker_name))
      for word in statement.words:
        sum_total += log(self.probability_of_word_given_speaker_with_name(word, speaker_name))
      return sum_total
    proportional_probabilities = [proportional_p_k_given_d(speaker_name) for speaker_name in self.speakers]
  
    # Make these values larger in order to ensure e^x does not equate to 0 due to roundoff error.
    max_value = max(proportional_probabilities)
    proportional_probabilities = [exp(p - max_value) for p in proportional_probabilities]

    # Now make these probabilities true probabilities by making them sum to 1
    dividing_factor = sum(proportional_probabilities)
    probabilities = [p / dividing_factor for p in proportional_probabilities]
    return dict(zip(self.speakers, probabilities))

class Speaker:
  """ A speaker that contains all statements made by the speaker.
  """
  def __init__(self):
    self.statements = []
    self.words = []
    self.bigrams = []
    self.parts_of_speech = []

  def add_line(self, statement):
    """ Adds a spoken line to this speaker's history.
        @statement
    """
    self.statements.append(statement)
    self.words += statement.words
    self.bigrams += statement.bigrams
    self.parts_of_speech += statement.parts_of_speech

  @CachedAttribute
  def statement_count(self):
    """ The total number of statements this speaker has made. c(k)
    """
    return len(self.statements)

  @CachedAttribute
  def word_count(self):
    """ The total number of words spoken by this speaker, where duplicate
        words are counted separately ('I am what I am' would be 5 words, not 3).
    """
    return len(self.words)

  @CachedAttribute
  def bigram_count(self):
    return len(self.bigrams)

  @CachedAttribute
  def word_counts(self):
    return Counter(self.words)

  @CachedAttribute
  def bigram_counts(self):
    return Counter(self.bigrams)

  @CachedAttribute
  def pos_counts(self):
    return Counter(self.parts_of_speech)

class LambdaMap:
  def __init__(self, train_statements, dev_statements, test_statements):
    self.statements = train_statements
    self.dev_statements = dev_statements
    self.test_statements = test_statements
    self.test_length = len(self.test_statements)
    self.model = {}
    self.learning_rate = 0.01

    self.speakers = set([statement.speaker for statement in self.statements])

    for speaker in self.speakers:
      self.model[speaker] = defaultdict(float)

  def negative_log_prob(self):
    total = 0
    for test_statement in self.test_statements:
      total -= log(self.p_k_given_d(test_statement)[test_statement.speaker])
    return total

  def judge_accuracy(self, test=False):
    statements = self.test_statements if test else self.dev_statements
    random.shuffle(statements)
    correct = 0
    for test_statement in statements:
      if test_statement.speaker == self.predict_speaker(test_statement):
        correct += 1
    return float(correct) / self.test_length

  def train(self):
    self.learning_rate *= .95
    random.shuffle(self.statements)
    for statement in self.statements:
      self.update_for_statement(statement)

  def p_k_given_d(self, statement):
    value_map = defaultdict(int)
    for speaker in self.speakers:
      total = self.model[speaker][""]
      for word in statement.words:
        total += self.model[speaker][word]
      value_map[speaker] = total
    value_map = {speaker: exp(value) for speaker, value in value_map.items()}
    sum_value = sum(value_map.values())
    return {speaker: value / sum_value for speaker, value in value_map.items()}

  def update_for_statement(self, statement):
    p_k_given_d_map = self.p_k_given_d(statement)
    for word in (statement.words + [""]):
      self.model[statement.speaker][word] += self.learning_rate
      for speaker in self.speakers:
        self.model[speaker][word] -= self.learning_rate * p_k_given_d_map[speaker]

  def predict_speaker(self, statement):
    value_map = self.p_k_given_d(statement)
    return max(value_map, key=lambda x: value_map[x])

class Statement:
  """ A single statement that contains a speaker's name
     and the words of that statement in a list.
  """
  def __init__(self, raw_line):
    """ Parses a raw line from a text file.
        @param raw_line A line of the pattern "speaker statement"
               where statement is a collection of multiple words.
    """
    raw_words = raw_line.strip().split()
    self.speaker = raw_words[0]
    self.words = raw_words[1:]

    # Find all bigrams
    self.bigrams = []
    for i in range(len(self.words) - 1):
      self.bigrams += self.words[i] + " " + self.words[i+1]

    self.parts_of_speech = nltk.pos_tag(self.words)
    self.parts_of_speech = [tup[1] for tup in self.parts_of_speech]

def test_bayes():
  print("#######################")
  print("# TESTING NAIVE BAYES #")
  print("#######################")
  with open("data/train") as train:
    content = train.readlines()
    statements = map(Statement, content)
    speakerMap = SpeakerMap(statements)

  speakers = ["trump", "clinton"]
  words = ["country", "president"]

  # Print c(k), the count of statements (documents) each speaker said
  print("c(k) values:")
  for speaker_name in speakers:
    count = speakerMap.statement_count_for_speaker_with_name(speaker_name)
    print(speaker_name + " spoke " + str(count) + " statements")
    
  # Print c(k, w), the number of times a speaker said a given word
  print("\nc(k, w) values:")
  for speaker_name in speakers:
    for word in words:
      word_count = speakerMap.word_count_for_word_and_speaker(word, speaker_name)
      print(speaker_name + " said the word " + word + " " + str(word_count) + " times.")

  # Print p(k), the probability a given speaker based on the total statements they spoke
  # divided by the total number of statements.
  print("\np(k) values:")
  for speaker_name in speakers:
    p_k = speakerMap.probability_of_speaker_with_name(speaker_name)
    print("P(" + speaker_name + "): " + str(p_k))

  # Print p(w | k), the probability of a word given a speaker
  print("\np(w | k) values:")
  for speaker_name in speakers:
    for word in words:
      p_w_k = speakerMap.probability_of_word_given_speaker_with_name(word, speaker_name)
      print("P(" + word + "|" + speaker_name + "): " + str(p_w_k))

  # Prink p(k | d), the probability of a speaker based on a given statement for all speakers
  print("\np(k | d) values:")
  with open("data/dev") as dev:
    first_line = dev.readline()
    probability_dict = speakerMap.probability_of_speakers_given_statement(Statement(first_line))
    for key in probability_dict:
      print(key + ": " + str(probability_dict[key]))

  with open("data/test") as test:
    content = test.readlines()

    def test_if_correct(line):
      """ Takes in a document and returns if the naive bayes can correctly predict the speaker
      """
      statement = Statement(line)
      unigram_probability_dict = speakerMap.probability_of_speakers_given_statement(statement)
      most_likely_speaker = max(unigram_probability_dict, key=unigram_probability_dict.get)
      return most_likely_speaker == statement.speaker

    def test_if_bigram_correct(line):
      statement = Statement(line)
      bigram_probability_dict = speakerMap.bigram_prob_of_speakers_given_statement(statement)
      most_likely_speaker = max(bigram_probability_dict, key=bigram_probability_dict.get)
      return most_likely_speaker == statement.speaker

    def test_if_pos_correct(line):
      """ Takes in a document and returns if the naive bayes can correctly predict the speaker
      """
      statement = Statement(line)
      pos_probability_dict = speakerMap.pos_probability_of_speakers_given_statement(statement)
      most_likely_speaker = max(pos_probability_dict, key=pos_probability_dict.get)
      return most_likely_speaker == statement.speaker

    def test_if_average_correct(line):
      statement = Statement(line)
      unigram_dict = speakerMap.probability_of_speakers_given_statement(statement)
      bigram_dict = speakerMap.bigram_prob_of_speakers_given_statement(statement)
      pos_dict = speakerMap.pos_probability_of_speakers_given_statement(statement)
      average_dict = {}
      for speaker in speakerMap.speakers:
        average_dict[speaker] = (unigram_dict[speaker] + bigram_dict[speaker] + pos_dict[speaker]) / 3
      most_likely_speaker = max(average_dict, key=average_dict.get)
      return most_likely_speaker == statement.speaker

    print("\nFinal tests:")
    correct = [test_if_correct(line) for line in content]
    correct_count = correct.count(True)
    print(str(correct_count) + " correct out of " + str(len(content)) + " using unigrams")

    bigram_correct = [test_if_bigram_correct(line) for line in content]
    bigram_correct_count = bigram_correct.count(True)
    print(str(bigram_correct_count) + " correct out of " + str(len(content)) + " using bigrams")

    pos_correct = [test_if_pos_correct(line) for line in content]
    pos_correct_count = pos_correct.count(True)
    print(str(pos_correct_count) + " correct out of " + str(len(content)) + " using parts of speech")

    avg_correct = [test_if_average_correct(line) for line in content]
    avg_correct_count = avg_correct.count(True)
    print(str(avg_correct_count) + " correct out of " + str(len(content)) + " using all of these factors")

  print("\nImplementation Choices:\n"\
      + "I did add n smoothing, with n = 0.1.  This type of smoothing resulted in the largest accuracy on dev.\n"\
      + "I also had to adjust the proportional probability values by a constant multiplicitive factor to avoid "\
      + "underflow when taking the exponential\n"
  )

def test_log_regression():
  print("#################################")
  print("# TESTING LOGRITHMIC REGRESSION #")
  print("#################################")
  with open("data/train") as train:
    train_content = train.readlines()
    train_statements = [Statement(line) for line in train_content]

  with open("data/test") as test:
    test_content = test.readlines()
    test_length = len(test_content)
    test_statements = [Statement(line) for line in test_content]

  with open("data/dev") as dev:
    dev_content = dev.readlines()
    dev_statements = [Statement(line) for line in dev_content]

  lambdaMap = LambdaMap(train_statements, dev_statements, test_statements)

  for i in range(30):
    lambdaMap.train()
    print("Iteration number: " + str(i + 1))
    print("Negative log likelihood: " + str(lambdaMap.negative_log_prob()))
    print("Accuracy on dev: " + str(lambdaMap.judge_accuracy()) + "\n")

  speakers = ["trump", "clinton"]
  words = ["country", "president"]

  # Print lambda(k) for trump, clinton
  print("λ(k) values:")
  for speaker in speakers:
    print("λ(" + speaker + "): " + str(lambdaMap.model[speaker][""]))

  # Print lambda(k, w) values
  print("\nλ(k, w) values:")
  for speaker in speakers:
    for word in words:
      print("λ(" + speaker + ", " + word + "): " + str(lambdaMap.model[speaker][word]))

  print("\nP(k | d) for the first line of dev:")
  print(lambdaMap.p_k_given_d(dev_statements[0]))

  print("\nAccuracy on test: " + str(lambdaMap.judge_accuracy(test=True)))

  print("\nImplementation Choices:\n"\
      + "I randomly shuffled the training lines before each iteration, as well as each test set I tested on\n"\
      + "I started with a learning rate of 0.1. This allowed for quick increases in accuracy initially, "\
      + "but I decreased this value by 5% each iteration.  This made the steps smaller towards the end.\n"\
      + "I chose 30 iterations because at this point, the learning rate is small enough that the model "\
      + "should be hovering around its maximum. Note: .95^30 ~= 20% of the original learning rate.\n"\
      + "for λ(k), I assumed there was a dummy word (the empty string in my case) that occurred once per document."
  )

if __name__ == "__main__":
  test_bayes()
  test_log_regression()
