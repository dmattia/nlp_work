import autograd.numpy as np
from autograd import grad
from collections import Counter, defaultdict
from math import log, exp

class SpeakerMap:
  """ A map of a speaker's name to a Speaker object.
      @param lines  The raw lines from an input file.
  """
  def __init__(self, statements):
    self._speaker_map = {}
    self.statement_count = 0
    for statement in statements:
      if statement.speaker not in self._speaker_map:
        self._speaker_map[statement.speaker] = Speaker()
      self._speaker_map[statement.speaker].add_line(statement)
      self.statement_count += 1
    self.speakers = [speaker for speaker in self._speaker_map]

  """
  @property
  def statement_count(self):
    total = 0
    for speaker in self.speakers:
      total += self._speaker_map[speaker].statement_count
    return total

  @property
  def speakers(self):
    return [speaker for speaker in self._speaker_map]
  """

  def statement_count_for_speaker_with_name(self, speaker_name):
    """ c(k)
    """
    return self._speaker_map[speaker_name].statement_count

  def word_count_for_word_and_speaker(self, word, speaker_name):
    """ c(k, w)
    """
    return self._speaker_map[speaker_name].word_counts[word]

  def probability_of_speaker_with_name(self, speaker_name):
    """ p(k)
    """
    return float(self._speaker_map[speaker_name].statement_count) / speakerMap.statement_count

  def probability_of_word_given_speaker_with_name(self, word, speaker_name):
    """ p(w | k)
    """
    speaker = self._speaker_map[speaker_name]
    word_count = speaker.word_counts[word]
    if word_count == 0:
      # Word has never been said by speaker.  Apply smoothing by pretending they said it once.
      word_count = 1
    return float(word_count) / speaker.word_count

  def probability_of_speakers_given_statement(self, statement):
    """ p(k | d) for all k.
        Takes in a statement and returns a dictionary containing all speakers as keys,
        with the cooresponding value being a float representing the percent chance that
        this statement came from the speaker.
    """
    def proportional_p_k_given_d(speaker_name):
      """ Returns exp(log(p(k) + sum(log(p(w | k))))), a number proportional to p_k
      """
      sum_total = log(self.probability_of_speaker_with_name(speaker_name))
      for word in statement.words:
        sum_total += log(self.probability_of_word_given_speaker_with_name(word, speaker_name))
      return exp(sum_total)
    proportional_p_k_given_d_values = [proportional_p_k_given_d(speaker_name) for speaker_name in self.speakers]
    # Sum of all p_k_given_d must be 1, so we can sum the proportional values and divide
    # each proportional value by that sum
    dividing_factor = sum(proportional_p_k_given_d_values)
    p_k_given_d = [proportional_value / dividing_factor for proportional_value in proportional_p_k_given_d_values]
    return dict(zip(self.speakers, p_k_given_d))

class Speaker:
  """ A speaker that contains all statements made by the speaker.
  """
  def __init__(self):
    self.statements = []
    self.words = []

  def add_line(self, statement):
    """ Adds a spoken line to this speaker's history.
        @statement
    """
    self.statements.append(statement)
    self.words += statement.words

  @property
  def statement_count(self):
    """ The total number of statements this speaker has made. c(k)
    """
    return len(self.statements)

  @property
  def word_count(self):
    """ The total number of words spoken by this speaker, where duplicate
        words are counted separately ('I am what I am' would be 5 words, not 3).
    """
    return len(self.words)

  @property
  def word_counts(self):
    return Counter(self.words)

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

if __name__ == "__main__":
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
    print(probability_dict)

  with open("data/test") as test:
    content = test.readlines()
    for line in content:
      statement = Statement(line)
      probability_dict = speakerMap.probability_of_speakers_given_statement(statement)
      max_key = max(probability_dict, key=probability_dict.get)
      print("Actual: " + statement.speaker + " Predicted: " + max_key)
