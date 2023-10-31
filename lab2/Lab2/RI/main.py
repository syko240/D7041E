import numpy as np
import text_functions as tf
import nltk
import time

nltk.download('wordnet')
lemmatizer = nltk.WordNetLemmatizer()  # create an instance of lemmatizer

dimentions = [1000, 4000, 10000]
repetitions = 5

def run_simulation(threshold=15000, dimension=2000, ones_number=2, window_size=2,
                   test_name="c:/Users/andre/D7041E/lab2/Lab2/RI/new_toefl.txt",
                   data_file_name="c:/Users/andre/D7041E/lab2/Lab2/RI/lemmatized.text"):

    STARTTIME = time.time() # start time stamp
    zero_vector = np.zeros(dimension)
    amount_dictionary = {}

    # Count how many times each word appears in the corpus
    text_file = open(data_file_name, "r")
    for line in text_file:
        if line != "\n":
            words = line.split()
            for word in words:
                if amount_dictionary.get(word) is None:
                    amount_dictionary[word] = 1
                else:
                    amount_dictionary[word] += 1
    text_file.close()

    dictionary = {} #vocabulary and corresponding random high-dimensional vectors
    word_space = {} #embeddings

    #Create a dictionary with the assigned random high-dimensional vectors
    text_file = open(data_file_name, "r")
    for line in text_file: #read line in the file
        words = line.split() # extract words from the line
        for word in words:  # for each word
            if dictionary.get(word) is None: # If the word was not yed added to the vocabulary
                if amount_dictionary[word] < threshold:
                    dictionary[word] = tf.get_random_word_vector(dimension, ones_number) # assign a  
                else:
                    dictionary[word] = np.zeros(dimension) # frequent words are assigned with empty vectors. In a way they will not contribute to the word embedding

    text_file.close()


    #Note that in order to save time we only create embeddings for the words needed in the TOEFL task

        #Find all unique words amongst TOEFL tasks and initialize their embeddings to zeros    
    number_of_tests = 0
    text_file = open(test_name, "r") #open TOEFL tasks
    for line in text_file:
            words = line.split()
            words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word, 'v'), 'n'), 'a') for word in
                    words] # lemmatize words in the current test
            word_space[words[0]] = np.zeros(dimension)
            word_space[words[1]] = np.zeros(dimension)
            word_space[words[2]] = np.zeros(dimension)
            word_space[words[3]] = np.zeros(dimension)
            word_space[words[4]] = np.zeros(dimension)
            number_of_tests += 1
    text_file.close()

    # Processing the text to build the embeddings
    lines = ["\n"] * (2 * window_size)
    text_file = open(data_file_name, "r")
    for i in range(window_size, 2 * window_size):
        lines[i] = text_file.readline().split()

    for line in text_file:
        lines.append(line.split())
        words = lines[window_size]
        length = len(words)
        for i, word in enumerate(words):
            if word in word_space:
                # Handle left context
                for k in range(1, window_size + 1):
                    if i - k >= 0:  # If there are enough words to the left in the current line
                        word_space[word] = np.add(word_space[word], np.roll(dictionary[words[i - k]], -k))
                    else:  # Attempt to get words from previous lines
                        prev_line = lines[window_size - k]
                        if prev_line and len(prev_line) >= k:  # Ensure the line exists and has enough words
                            word_space[word] = np.add(word_space[word], np.roll(dictionary[prev_line[-k]], -k))

                # Handle right context
                for k in range(1, window_size + 1):
                    if i + k < length:  # If there are enough words to the right in the current line
                        word_space[word] = np.add(word_space[word], np.roll(dictionary[words[i + k]], k))
                    else:  # Attempt to get words from the next lines
                        next_line = lines[window_size + k] if window_size + k < len(lines) else None
                        if next_line and k <= len(next_line):  # Ensure the line exists and has enough words
                            word_space[word] = np.add(word_space[word], np.roll(dictionary[next_line[k - 1]], k))
        lines.pop(0)

    text_file.close()



    #Testing of the embeddings on TOEFL
    a = 0.0 # accuracy of the encodings    
    i = 0
    text_file = open(test_name, 'r')
    right_answers = 0.0 # variable for correct answers
    number_skipped_tests = 0.0 # some tests could be skipped if there are no corresponding words in the vocabulary extracted from the training corpus
    while i < number_of_tests:
            line = text_file.readline() #read line in the file
            words = line.split()  # extract words from the line
            words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word, 'v'), 'n'), 'a') for word in
                    words]  # lemmatize words in the current test
            try:
                
                if not(amount_dictionary.get(words[0]) is None): # check if there word in the corpus for the query word
                    k = 1
                    while k < 5:
                        # if amount_dictionary.get(words[k]) is None:
                        #     word_space[words[k]] = np.random.randn(dimension)
                        if np.array_equal(word_space[words[k]], zero_vector): # if no representation was learnt assign a random vector
                            word_space[words[k]] = np.random.randn(dimension)
                        k += 1
                    right_answers += tf.get_answer_mod([word_space[words[0]],word_space[words[1]],word_space[words[2]],
                                word_space[words[3]],word_space[words[4]]]) #check if word is predicted right
            except KeyError: # if there is no representation for the query vector than skip
                number_skipped_tests += 1
                print("skipped test: " + str(i) + "; Line: " + str(words))
            except IndexError:
                print(i)
                print(line)
                print(words)
                break
            i += 1
    text_file.close()
    a += 100 * right_answers / number_of_tests
    print(str(dimension) + " Percentage of correct answers: " + str(100 * right_answers / number_of_tests) + "%")

    ENDTIME = time.time() # end time stamp
    TIMEELAPSED = ENDTIME - STARTTIME # get elapsed time

    print("Time elapsed: ", TIMEELAPSED, "s")


run_simulation()

'''for dim in dimentions:
    print(f"\n---- Running for dimension {dim} ----\n")
    for rep in range(repetitions):
        print(f"\n--- Repetition {rep + 1} ---\n")
        run_simulation(dimension=dim)'''
