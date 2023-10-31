import gensim, logging, numpy as np
import help_functions as hf
import nltk

# Define dimensions and repetitions
dimensions = [10, 50, 100, 500, 1000]
repetitions = 5

# Download wordnet
nltk.download('wordnet')

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

lemmatizer = nltk.WordNetLemmatizer()

# Load sentences from file
sentences = []
with open("lemmatized.text", "r") as file:
    for line in file:
        sentences.append(line.split())

# Define threshold
threshold = 0.00055

for dimension in dimensions:
    print(f"\n---- Running for dimension {dimension} ----\n")
    for rep in range(repetitions):
        print(f"\n--- Repetition {rep + 1} ---\n")
        sum = 0.0

        # Train model
        model = gensim.models.Word2Vec(sentences, min_count=1, sample=threshold, sg=1, vector_size=dimension)

        print(len(model.wv.key_to_index))

        # TOEFL tests
        i = 0
        number_of_tests = 80
        right_answers = 0
        number_skipped_tests = 0
        with open('new_toefl.txt', 'r') as text_file:
            while i < number_of_tests:
                line = text_file.readline()
                words = line.split()
                try:
                    words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word, 'v'), 'n'), 'a') for
                             word in words]
                    vectors = []
                    if words[0] in model.wv:
                        k = 1
                        vectors.append(model.wv[words[0]])
                        while k < 5:
                            if words[k] in model.wv:
                                vectors.append(model.wv[words[k]])
                            else:
                                vectors.append(np.random.randn(dimension))
                            k += 1
                        right_answers += hf.get_answer_mod(vectors)
                except KeyError:
                    number_skipped_tests += 1
                    print("skipped test: " + str(i) + "; Line: " + str(words))
                except IndexError:
                    print(i)
                    print(line)
                    print(words)
                    break
                i += 1

        sum += 100 * float(right_answers) / float(number_of_tests)
        print("Dimension = " + str(dimension) + " Repetition = " + str(rep + 1) + " Threshold ferq = " + str(
            threshold) + " Percentage of correct answers: " + str(sum) + "%")