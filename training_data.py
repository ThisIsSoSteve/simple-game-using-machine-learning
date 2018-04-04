import numpy as np

class Training_data:
    def __init__(self, max_state_length):
        self.batched_features = []
        self.batched_labels = []
        self.sequence_lengths = []
        self.max_state_length = max_state_length

    def number_of_examples(self):
        number_of_batched_features = len(self.batched_features)
        number_of_batched_labels = len(self.batched_features)
        if number_of_batched_features == number_of_batched_labels:
            return number_of_batched_features
        else:
            return -1


    def add_batch(self, training_features, training_labels):
        number_of_training_features = len(training_features)
        number_of_training_labels = len(training_labels)

        if number_of_training_features != number_of_training_labels:
            print('ERROR add_batch - number of features and labels did not match! features = {}, labels = {}'
            .format(number_of_training_features, number_of_training_labels))
            return

        for i in range(number_of_training_features):
            # print('i = {}'.format(i))
            # print('feature = {}'.format(training_features[0:i+1]))
            # print('label = {}'.format(training_labels[i]))

            #add padding states
            temp_training_features = training_features[0:i+1]

            #record sequence_lengths
            self.sequence_lengths.append(float(len(temp_training_features)))

            new_records_size = self.max_state_length - len(temp_training_features)

            padding_array = np.zeros((new_records_size, len(temp_training_features[0])))

            temp_training_features = np.append(temp_training_features, padding_array,0)


            self.batched_features.append(temp_training_features)
            self.batched_labels.append(training_labels[i])

    def get_random_batch(self, number_of_batches = 1):
        
        if number_of_batches > self.number_of_examples():
            number_of_batches = self.number_of_examples()

        returning_features = []
        returning_labels = []
        returning_sequence_lengths = []

        for _ in range(number_of_batches):
            random_number = np.random.randint(low=0, high=self.number_of_examples())

            returning_features.append(self.batched_features[random_number])
            returning_labels.append(self.batched_labels[random_number])
            returning_sequence_lengths.append(self.sequence_lengths[random_number])

        batched_feature = np.reshape(returning_features,[number_of_batches, self.max_state_length, 6])
        returning_labels =  np.reshape(returning_labels, [number_of_batches, 4])

        return batched_feature, returning_labels, returning_sequence_lengths


    def get_batch_by_index(self, index_number):
        if index_number > self.number_of_examples() or index_number < 0:
            return None, None, None

        batched_feature = self.batched_features[index_number]
        batched_feature = np.reshape(batched_feature,[1, len(batched_feature), len(batched_feature[0])])

        return batched_feature, self.batched_labels[index_number], self.sequence_lengths[index_number]

    def get_all_batches(self):

        batched_feature = np.reshape(self.batched_features,[len(self.batched_features), self.max_state_length, 6])

        return batched_feature, self.batched_labels, self.sequence_lengths
       

if __name__ == "__main__":
    print('training_data test')

    data_store = Training_data(12)

    #TEST 1
    state = np.array([[1.0, 0.5, 0.4, 1.0, 0.3, 0.5]])
    label = np.array([[1.0, 0.24, 0.643, 0.123]])

    data_store.add_batch(state, label)
    print('addding example 1: result: {}'.format(data_store.number_of_examples()))
    if data_store.number_of_examples() == 1:
        print('addding example 1: passed')
    else:
        print('addding example 1: failed')

    #TEST 2 - breaks down the final winning game into the different states and stores them
    state = np.array([[1.0, 0.5, 0.4, 1.0, 0.3, 0.5], [0.1, 0.6, 0.2, 1.1, 1.0, 0.6]])
    label = np.array([[2.0, 0.24, 0.643, 0.123], [3.0, 0.12, 0.34, 0.43]])

    data_store.add_batch(state, label)
    print('addding example 2: result: {}'.format(data_store.number_of_examples()))
    if data_store.number_of_examples() == 3:
        print('addding example 2: passed')
    else:
        print('addding example 2: failed')

    print('final result of features: {}'.format(data_store.batched_features))
    print('final result of labels: {}'.format(data_store.batched_labels))

#batch
    #states
        #features
'''
training_data = []
training_labels = []

features_1 = [1.0, 0.5, 0.4, 1.0, 0.3, 0.5]
label_1 = [0.56, 0.12, 0.34, 0.43]

features_2 = [0.1, 0.6, 0.2, 1.1, 1.0, 0.6]
label_2 = [0.23, 0.24, 0.643, 0.123]

states_1 = []

states_1.append(features_1)
training_labels.append(label_1)

states_2 = []

states_2.append(features_1)
states_2.append(features_2)

training_data.append(states_1)
training_data.append(states_2)
training_labels.append(label_2)

print('features')
print(training_data)
print(training_data[0])#batch 1
print(training_data[1])#batch 2
print('labels')
print(training_labels)
print(training_labels[0])
print(training_labels[1])

numpy_training_data = np.array(training_data[1])
numpy_training_labels = np.array(training_labels)

print(numpy_training_data)
print(numpy_training_data.shape)
print(numpy_training_labels)
'''