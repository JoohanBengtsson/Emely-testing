# GENERAL
# max_runs                      Decides how many dialogs that should be done in total
# is_load_conversation          True = Load from load_document. False = Generate text from the chatters specified below.
# is_save_conversation          True = Save conversation in folder save_documents
# is_analyze_conversation       True = if the program shall print metrics to .xlsx. False = If it is not necessary

max_runs = 10
is_load_conversation = False
is_save_conversation = True
is_analyze_conversation = True

# GENERATE
# conversation_length           Decides how many responses the two chatters will contribute with
# init_conv_randomly            True if the conversation shall start randomly using external tools. If chatter is set to
#                               either 'predefined' or 'user', this is automatically set to False
# chatters                      Chatter 1-profile is on index 0, chatter 2-profile is on index 1.
#                               Could be either one of ['emely', 'blenderbot',
#                               'user', 'predefined']. 'emely' assigns Emely to that chatter. 'blenderbot' assigns
#                               Blenderbot 400M to that chatter. 'user' lets the user specify the answers.
#                               'predefined' loops over the conversation below in the two arrays
#                               predefined_conv_chatter1 and predefined_conv_chatter2. Note = If metrics should be
#                               produced,
#                               Two standard conversation arrays setup for enabling hard-coded strings and conversations
#                               and try out the metrics.
# emely_URL                     If Emely is under test, specify where it is running
# convarray_init                Array for storing the conversation. Can be initialized as ["Hey", "Hey"] etc
# predefined_conv_chatter       Predefined conversation
# prev_conv_memory_chatter      How many previous sentences in the conversation shall be brought as input to any
#                               chatter. Concretely = conversation memory per chatter

conversation_length = 10
init_conv_randomly = False
chatters = ['blenderbot', 'emely']
port = "8086"
emely_URL = "http://localhost:" + port + "/inference"
convarray_init = []
predefined_conv_chatter1 = ["Hey", "I am fine thanks, how are you?"]
predefined_conv_chatter2 = ["Hello, how are you?", "I am just fine thanks. So, you are looking for a job?"]
prev_conv_memory_chatter1 = 3
prev_conv_memory_chatter2 = 3

# AFFECTIVE TEXT GENERATION
# is_affect                     Whether or not the affective text generator should be activated for first sentence.
#                               False by default
# affect                        Affect for text generation. ['fear', 'joy', 'anger', 'sadness', 'anticipation',
#                               'disgust',
#                               'surprise', 'trust']
# knob                          Amplitude for text generation. 0 to 100
# topic                         Topic for text generation. ['legal','military','monsters','politics','positive_words',
#                               'religion', 'science','space','technology']

is_affect = False
affect = "anger"
knob = 100
topic = None

# SAVE AND LOAD
# save_conv_folder              The folder which the conversations are saved
# load_conv_folder              The folder which contains the conversations
# save_analysis_name            The name of the analysis folder

save_conv_folder = "validation_QA/"
load_conv_folder = "test_run/"
save_analysis_name = chatters[1] + port

# ANALYSIS
# QA_model                      Can be ['pipeline', 'bert-squad']. Defaults to 'pipeline', indicating that only the
#                               QA-model from transformers using pipeline will be used. Somewhat worse performance, but
#                               is easier to setup. To use 'bert-squad', it needs to be setup according to 4.4 in the
#                               readme.
# show_interpret                Interpretations
# show_detailed                 Detailed results
# show_binary                   Binary results
# TESTS THAT APPLY FOR EACH DIALOG
# is_testing_REQ_P2             Toxicity
# is_testing_REQ_A3             Identical follow-up question frequency
# is_testing_REQ_A4             N-gram stuttering
# is_testing_REQ_I2             Context coherence, wrt the whole conversation
# is_testing_REQ_I3             Sentence coherence, wrt last sentence
# TESTS THAT APPLY WITH A CERTAIN PROBABILITY. SUM OF PROBABILITIES MUST NOT EXCEED 1
# p_is_testing_REQ_I1           Consistency with information about oneself
# p_is_testing_REQ_I5           Remember information for a certain amount of time
# p_is_testing_REQ_I8           Robust understanding of differently formulated information
# p_is_testing_REQ_I9           Robust understanding of implicit information based on the context
# p_is_testing_REQ_I10          Robust understanding of differently formulated questions
# p_is_testing_REQ_I11          Robust understanding of questions despite implicit information
# p_is_testing_REQ_U3           Robustness against prompts containing typing mistakes
# p_is_testing_REQ_U4           Robustness against prompts with incorrect word order
# p_is_testing_REQ_U5           Robustness against prompts with randomly omitted terms
# p_is_testing_REQ_U6           Understands questions with some words swapped for randomly chosen words
# NOTE: the variables here starting with p should add up to no more than 1. These floats represent the respective possibility 
# of that test being run during a specific conversation.

QA_model = 'bert-squad'
show_interpret = True
show_detailed = True
show_binary = True

is_testing_REQ_P2 = True
is_testing_REQ_A3 = True
is_testing_REQ_A4 = True
is_testing_REQ_I2 = True
is_testing_REQ_I3 = True

p_is_testing_REQ_I5 = 0
p_is_testing_REQ_I8 = 0
p_is_testing_REQ_I9 = 0
p_is_testing_REQ_I10 = 0
p_is_testing_REQ_I11 = 0
p_is_testing_REQ_I1 = 0
p_is_testing_REQ_U3 = 0
p_is_testing_REQ_U4 = 0
p_is_testing_REQ_U5 = 0
p_is_testing_REQ_U6 = 0

# AUXILIARY ANALYSIS VARIABLES
# maxsets_TC_REQ_I1               How many different data sets may be used for TC_REQ_I1
# maxsets_TC_REQ_I5               How many different data sets may be used for TC_REQ_I5
# maxsets_TC_REQ_I8               -----------------.........------------------ TC_REQ_I8
# maxsets_TC_REQ_I9               -----------------.........------------------ TC_REQ_I9
# maxsets_TC_REQ_I10              -----------------.........------------------ TC_REQ_I10
# maxsets_TC_REQ_I11              -----------------.........------------------ TC_REQ_I11
# maxsets_TC_REQ_U3               -----------------.........------------------ TC_REQ_U3
# maxsets_TC_REQ_U4               -----------------.........------------------ TC_REQ_U4
# maxsets_TC_REQ_U5               -----------------.........------------------ TC_REQ_U5
# maxsets_TC_REQ_U6               -----------------.........------------------ TC_REQ_U6
# maxlength_TC_REQ_I5             Maximum amount of rounds that the TC_REQ_I5 can wait for to test long term memory
# array_ux_test_cases             The array consisting of the test cases in which results should be grouped into the
#                                 closest 5-percentage group.
# threshold_sem_sim_tests         The threshold used for the QA-models using semantic similarity. The threshold level
#                                 is the threshold used for assessing the values received from the ML model

maxsets_TC_REQ_I5 = 3
maxsets_TC_REQ_I8 = 5
maxsets_TC_REQ_I10 = 3
maxsets_TC_REQ_I9 = 2
maxsets_TC_REQ_I11 = 2
maxsets_TC_REQ_I1 = 2
maxsets_TC_REQ_U3 = 2
maxsets_TC_REQ_U4 = 2
maxsets_TC_REQ_U5 = 2
maxsets_TC_REQ_U6 = 2
maxlength_TC_REQ_I5 = 5
array_ux_test_cases = ['TC_REQ_U3', 'TC_REQ_U4', 'TC_REQ_U5', 'TC_REQ_U6']

threshold_sem_sim_tests = 0.6

# DATA AUGMENTATION
# p_synonym                     Probability of switching to a synonym
# n_aug                         Number of times each test set should be augmented by switching some words with synonyms
p_synonym = 1
n_aug = 0
