# GENERAL
# max runs                      Decides how many conversations that should be done in total
# is_load_conversation          True = Load from load_document. False = Generate text from the chatters specified below.
# is_save_conversation          True = Save conversation in folder save_documents
# present_metrics               True = if the program shall print metrics to .xlsx. False = If it is not necessary
# save_conv_document            Name for saved conversation

max_runs = 1
is_load_conversation = False
is_save_conversation = False
is_analyze_conversation = True
save_conv_document = "saved_conversation.txt"

# LOAD
# load_document                 The document which contains the conversation.
load_document = "saved_conversations/sample_text.txt"

# GENERATE
# conversation_length           Decides how many responses the two chatters will contribute with
# init_conv_randomly            True if the conversation shall start randomly using external tools. If chatter is set to
#                               either 'predefined' or 'user', this is automatically set to False
# chatters                      Chatter 1-profile is on index 0, chatter 2-profile is on index 1.
#                               Could be either one of ['emely', 'blenderbot', 'user', 'predefined']
#                               'emely' assigns Emely to that chatter. 'blenderbot' assigns Blenderbot to that chatter.
#                               'user' lets the user specify
#                               the answers. 'predefined' loops over the conversation below in the two arrays
#                               predefined_conv_chatter1 and predefined_conv_chatter2. Note = If metrics should be
#                               produced,
#                               Two standard conversation arrays setup for enabling hard-coded strings and conversations
#                               and try out the metrics.
# convarray                     Array for storing the conversation. Can be initialized as ["Hey", "Hey"] etc
# predefined_conv_chatter       Predefined conversation 1
# prev_conv_memory_chatter      How many previous sentences in the conversation shall be brought as input to any
#                               chatter. Concretely = conversation memory per chatter

conversation_length = 50
init_conv_randomly = False
chatters = ['emely', 'emely']
convarray_init = []
predefined_conv_chatter1 = ["Hey", "I am fine thanks, how are you?"]
predefined_conv_chatter2 = ["Hello, how are you?", "I am just fine thanks. Do you have any pets?"]
prev_conv_memory_chatter1 = 2
prev_conv_memory_chatter2 = 3

# AFFECTIVE TEXT GENERATION
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

# ANALYSIS
# save_analysis_names           Names in output files
# show_interpret                Interpretations
# show_detailed                 Detailed results
# show_binary                   Binary results
# is_MLP1TC1                    Toxicity
# is_MLI2TC1                    Context coherence
# is_MLI3TC1                    Sentence coherence
# is_analyze_question_freq      Question frequency
# is_MLA6TC1                    Stuttering
# is_MLI1TC1                    Remember information for a certain amount of time
# is_MLI4TC1                    Understand different formulated information
# is_MLI5TC1                    Understand different formulated questions
# Is_MLI6TC1                    Understand information based on context
# Is_MLI7TC1                    Understand questions based on context
# is_MLI13TC1                   Consistency with own information
# p_MLU3TC1                     Understands questions with randomly inserted typing mistakes
# p_MLU4TC1                     Understands questions with randomly masked words

save_analysis_names = [chatters[0], chatters[1]]
show_interpret = False
show_detailed = False
show_binary = True

is_MLP1TC1 = False
is_MLI2TC1 = False
is_MLI3TC1 = False
is_analyze_question_freq = False
is_MLA6TC1 = False
p_MLI1TC1 = 0
p_MLI4TC1 = 0
p_MLI5TC1 = 0
p_MLI6TC1 = 0
p_MLI7TC1 = 0
p_MLI13TC1 = 0
p_MLU3TC1 = 0
p_MLU4TC1 = 0.5

maxsets_MLI1TC1 = 1
maxsets_MLI4TC1 = 3
maxsets_MLI5TC1 = 3
maxsets_MLI6TC1 = 2
maxsets_MLI7TC1 = 2
maxsets_MLI13TC1 = 2
maxsets_MLU3TC1 = 2
maxsets_MLU4TC1 = 2

maxlength_MLI1TC1 = 5
