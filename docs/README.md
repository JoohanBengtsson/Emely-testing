# 1. Introduction

This project aims to provide an open-source test framework that could be used to test any chatbot. The script is setup so that it is easy to add a new chatbot or text generator in order to assess it, more details about this can be found in **Implementation details**.

The script will produce a conversation between two chatters, hereafter called chatter1 respectively chatter2, and then assess the conversation with regards to some predefined quality aspects. The quality aspects will be defined below in the **Software Requirements Specification** chapter. These quality aspects will be assessed and then written to a .xlsx-file, for the user to use for further assessment of the chatbot.

# 2. Requirements evaluated by this testing framework

The framework is evaluating a set of requirements within four different categories:
* Understanding - The understanding of the input itself.
* Intelligence  - The understandong of the context and the conversation.
* Personality   - The choice of answers.
* Answering     - The correctness of the answer.

## 2.1 Explanation of requirements and test cases

| Category      | Code   | Explanation                                                            | Example                                | Test case category        |
|---------------|--------|------------------------------------------------------------------------|----------------------------------------|---------------------------|
| Understanding | ML-U3  | Understanding of sentences with typing mistakes                        | Whqt is tje name rf my cat?            | Question-Answer           |
|               | ML-U4  | Understanding of sentences with incorrect word order                   | What the is name of cat my?            | Question-Answer           |
|               | ML-U5  | Understanding of sentences with some words left out                    | What the name my cat?                  | Question-Answer           |
|               | ML-U6  | Understanding of sentences with some words replaced with a random word | What banana the name of my cat?        | Question-Answer           |
| Intelligence  | ML-I1  | Long term memory assessment                                            |                                        | Question-Answer           |
|               | ML-I2  | Coherence assessment wrt the conversation                              | “I like cars”, “That sounds tasty”     | Sentence-BERT             |
|               | ML-I3  | Coherence assessment wrt the input sentence                            | The kitten which I own is called John. | Question-Answer           |
|               | ML-I4  | Different formulated information assessment                            | The kitten which I own is called John. | Question-Answer           |
|               | ML-I5  | Different formulated questions assessment                              | What name does my cat possess?         | Question-Answer           |
|               | ML-I6  | Context dependent information understanding                            | I have a cat. His name is John.        | Question-Answer           |
|               | ML-I7  | Context dependent questions understanding                              | I love my cat. What was its name?      | Question-Answer           |
|               | ML-I13 | Consistency with own information                                       |                                        | Question                  |
| Personality   | ML-P1  | Toxicities assessment                                                  |                                        | Detoxifyer                |
| Answering     | ML-A6  | Repetition avoidance assessment                                        | I have I have I have a cat.            | N-grams                   |
|               | ML-A7  | Repeated questions avoidance assessment                                |                                        | Simple check              |

* Understanding of sentences with typing mistakes (ML-U3):
  How successful the chatbot is to understand sentences with words having typing mistakes. It is given some information and then asked about the information with typing mistakes. The typing mistake introduction only applies on the questions. The amount of words having typing mistakes and the proportion of typing mistakes in each of these words is randomized for each test.
  The presented score is a primary list containing secondary lists with format [successful attempts, total number of attempts]. Each element of the primary list is related to the (number of typing mistake words * proportion typing mistakes). It gives a base for creating a histogram over success rate over the number of swaps.
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Understanding of sentences with incorrect word order (ML-U4):
  How successful the chatbot is to understand sentences with words presented in an incorrect order. It is given some information and then asked about the information with words swapped. The word swap only applies on the questions. The amount of words swapped is randomized for each test.
  The presented score is a primary list containing secondary lists with format [successful attempts, total number of attempts]. Each element of the primary list is related to the number of swaps in a sentence. The format is therefore [[success 0 swaps, total 0 swaps], [success 1 swaps, total 1 swaps], [success 2 swaps, total 2 swaps], ...]. It gives a base for creating a histogram over success rate over the number of swaps.
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Understanding of sentences with some words left out (ML-U5):
  How successful the chatbot is to understand sentences with some words left out. It is given some information and then asked about the information with words left out. The word masking only applies on the questions. The amount of words swapped is randomized for each test.
  The presented score is a primary list containing secondary lists with format [successful attempts, total number of attempts]. Each element of the primary list is related to the proportion of words left out in a sentence. The format is therefore [[success 0-5% removed, total 0-5% removed], [success 5-10% removed, total 5-10%removed], [success 10-15% removed, total 10-15% removed], ...]. It gives a base for creating a histogram over success rate over the number of words removed.
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Understanding of sentences with some words replaced with a random word (ML-U6):
  How successful the chatbot is to understand sentences with some words replaced with a random word. It is given some information and then asked about the information with words replaced. The word replacement only applies on the questions. The amount of words replaced is randomized for each test.
  The presented score is a primary list containing secondary lists with format [successful attempts, total number of attempts]. Each element of the primary list is related to the proportion of words replaced in a sentence. The format is therefore [[success 0-5% replaced, total 0-5% replaced], [success 5-10% replaced, total 5-10% replaced], [success 10-15% replaced, total 10-15% replaced], ...]. It gives a base for creating a histogram over success rate over the number of words replaced.
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Long term memory assessment (ML-I1) - how successful the chatbot is to remember details after varying amount of conversation rounds.
  The presented score is a list with format [successful attempts, total number of attempts].
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Coherence assessment wrt the conversation (ML-I2):
  Assesses if the response produced by a chatter coherent with respect to the whole conversation**. Assessed with Sentence-BERT.
  The presented score is a list with format [successful attempts, total number of attempts].
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Coherence assessment wrt the input sentence (ML-I3):
  Assesses if the response produced by a chatter coherent with respect to the input sentence. Assessed with Sentence-BERT.
  The presented score is a list with format [successful attempts, total number of attempts].
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Different formulated information assessment (ML-I4):
  How successful the chatbot is to understand different formulated information. It is first given some information formulated in different ways and then asked about the information.
  The presented score is a list with format [successful attempts, total number of attempts].
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Different formulated questions assessment (ML-I5):
  How successful the chatbot is to understand different formulated questions. It is first given some information and then asked about the information with questions formulated in different ways.
  The presented score is a list with format [successful attempts, total number of attempts].
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Context dependent information understanding (ML-I6):
  How successful the chatbot is to understand information that requires context. It is first given some context in one sentence and then the information in one sentence. For example "I like my cat. His name is Mittens". It is then asked about the information.
  The presented score is a list with format [successful attempts, total number of attempts].
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Context dependent questions understanding (ML-I7):
  How successful the chatbot is to understand questions that requires context. It is first given some information. It is then asked about the information in a context dependent way. For example "I like my cat. What is its name?".
  The presented score is a list with format [successful attempts, total number of attempts].
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Consistency with own information (ML-I13):
  How consistent the chatbot is when asked about its own information. It is then asked about its own information several times and the similarity of the answers are assessed. The correct answer is set to the first answer of the chatbot.
  The presented score is a list with format [successful attempts, total number of attempts].
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Toxicities assessment (ML-P1)
  different kinds of toxicities are assessed. Test passes if the value of the certain test variable is below a certain threshold.
  The presented score is a list with format [successful attempts, total number of attempts].
  The detailed information (optional) is the assessed %-risk of the response being interpreted as the specific toxicity type.

* Repetition avoidance assessment (ML-A6):
  How successful is the chatboyt at avoiding unnecessary repeating of sentences or words. Assessed using using Ngrams. It is created by taking the mean of the N-gram order for N = 2:max times the amount of times that N-gram occurs, mean(order*occurrences of that order) for order 2:max. A 2-gram occurring six times adds the same as a 6-gram occurring 2 times. The test passes if the stuttering value is below a certain threshold.
  The presented score is a list with format [successful attempts, total number of attempts].
  The detailed value is the assessed stuttering value.

* Repeated questions avoidance assessment (ML-A7)
   Assesment if the chatbot repeats any question several times.
   The presented score is a list with format [successful attempts, total number of attempts].

** The whole conversation is not brought as input, since chatbots have varying sizes of input they are able to handle. Instead, a function is implemented to bring the last sentences consisting a maximum of X tokens, in order to handle this.

# 3 Test framework
## 3.1 Framework contents
### 3.1.1 files
* main.py                   - contains the main method and central functions in the test architecture.
* test_functions.py         - contains the functions which are used to analyse the test responses.
* util_functions.py         - contains utility functions which can be of help anywhere in the code.
* config.py                 - configuration document. Contains all settings. Only document from where a user should change the code.
* testset_database.py       - contains the dataset of all tests which are used.
* validation_of_metrics.py  - only used when validating accuracy and finding thresholds.

### 3.1.2 external models
* BERT-SQuAD
  * Input: A string with information, a string with some question about the information.
  * Output: The answer to the question. (string)
* SentenceTransformer
  * Input: Two strings.
  * Output: Semantic similarity between the two strings (float between 0 and 1).
* Detoxifyer
  * input: Some string.
  * Output: The toxicity of the string based on seven measurements:
    * toxicity        (float between 0 and 1)
    * severe toxicity (float between 0 and 1)
    * obscene         (float between 0 and 1)
    * identity attack (float between 0 and 1)
    * insult          (float between 0 and 1)
    * threat          (float between 0 and 1)
    * sexual_explicit (float between 0 and 1)
* Sentence-BERT
  * input: Two strings
  * output: The coherence of the second string given the first string (float between around -12 and 12).

## 3.2 Test architecture

1. The function init_tests() is called. It allocates which tests should be performed and where, based on the configurations. The output is the test allocation list test_ids and the corresponding test sets in test_sets.
2. The function generate_conversation() is called, which generates the conversation and injects tests based on the test_ids array. The output is the generated conversation.
3. The function analyze_conversation() is called, which assesses the answers. The output is a data frame with results for each test. The summary of the entire run is added in the bottom and exported to the results summary document.
4. The test is repeated until a specified amount if runs are finished. The summary of the entire results summary document is added in the bottom.

![Image of test architecture](https://github.com/JoohanBengtsson/Emely-testing.git/img/Architecture.png)

## 3.3 Test case categories

There are four test case categories used.
* Question-Answer
* Question
* N-grams
* Sentence-BERT
* Detoxifyer
* Simple check

### 3.3.1 Question-Answer
In Question-Answer test, the tester first gives some information and then asks a question and demands an answer about the information. If the answer is correct, the test has passed. There are two types of questions, one for open questions and one for closed (yes/no).

#### 3.3.1.1 Open questions
The algorithm for assessing open questions is shown below:
1. To understand whether the answer is correct, the answer is inserted into the BERT-SQuAD model along with a question specified in the test dataset to get the right information.
2. The output is a more condensed answer which is inserted into the SentenceTransformers model along with the correct answer specified in the dataset.
3. The output is the semantic similarity between the two answers, and is an interpretation of the correctness of the answer.
4. If the value is above a certain threshold, the test has passed.

![Image of QA model for open questions](https://github.com/JoohanBengtsson/Emely-testing.git/img/QA-model.png)

#### 3.3.1.2 Closed questions
The algorithm for assessing closed questions is shown below:
1. The answer is scanned for the formulations "no", "don't" and "do not". If the answer contains any of these, the answer is labeled False (No). Otherwise, the answer is set to True (yes).
2. If the answer is the same as the correct one in the dataset, the test has passed.

### 3.3.2 Question
The Question test is the same as Question-Answer, with the exception that there is no predefined correct answer. Instead, the bot itself is asked for some information which is then considered the correct answer. The bot can thereafter be asked the same question again and the correctness of that answer can be assessed based on the previous answer.

### 3.3.3 N-grams
N-grams testing makes use of the structure of the answer instead of analysing the contents of the answer. It can therefore be used to assess if there are any trends or abnormalities in the structure of the dataset.

### 3.3.4 Sentence-BERT
Sentence-BERT test method makes direct use of the Sentence-BERT model to assess the coherence of the answer.

### 3.3.5 Detoxifyer
Detoxifyer test method makes direct use of the Detoxifyer model to assess the suitability of the answer.

### 3.3.5 Simple check
Simple check test method tests a simple property of the conversation.


# 4. Instructions

This section aims to guide the reader on how to clone, run  and use the script.

## 4.1 Clone and setup repository locally

It is a precondition that the user has the following parts ready on the computer:
* Python, with pip, virtual environments and an IDE

Whenever the preconditions are fulfilled, the user may start using the script. In order to do so, the user needs to clone the repo. That is done either through the command prompt or through a tool for source control. Here, the command prompt-way is demonstrated:

1. Start the command prompt
2. Navigate to wherever you want to place the local repository
3. Make the command:

``git clone https://github.com/JoohanBengtsson/Emely-testing.git``

4. To setup the virtual environment, follow the following steps:

* Navigate into the directory: ``cd Emely-testing``  
and then create the environment: ``python -m venv env``
* Activate the environment:
``env\Scripts\activate``
* Install the required packages: ``pip install -r requirements.txt``
* If your computer has a GPU and you have a **Windows**-computer, the following line will reinstall Pytorch with support for using the GPU: ``pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html``, otherwise visit https://pytorch.org/get-started/locally/ for more information on how to install Pytorch with GPU-support.
* Start the script using the recently setup environment within your preferred IDE.
5. Prior to running the script, some setting variables need explanation, which can be found in *3.2 Setting variables*. Some variables marked with * need to be specified prior to running the script. After specifying these values, the script is ready to be run, at least with the on forehand implemented conversational agents. If the agent that should be run is not implemented, please check *3.3 Implementation details of a new chatter*.

## 4.2 Setting variables

In this subsection, all the current setting variables will be explained.

| Variable name                | Variable description                                                                                                                      |
|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| **GENERAL**                      |                                          |
| max_runs*                     | Decides how many conversations that should be done in total |
| is_load_conversation         | True = Load from load_document. False = Generate text from the chatters specified below. |
| is_save_conversation         | True = Save conversation in folder save_documents |
| is_analyze_conversation      | True = if the program shall print metrics to .xlsx. False = If it is not necessary |
| **GENERATE**                     |                                          |
| conversation_length*          | Decides how many responses the two chatters will contribute with |
| init_conv_randomly           | True if the conversation shall start randomly using external tools. If chatter is set to |
|                              | either 'predefined' or 'user', this is automatically set to False |
| chatters*                     | Chatter 1-profile is on index 0, chatter 2-profile is on index 1. Could be either one of ['emely', 'blenderbot', 'user', 'predefined'].
|                              | 'emely' assigns Emely to that chatter.
|                              |'blenderbot' assigns Blenderbot to that chatter.  |
|                              |'user' lets the user specify the answers. |
|                              | 'predefined' loops over the conversation below in the two arrays predefined_conv_chatter1 and predefined_conv_chatter2. Two standard conversation arrays setup for enabling hard-coded strings and conversations and try out the metrics. |
|          convarray_init                     |Array for storing the conversation. Can be initialized as ["Hey", "Hey"] etc |
| predefined_conv_chatter1 / predefined_conv_chatter2      | Predefined conversations as per chatter (the number corresponds to the specific chatter, where number 2 is the one being tested).                  |
| prev_conv_memory_chatter1 / prev_conv_memory_chatter2     | How many previous sentences in the conversation shall be brought as input to any chatter. Concretely = conversation memory per chatter, if it needs to be controlled for a chatter. |
| **AFFECTIVE TEXT GENERATION**    |                                          |
| affect                       | Affect for text generation. ['fear', 'joy', 'anger', 'sadness', 'anticipation', 'disgust', 'surprise', 'trust']                     |
| knob                         | Amplitude for text generation. 0 to 100  |
| topic                        | Topic for text generation. ['legal','military','monsters','politics','positive_words', 'religion','science','space','technology'] |
| **SAVE AND LOAD**                |                                          |
| save_conv_folder             | The folder in which the conversations are saved |
| load_conv_folder             | The folder in which the conversations are contained |
| save_analysis_name           | The name of the analysis folder          |
| **ANALYSIS***                    |                                          |
| show_interpret               | Interpretations, whether the result should show how the script interprets the analyzed responses. True or False (ToF)                         |
| show_detailed                | Detailed results, that is the scores received from the used ML-models. ToF                        |
| show_binary                  | Binary results - whether an answer's detailed value passes or not a threshold value. ToF                          |
| is_analyze_question_freq     | Question frequency - ToF                       |
| is_MLP1TC1                   | Toxicity analysis using a toxicity analysis tool. ToF                                 |
| is_MLI2TC1                   | Context coherence, wrt the whole conversation. ToF                        |
| is_MLI3TC1                   | Sentence coherence, wrt last sentence. ToF                       |
| is_MLA6TC1                   | Stuttering. ToF                               |
| p_MLI1TC1                    | Remember information for a certain amount of time. Any float value in the range [0, 1]. |
| p_MLI4TC1                    | Understand differently formulated information. [0, 1] |
| p_MLI5TC1                    | Understand differently formulated questions. [0, 1] |
| p_MLI6TC1                    | Understand information based on context. [0, 1]  |
| p_MLI7TC1                    | Understand questions based on context. [0, 1]    |
| p_MLI13TC1                   | Consistency with own information. [0, 1]         |
| p_MLU3TC1                    | Understands questions with randomly inserted typing mistakes. [0, 1] |
| p_MLU4TC1                    | Understands questions with randomly swapped word order. [0, 1] |
| p_MLU5TC1                    | Understands questions with randomly masked words. [0, 1] |
| p_MLU6TC1                    | Understands questions with some words swapped for randomly chosen words. [0, 1] |
| **AUXILIARY ANALYSIS VARIABLES** |                                          |
| maxsets_MLI1TC1              | How many different data sets may be used for MLI1TC1. Depends on how many QA-data sets that are available, but the value should be in the range [1, 5]. |
| maxsets_MLI4TC1              | How many different data sets may be used for MLI4TC1. [1, 5] |
| maxsets_MLI5TC1              | How many different data sets may be used for MLI5TC1. [1, 5] |
| maxsets_MLI6TC1              | How many different data sets may be used for MLI6TC1.|
| maxsets_MLI7TC1              | How many different data sets may be used for MLI7TC1.|
| maxsets_MLI13TC1             | How many different data sets may be used for MLI13TC1.|
| maxsets_MLU3TC1              | How many different data sets may be used for MLU3TC1. [1, 5] |
| maxsets_MLU4TC1              | How many different data sets may be used for MLU4TC1. [1, 5] |
| maxsets_MLU5TC1              | How many different data sets may be used for MLU5TC1. [1, 5] |
| maxsets_MLU6TC1              | How many different data sets may be used for MLU6TC1. [1, 5] |
| maxlength_MLI1TC1            | Maximum amount of rounds that the ML1TC1 can wait for to test long term memory. Value should be in the range [1, conversation_length - 1]. |
| array_ux_test_cases          | The array consisting of the test cases related to understanding, in which it is relevant to store the results and map the results to different levels of inserted errors.              |
| threshold_sem_sim_tests      | The threshold used for the QA-models using semantic similarity. The threshold level is the threshold used for assessing the values received from the ML model. |
| **DATA AUGMENTATION**            |                                          |
| p_synonym                    | Probability of switching to a synonym    |
| n_aug                        | Number of times each test set should be augmented by switching some words with synonyms |


## 3.3 Implementation details of a new chatter

The script is currently offering four chatter profiles for generating the conversation, namely:
* Emely
* Blenderbot 400M
* User - enabling the user to interact with the other chatter
* Predefined sentences located in predefined_conv_chatter1 and predefined_conv_chatter2

These four chatter types have been implemented as classes, in the **Classes-section** within the code. These classes all have the method **get_response(self, conv_array)** in common. This method is what mainly differs between the chatter types on how text is generated. Hence, if the user wants to add another chatbot, it only takes to go through three steps:
1. Define a class and its own get_response(self, conv_array)-method, more specifically how to use the source for generating input.

```
class {BOT_CLASS_NAME}:
    def __init__(self):

    def get_response(self, convarray):
        response = {CODE FOR GENERATING RESPONSE FROM THE SOURCE}
        return response
```

2. Go to the method assign_model() and add:

```python
elif chatter_profile == {NAME_OF_BOT}:
    return {BOT_CLASS_NAME}()
```

3. In the config-script, go to the attribute array **chatters** and change either one or both of the indices 0 and 1 to {NAME_OF_BOT}
```python
chatters = ['emely', 'blenderbot']  # Chatter 1-profile is on index 0, chatter 2-profile is on index 1.
# Could be either one of ['emely', 'blenderbot', 'user', 'predefined']
```

Voilà: the framework is setup to include your robot as well and you are ready to test your robot.

# 4. Software Requirements Specification

Context diagram: The ML model as part of the overall system.

## 4.1 User stories

- Emely as the recruiter
- Emely as the fika buddy

## 4.2 System Requirements

- Performance requirements (throughput, inference time, ..)
- Input filtering (requirements on rule-based filtering of toxic content)
- User experience

## 4.3 ML Model Requirements

- No stuttering
- Reasonable memory
- Non-toxic
