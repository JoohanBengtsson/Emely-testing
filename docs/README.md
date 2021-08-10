# 1. Introduction

This project aims to provide an open-source test framework that could be used to test any chatbot. The script is setup so that it is easy to add a new chatbot or text generator in order to assess it, more details about this can be found in **Implementation details**. 

The script will produce a conversation between two chatters, hereafter called chatter1 respectively chatter2, and then assess the conversation with regards to some predefined quality aspects. The quality aspects will be defined below in the **Software Requirements Specification** chapter. These quality aspects will be assessed and then written to a .xlsx-file, for the user to use for further assessment of the chatbot.

# 2. Metrics evaluated by this testing framework
* Stuttering - using Ngrams.
* Repeated questions - is any question repeated several times
* Coherence assessment - with regards to the last sentence and the whole last conversation** respectively, is the response produced by a chatter coherent.
* Toxicities - different kinds of toxicities are assessed. The presented numbers are the %-risk of the response being interpreted as the specific toxicity type.

** The whole conversation is not brought as input, since chatbots have varying sizes of input they are able to handle. Instead, a function is implemented to bring the last sentences consisting a maximum of X tokens, in order to handle this.

# 3. Instructions

This section aims to guide the reader on how to clone, run  and use the script.

## 3.1 Clone and setup repository locally

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

## 3.2 Setting variables

In this subsection, all the current setting variables will be explained.

| Variable name                | Variable description                     |
|------------------------------|------------------------------------------|
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
