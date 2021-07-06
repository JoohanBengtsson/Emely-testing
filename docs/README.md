# 1. Introduction

This project aims to provide an open-source test framework that could be used to test any chatbot. The script is setup so that it is easy to add a new chatbot or text generator in order to assess it, more details about this can be found in **Implementation details**. 

The script will produce a conversation between two chatters, hereafter called chatter1 respectively chatter2, and then assess the conversation with regards to some predefined quality aspects. The quality aspects will be defined below in the **Software Requirements Specification** chapter. These quality aspects will be assessed and then written to a .xlsx-file, for the user to use for further assessment of the chatbot.

# 2. Metrics evaluated by this testing framework
* Stuttering - using Ngrams
* Repeated questions - is any question repeated several times
* Coherence assessment - with regards to the last sentence and the whole last conversation** respectively, is the response produced by a chatter coherent.
* Toxicities - different kinds of toxicities are assessed. The presented numbers are the %-risk of the response being interpreted as the specific toxicity type.

** The whole conversation is not brought as input, since chatbots have varying sizes of input they are able to handle. Instead, a function is implemented to bring the last sentences consisting a maximum of X tokens, in order to handle this.

# 3. Instructions

## 3.1 Run setup

It is a precondition that the user has the following parts ready on the computer:
* Python, with pip, virtual environments and an IDE
* Emely (if the user wants to test Emely)

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
* If your computer has a GPU and you have a **Windows**-computer, the following line will reinstall Pytorch with support for using the GPU: ``pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html``
* Start the script using the recently setup environment within your preferred IDE.
5. Prior to running the script, some setting variables need explanation, which can be found in *3.2 Setting variables*

## 3.2 Setting variables

In this subsection, all the current setting variables will be explained.

* present_metrics  # True: if the user wants the script to do the whole analysis and write the results to a .xlsx-file. False: if the whole analysis and report is not needed, mostly applicable whenever the user wants to work on the code and have a reduced script time.
* init_conv_randomly  # True: if the conversation should be randomly initiated. That is, chatter1 starts with a 'Hey' upon which chatter2 uses a random generator to start a conversation. False: if a random start is not wished for.
* convarray = []  # The array that will store the whole array and are then subject for the analysis. Not really subject to any changes prior to running the script.
* conversation_length  # How many lines each will the two chatters have during the conversation. Note: increasing this length will obviously increase the total time the script takes.
* load_conversation  # True: the chatters specified in the below mentioned array *chatters* generates the conversation. False: the conversation is generated from the document whose name should be assigned to the variable *load_document*
* load_document  # If *generate_conversation* equals False, then the script will try to read the document with the name (and path) assigned to this variable. That is, for whatever text document the user wants to read into the script, specify the name (and path) of the document to this variable and set *generate_conversation* = False.
* save_conversation  # True: the script saves the conversation to the document specified in *save_document*. False: the script is not saved.
* save_document  # The file name of the script in which the conversation is saved if *save_conversation* equals True.
* chatters = [{chatter1}, {chatter2}]  # On these two indices in the array, the two chatters are specified. Here the user may choose between the currently implemented chatter-profiles.
* predefined_conv_chatter1 and predefined_conv_chatter2  # If chatterprofile predefined is chosen for index 0, predefined_conv_chatter1 are the predefined sentences that will be loaded. Vice versa goes for chatter2.
* prev_conv_memory_chatter1 and prev_conv_memory_chatter2  # Regulates how many previous rounds of the conversation that will be brought as input to the chatter1 respectively chatter2. Comes in handy when it is known that one chatbot only has a restricted capability in reading input.
* is_affect  # True: whenever init_conv_randomly is equal to True, this means that the generation will be done using an affective text generator, where the affection can be specified prior to running, which is further described below.
* affect  # If *is_affect* == True, it is here where the user shall specify the specific affect that should be used for the affective text generation. Can choose among ['fear', 'joy', 'anger', 'sadness', 'anticipation', 'disgust','surprise', 'trust']
* knob  # An amplitude scale between 0 and 100, how much of the specific affect should be present in the generated sentence.
* topic  # The specific topic that the generated text should be about.


## 3.3 Implementation details of a new chatter

The script is currently offering four chatter types, namely: 
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

3. Go to the attribute array **chatters** and change either one or both of the indices 0 and 1 to {NAME_OF_BOT} 
```python
chatters = ['emely', 'blenderbot']  # Chatter 1-profile is on index 0, chatter 2-profile is on index 1.
# Could be either one of ['emely', 'blenderbot', 'user', 'predefined']
```

Voilà: the framework is setup to include your robot as well.

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
