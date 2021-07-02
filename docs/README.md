# Introduction

This project aims to provide an open-source test script that could be used to test any chatbot. The script is setup so that it is easy to add a new chatbot or text generator in order to assess it, more details about this can be found in **Implementation details**. 

The script will produce a conversation between two chatters and then assess the conversation with regards to some quality aspects. The quality aspects will be defined below in the **Software Requirements Specification** chapter. These quality aspects will be assessed and then written to a .xlsx-file, for the user to use for further assessment of the chatbot.

# Implementation details

The script is currently offering four chatter types, namely: 
* Emely
* Blenderbot 400M
* User - enabling the user to interact with the other chatter
* Predefined sentences located in prefined_conv_chatter1 and predfined_conv_chatter2

These four chatter types have been implemented as classes, in the **Classes-section** within the code. These classes all have the method **get_response(self, conv_array)** in common. This method is what mainly differs between the chatter types on how text is generated. Hence, if the user wants to add another chatbot, it only takes to three steps:
1. Define a class and its own get_response(self, conv_array)-method, more specifically how to use the source for generating input.
2. Go to the method assign_model() and add 
``elif chatter_profile == {NAME OF BOT}:
        return {Class name of newly created class}(nbr)``

# Software Requirements Specification

Context diagram: The ML model as part of the overall system.

## User stories

- Emely as the recruiter
- Emely as the fika buddy

## System Requirements

- Performance requirements (throughput, inference time, ..)
- Input filtering (requirements on rule-based filtering of toxic content)
- User experience

## ML Model Requirements

- No stuttering
- Reasonable memory
- Non-toxic
