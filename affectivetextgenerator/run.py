from model import run_pplm_example, resultContainer
from flask_socketio import SocketIO, join_room, emit, send

# Knob =  0.8
# Affect = "joy"
# Topic = "legal"
# Prompt = "There was a"

def generate(prompt, topic, affect, knob):
    knob/=100
    #print("Recieved request\n", "Prompt: ", prompt, "topic: ", topic, "affect: ", affect, "knob: ", knob)
    if prompt == "Enter prefix" or prompt == "":
        return "", False
    #emit('word', {"value": "Generating..."}, broadcast=True)
    result = run_pplm_example(
          affect_weight=1,  # it is the convergence rate of affect loss, don't change it :-p
          knob = knob, # 0-1, play with it as much as you want
          cond_text=prompt,
          num_samples=1,
          bag_of_words=topic,
          bag_of_words_affect=affect,
          length=30,
          stepsize=0.01,
          sample=True,
          num_iterations=3,
          window_length=5,
          gamma=1.5,
          gm_scale=0.95,
          kl_scale=0.01,
          verbosity='mute'
      )
    #print(result)
    return result, True

if __name__ == "__main__":
    generate("You are the", None, "sadness", 100)

# Example generated sentences
#Two years after being shot, a woman says she is still suffering from post-traumatic stress and still has PTSD.
#A man who had been shot dead in the early hours of the morning has died.

# You are a newbie with a very low score who is about to lose it.
# You have no previous achievements and you have zero friends.