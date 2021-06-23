from nltk import ngrams
from collections import Counter

def stutter(convarray):
    n = len(convarray)
    # Preallocate
    stutterval = 0
    maxkeys = [None] * (n - 1)
    maxvals = [None] * (n - 1)

    # Find the most repeated gram of each length
    for order in range(1, n):
        grams = Counter(ngrams(convarray, order))
        maxkeys[order - 1] = max(grams, key=grams.get)
        maxvals[order - 1] = max(grams.values())

    # Evaluate stutter
    # If length is less than 3, no stutter
    if len(convarray) < 3:
        stutterval = 0
        return maxvals, maxkeys, stutterval

    # Amount of stutter is mean amount of stutter words for each gram
    stutterval = sum([(maxvals[i]-1)*(i+1)/n for i in range(n-1)])
    return maxvals, maxkeys, stutterval



if __name__ == "__main__":
    # scentence = ["Hello,", "my", "name" ,"is", "Harald"]
    sentence = 'Are you going go to going to go to me'
    convarray = sentence.split()
    maxvals, maxkeys, stutterval = stutter(convarray)
    print("Stutter value: " + str(stutterval))



# n = 6
# sixgrams = ngrams(convarray, n)
# sixgramscount = Counter(sixgrams)
# maxkey = max(sixgramscount)
# maxval = max(sixgramscount.values())

# print(["Key: " , maxkey , ", val: " , maxval])

