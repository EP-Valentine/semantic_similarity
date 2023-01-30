import spacy
nlp = spacy.load("en_core_web_md")


#____ WORKING WITH WORDS (snippet): _________________________________________________________________________________________
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2)) # 0.59 cat/monkey   
print(word3.similarity(word2)) # 0.40 banana/monkey
print(word3.similarity(word1)) # 0.22 banana/cat



#____ WORKING WITH VECTORS (snippet): _________________________________________________________________________________________
vectors = "when we have a series of words to compare with each other: we can use for loops to do so: "
tokens = nlp("cat apple monkey banana")

for token1 in tokens:
    for token2 in tokens:
        tokenprint = f"{token1.text} & {token2.text}"
        print(f"{tokenprint:20}| Sim: {round(token1.similarity(token2),2)}")



#____ WORKING WITH SENTENCES (snippet): _________________________________________________________________________________________
compare = "why is my cat on the car"

sentences = ["where did my dog go",
             "Hello, there is my car",
             "I've lost my car in my car",
             "I'd like my boat back",
             "I will name my dog Diana"]

compare = nlp(compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(compare)
    print(f"{sentence} - {similarity}")



#_______________________________________________________________________________________________________________________________________

'''   
          Task 1: Notes About Similarity
        ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
         • Interestingly, 'cat and monkey' appear more similar due to their animal categorisation than 'banana and monkey';
           despite monkeys being iconically associated with bananas in pop culture and literature.

    OWN EXAMPLE:         
''' 

worda = nlp("comic")
wordb = nlp("novel")
wordc = nlp("art")

print(f"{worda}/{wordb}: {round(worda.similarity(wordb),2)}")   # comic/novel: 0.53
print(f"{wordc}/{wordb}: {round(wordc.similarity(wordb),2)}")   # art/novel: 0.23
print(f"{wordc}/{worda}: {round(wordc.similarity(worda),2)}")   # art/comic: 0.32




'''   
          Task 1: Note About Example.py ran using "en_core_web_md" vs "en_core_web_sm"
        ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
         • When ran using the "en_core_web_md" model, the program runs smoothly, however, when ran using the "en_core_web_sm"
           model, the below UserWarning is displayed:
'''
                # c:\Users\Emily\Dropbox\EP22110004807\Software Engineer Bootcamp\T38\example.py:38: UserWarning: [W007] 
                # The model you're using has no word vectors loaded, so the result of the Doc.similarity method will be
                # based on the tagger, parser and NER, which may not give useful similarity judgements. 
                # This may happen if you're using one of the small models, e.g. `en_core_web_sm`, which don't ship with 
                # word vectors and only use context-sensitive tensors. You can always add your own word vectors, or use 
                # one of the larger models instead if available.                

