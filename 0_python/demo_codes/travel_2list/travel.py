#!/usr/bin/python

question=["name","sex","age"]
answer=["zzw","male","27"]

for a,b in zip(question,answer):
    print("What's your {0}? It's {1}!".format(a,b))
