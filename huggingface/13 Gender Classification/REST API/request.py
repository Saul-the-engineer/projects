# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 23:06:05 2022

@author: saulg
"""

import requests

KERAS_REST_API_URL = "http://localhost:5000/predict"

text1 = """thisiswarda, , Photography and teaching credits:""" 
text1 = """andrew_watson21, Life fast, pet dogs., """ 
text1 = """graceeluke, A child of God, Crunchy Leaves"""
text1 = """daniellingieldjarvis, vot.utah.gov/, She doesn't even go here!"""
text1 = """Tamahra_, ATSU AuD '24, We finally got our Christmas tree and decor up!
Mason was the best little helper!! Christmas ready at Casa de Weiss. Elle at the
end is her ALWAYS asking for love and cuddles!"""

text1 = """Dani, Live, Laugh, Love, Merry Christmas Everyone."""
text1 = """Dan, Live, Laugh, Love, Merry Christmas Everyone."""
text1 = """Danny, Live, Laugh, Love, Merry Christmas Everyone."""

text1 = "BigGay, Cute, sweet and even funny, Staring at her and thinking, How did a girl like her end up with a girl like me."
text = "Big_Gay, Cute, sweet and even funny, Staring at her and thinking, How did a girl like her end up with a girl like me."

# Andrew Tate
text1 = """Cobratate, Light-Heavyweight Kickboxing World Champion. Escape the 
Matrix, Mastery is a funny thing. It's almost as if, on a long enough time 
scale, losing simply isn't an option. Such is the way of Wudan"""

# Elon Musk
text1 = """elonmusk, , twitter deal temporarily on hold pending details 
supporting calculation that spam/fake accounts"""

# Alexandria Ocasio-Cortez
text1 = """AOC, US Representative,NY-14 (BX & Queens). In a modern, moral, & 
wealthy society, no American should be too poor to live. 💯% People-Funded, 
no lobbyist💰. She/her.,I see people are rushing out to fill up their cars 
for this hurricane at the gas station This wouldn't be an issue if they 
had electric cars.If the power is out for a week how are they going to get gas?
We need to start planning ahead and moving forward"""


payload = {
    "text": text
    }

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensure the request was successful
if r["success"]:
    # loop over the predictions and display them
    for (i, result) in enumerate(r["predictions"]):
        label = max(r["predictions"][0], key=r["predictions"][0].get)
        probability = result[label]
        print(f"Item {i + 1}: {label}; Probability: {probability:.3f}.")

# otherwise, the request failed
else:
    print("Request failed")