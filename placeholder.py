import json
from lxml import html
import requests


def getCommonWords(url):
    # incomplete    
    # map(requests.get, url.keys())
    p, t, w = {}, {}, {}
    for pos in ['verb', 'noun', 'adj']:
        p[pos] = requests.get(url[pos])
        t[pos] = html.fromstring(p[pos].content)
        w[pos] = t[pos].xpath('//a[@target="_blank"]/text()')
    return w

    
if __name__ == '__main__':

    w = {}
    # where I got most common words from
    # url = {'verb': http://www.talkenglish.com/vocabulary/top-1000-verbs.aspx,
    #        'noun': http://www.talkenglish.com/vocabulary/top-1500-nouns.aspx,
    #        'adj': http://www.talkenglish.com/vocabulary/top-500-adjectives.aspx}
    # w = getCommonWords(url)
    
    with open('commonWords.json', 'r') as f: w = json.load(f)
    # print w['adj'][:20]

    s = {'noun': ['what is your favourite <word>?',
                  'tell me something about <word>.',
                  'oh no, a <word>!',
                  'a <word> is so delicious.',
                  "what's the big deal about <word>?",
                  'do you like <word>?',
                  "Your friend's <word> looks so good.",
                  "I'm sure you're not good at <word>.",
                  "I like my <word> strong."],
         
         'verb': ['I really want to <word>.',
                  'Would you like to <word> with me?',
                  "Let's all go <word>.",
                  "That sounds like a <word> idea."],
         
         'adj': ['I think black people are so <word>.',
                 'I think women are so <word>.',
                 'Sometimes you get very <word>, did you know that?']}
    