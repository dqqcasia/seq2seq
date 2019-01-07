import codecs
import sys

with codecs.open(sys.argv[1], 'r', 'utf8') as f:
    for line in f:
        if line != '':

            #print line.replace('...', '……'),
            newline = ''
            for word in line.strip().split():

                newword = word.replace('@@','')

                if word == '...':
                    newword  = '……'
                #if word == '……':
                #    newword = '...'
                if word == '?':
                    newword = '？'
                if word == '!':
                    newword = '！'
                if word == ',':
                    newword = '，'
                if word == '。':
                    newword = '。'
                if word == '。':
                    newword = '。'
                if word == '(':
                    newword = '（'
                if word == ')':
                    newword = '）'

                if word == '…':
                    newword = '……'
                if word == '......':
                    newword = '……'
                if word == ':':
                    newword = '：'

                if word == ';':
                    newword = '；'

                # the consecutive number should not be seperated but the consecutive alpha should be seperated
                if word.isdigit():
                    newline+=newword
                else:
                    newline +=' '
                    newline += newword

            print(newline)
