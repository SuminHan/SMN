from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
    'The last document?',    
]
vect = CountVectorizer()
vect.fit(corpus)
print(vect.transform(corpus).toarray())


from sklearn.feature_extraction.text import TfidfVectorizer
tfidv = TfidfVectorizer().fit(corpus)
rtfidv = {v: k for k, v in tfidv.vocabulary_.items()}

import numpy as np
np.set_printoptions(precision=3)

print(corpus)
print([rtfidv[i] for i in range(len(rtfidv))])

foo = tfidv.transform(corpus).toarray()
mask = foo != 0
print(foo)
print(np.reciprocal(foo))
foo[mask] = np.reciprocal(foo[mask])
print(foo)
