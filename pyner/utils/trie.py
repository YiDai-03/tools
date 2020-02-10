import collections
class TrieNode:
    # Initialize your data structure here.
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False
        self.word_num = -1

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, num):
        
        current = self.root
        for letter in word:
            if (letter == 0):
                break
            current = current.children[letter]
        current.is_word = True
        current.word_num = num

    def search(self, word):
        current = self.root
        for letter in word:
            if (letter == 0 ):
                break
            current = current.children.get(letter)

            if current is None:
                return False, -1
        return current.is_word, current.word_num

    def startsWith(self, prefix):
        current = self.root
        for letter in prefix:
            current = current.children.get(letter)
            if current is None:
                return False
        return True


    def enumerateMatch(self, word, space="_", backward=False):
        matched = []
        matched_lens = []
        ## while len(word) > 1 does not keep character itself, while word keed character itself
        while len(word) > 1:

            wd, num = self.search(word)
            if wd:
                matched.append(num)
                matched_lens.append(len(word))
                
            del word[-1]
        return [matched,matched_lens]

