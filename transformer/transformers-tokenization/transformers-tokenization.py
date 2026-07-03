import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        self.word_to_id: Dict[str, int] = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        vocab = set()
        [vocab.update(text.lower().split()) for text in texts]

        vocab = sorted(vocab)
        curr_id = 4
        for voc in vocab:
            self.word_to_id[voc] = curr_id
            curr_id += 1

        # Обратный словарь
        self.id_to_word = {
            idx: word
            for word, idx in self.word_to_id.items()
        }
        self.vocab_size = 4 + len(vocab)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        encode_text = []
        if text:
            text = text.lower()
            words_split = text.split(" ")
    
            
            
            # for word in words_split:
            #     encode_text.udate(word)
            
            
            
            for word in words_split:
                if word in self.word_to_id:
                    encode_text.append(self.word_to_id[word])
                else:
                    encode_text.append(self.word_to_id[self.unk_token])
    
        return encode_text
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        new_string = []

        for id in ids:
            if id < self.vocab_size:
                new_string.append(self.id_to_word[id])
            else:
                new_string.append(self.id_to_word[1])

        return " ".join(new_string)
        
