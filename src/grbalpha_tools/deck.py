from collections import namedtuple
from dataclasses import dataclass
from collections.abc import MutableSequence

#Card = namedtuple('Card', ['color', 'value'])

@dataclass(frozen=True)
class Card:
    color: str
    value: str

@dataclass(frozen=True)
class ToggledCard(Card):
    is_revealed: bool = False


class Deck(MutableSequence):
    values = [str(n) for n in range(2,11)] + list('JQKA')
    colors = ['diamond','spade','club','heart']

    def __init__(self,card_factory=Card):
        self._cards = [ # '_' ... znamena nesahat
            card_factory(color,value)
            for color in Deck.colors
            for value in Deck.values
        ]
    
    def __len__(self):
        return len(self._cards)

    def __getitem__(self,position):
        return self._cards[position]
    
    def __delitem__(self, i):
        del self._cards[i]
    
    def insert(self, i, o):
        self._cards.insert(i, o)
    
    def __setitem__(self, i, o):
        self._cards[i] = o


    