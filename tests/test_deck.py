from ultimatni_prirucka.deck import Deck, Card, ToggledCard
from random import shuffle
import pytest
from dataclasses import FrozenInstanceError

def test_create_card():
    Card('color','value')

def test_create_deck():
    Deck()

def test_deck_has_52_cards():
    deck = Deck()
    assert len(deck) == 52

def test_can_be_shuffled():
    deck = Deck()
    shuffle(deck)

def test_look_at_card():
    deck = Deck()
    card = deck[0]
    assert card == Card("diamond","2")

def test_look_at_cards():
    deck = Deck()
    first_three_cards = deck[0:3]

def test_card_is_immutable():
    card = Card('c','v')
    with pytest.raises(FrozenInstanceError):
        card.color = 'c2'

def test_use_toggled_cards():
    deck = Deck(card_factory=ToggledCard)
    assert deck[0].is_revealed == False 

def test_pop_card():
    deck = Deck()
    card = deck.pop()
    assert card == Card(color='heart', value='A')
    assert len(deck) == 51

def test_remove_card():
    deck = Deck()
    card = Card(color='heart', value='A')
    deck.remove(card)
    assert len(deck) == 51
    assert card not in deck

