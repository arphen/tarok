Feature: Card Deck
  The Slovenian Tarok deck has 54 cards with specific properties.

  Scenario: Deck has exactly 54 cards
    Given a fresh tarok deck
    Then the deck should contain 54 cards

  Scenario: Deck has 22 taroks
    Given a fresh tarok deck
    Then the deck should contain 22 tarok cards

  Scenario: Deck has 32 suit cards
    Given a fresh tarok deck
    Then the deck should contain 32 suit cards

  Scenario: Deck has 4 suits with 8 cards each
    Given a fresh tarok deck
    Then each suit should have 8 cards

  Scenario: All cards in the deck are unique
    Given a fresh tarok deck
    Then all cards should be unique

  Scenario: Total raw card points equal 106
    Given a fresh tarok deck
    Then the total raw card points should equal 106

  Scenario: Total counted card points equal 70
    Given a fresh tarok deck
    Then the total counted card points should equal 70

  Scenario: Trula cards are worth 5 points each
    Given a fresh tarok deck
    Then Pagat should be worth 5 points
    And Mond should be worth 5 points
    And Škis should be worth 5 points

  Scenario: Kings are worth 5 points
    Given a fresh tarok deck
    Then all kings should be worth 5 points

  Scenario: Queens are worth 4 points
    Given a fresh tarok deck
    Then all queens should be worth 4 points

  Scenario: Knights are worth 3 points
    Given a fresh tarok deck
    Then all knights should be worth 3 points

  Scenario: Jacks are worth 2 points
    Given a fresh tarok deck
    Then all jacks should be worth 2 points

  Scenario: Pip cards and plain taroks are worth 1 point
    Given a fresh tarok deck
    Then all pip cards should be worth 1 point
    And plain taroks should be worth 1 point
