Feature: Dealing Cards
  Cards are dealt to 4 players plus a talon.

  Scenario: Deal gives 12 cards to each player and 6 to talon
    Given a new game with 4 players
    When cards are dealt
    Then each player should have 12 cards
    And the talon should have 6 cards
    And no cards should be duplicated across hands and talon

  Scenario: All 54 cards are accounted for after dealing
    Given a new game with 4 players
    When cards are dealt
    Then the total number of cards in hands and talon should be 54

  Scenario: Phase transitions to bidding after deal
    Given a new game with 4 players
    When cards are dealt
    Then the game phase should be "bidding"
