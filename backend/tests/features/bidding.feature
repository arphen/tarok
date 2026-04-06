Feature: Bidding
  Players bid ascending contracts or pass.

  Scenario: First player can bid any contract or pass
    Given a game in bidding phase
    When it is the first player's turn
    Then the legal bids should include pass and all contracts

  Scenario: Later bids must be higher than current highest
    Given a game where "three" has been bid
    When the next player considers bidding
    Then only contracts higher than "three" should be available

  Scenario: All pass triggers klop
    Given a game in bidding phase
    When all four players pass
    Then the game should enter klop mode

  Scenario: Single remaining bidder wins the contract
    Given a game where 3 players passed and 1 bid "three"
    Then that player should be the declarer
    And the contract should be "three"
