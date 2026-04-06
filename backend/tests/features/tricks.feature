Feature: Trick Taking Rules
  Slovenian Tarok follows strict suit-following and tarok-playing rules.

  Scenario: Must follow lead suit when possible
    Given a player has hearts and clubs in hand
    When a heart is led
    Then the player can only play hearts

  Scenario: Must play tarok if cannot follow suit
    Given a player has only clubs and taroks in hand
    When a heart is led
    Then the player can only play taroks

  Scenario: Can play anything if no suit and no tarok
    Given a player has only clubs and diamonds in hand
    When a heart is led
    Then the player can play any card

  Scenario: Must follow tarok with tarok
    Given a player has taroks and suit cards in hand
    When a tarok is led
    Then the player can only play taroks

  Scenario: Higher tarok beats lower tarok
    Given tarok XV is played against tarok X
    Then tarok XV should win

  Scenario: Any tarok beats any suit card
    Given tarok I is played against the king of hearts
    Then the tarok should win

  Scenario: Lead suit wins against off-suit
    Given king of hearts leads and king of clubs follows
    Then king of hearts should win

  Scenario: After 12 tricks the phase is scoring
    Given a game in trick play phase with 4 random players
    When all 12 tricks are played
    Then the game phase should be "scoring"
