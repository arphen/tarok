Feature: Scoring
  Scoring follows official Slovenian Tarok point tables.

  Scenario: Contract base values match rules
    Then contract "three" should have base value 10
    And contract "two" should have base value 20
    And contract "one" should have base value 30
    And contract "solo_three" should have base value 20
    And contract "solo_two" should have base value 30
    And contract "solo_one" should have base value 40
    And contract "solo" should have base value 50

  Scenario: Winning by 1 point adds 1 to the base
    Given the declarer scored 36 card points playing "three"
    Then the game score should be 11

  Scenario: Losing by 3 points subtracts from the base
    Given the declarer scored 32 card points playing "three"
    Then the game score should be -13

  Scenario: Scores are zero-sum in 2v2
    Given a completed game with 4 random players
    Then the sum of all scores should be 0

  Scenario: Silent trula bonus is 10
    Given the declarer team collected trula silently
    Then the trula bonus should be 10

  Scenario: Announced trula bonus is 20
    Given the declarer team announced and collected trula
    Then the trula bonus should be 20

  Scenario: Failed announced trula is -20
    Given the declarer team announced trula but opponents collected it
    Then the trula bonus should be -20

  Scenario: Silent pagat ultimo bonus is 25
    Given the declarer team won pagat ultimo silently
    Then the pagat bonus should be 25

  Scenario: Announced pagat ultimo bonus is 50
    Given the declarer team announced and won pagat ultimo
    Then the pagat bonus should be 50

  Scenario: Kontra doubles the base game score
    Given the declarer scored 36 card points playing "three"
    And the opponents kontra the game
    Then the game score should be 22

  Scenario: Re quadruples the base game score
    Given the declarer scored 36 card points playing "three"
    And the declarers re the game
    Then the game score should be 44

  Scenario: Kontra on announced trula doubles the bonus
    Given the declarer team announced and collected trula
    And the opponents kontra trula
    Then the trula bonus should be 40

  Scenario: 2v2 declarer and partner receive identical scores
    Given the declarer scored 36 card points playing "three"
    Then the declarer and partner should have the same score

  Scenario: 2v2 opponents receive identical scores
    Given the declarer scored 36 card points playing "three"
    Then both opponents should have the same score

  Scenario: 2v2 scores are symmetric
    Given the declarer scored 36 card points playing "three"
    Then the declarer score should equal the negated opponent score
