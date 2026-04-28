/// 3-player adapter for StockSkis m6.
///
/// This keeps m6's card-play style while mapping bidding/talon/discard
/// decisions to the 3-player contract/phase model.
use crate::bots::stockskis_m6;
use crate::card::*;
use crate::game_state::*;

fn map_3p_highest_to_4p(highest: Option<Contract>) -> Option<Contract> {
    match highest {
        None => None,
        Some(Contract::Berac) => Some(Contract::Berac),
        Some(Contract::SoloThree) => Some(Contract::Three),
        Some(Contract::SoloTwo) => Some(Contract::Two),
        Some(Contract::SoloOne) => Some(Contract::One),
        _ => None,
    }
}

fn map_4p_bid_to_3p(contract: Contract) -> Option<Contract> {
    match contract {
        Contract::Berac => Some(Contract::Berac),
        Contract::Three => Some(Contract::SoloThree),
        Contract::Two => Some(Contract::SoloTwo),
        Contract::One => Some(Contract::SoloOne),
        Contract::SoloThree => Some(Contract::SoloOne),
        Contract::SoloTwo => Some(Contract::SoloOne),
        Contract::SoloOne => Some(Contract::SoloOne),
        Contract::Solo => Some(Contract::SoloOne),
        _ => None,
    }
}

pub fn evaluate_bid_m6_3p(hand: CardSet, highest_so_far: Option<Contract>) -> Option<Contract> {
    debug_assert_eq!(hand.len(), 16, "3p hand should be 16 cards, got {}", hand.len());

    let highest_4p = map_3p_highest_to_4p(highest_so_far);
    let bid_4p = stockskis_m6::evaluate_bid_m6(hand, highest_4p)?;
    let bid_3p = map_4p_bid_to_3p(bid_4p)?;

    if let Some(highest) = highest_so_far {
        if bid_3p.strength() <= highest.strength() {
            return None;
        }
    }

    Some(bid_3p)
}

pub fn choose_talon_group_m6_3p(groups: &[Vec<Card>], hand: CardSet) -> usize {
    stockskis_m6::choose_talon_group_m6(groups, hand, None)
}

pub fn choose_discards_m6_3p(hand: CardSet, must_discard: usize) -> Vec<Card> {
    stockskis_m6::choose_discards_m6(hand, must_discard, None)
}

pub fn choose_card_m6_3p(hand: CardSet, gs: &GameState, player: u8) -> Card {
    debug_assert_eq!(
        gs.variant,
        Variant::ThreePlayer,
        "stockskis_m6_3p::choose_card_m6_3p called on {:?} state",
        gs.variant
    );
    stockskis_m6::choose_card_m6(hand, gs, player)
}
