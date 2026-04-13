"""Test deterministic trace replay."""
import tarok_engine as te
from tarok.adapters.ai.rust_game_loop import _BID_IDX_TO_RUST, _talon_cards, _build_talon_groups
from tarok.entities.card import DECK

N_GAMES = 100
result = te.run_self_play(N_GAMES, 4, seat_config='bot_v5,bot_v5,bot_v5,bot_v5')
traces = result['traces']
all_match = True

for g in range(N_GAMES):
    trace = traces[g]
    hands = result['initial_hands'][g].tolist()
    talon = result['initial_talon'][g].tolist()
    scores_original = result['scores'][g].tolist()
    contract = int(result['contracts'][g])
    declarer = int(result['declarers'][g])
    dealer = int(trace['dealer'])

    gs = te.RustGameState()
    gs.dealer = dealer
    gs.deal_hands(hands, talon)

    for p, a in trace['bids']:
        r = _BID_IDX_TO_RUST[a] if a < len(_BID_IDX_TO_RUST) else None
        gs.add_bid(p, r)

    gs.contract = contract
    if declarer >= 0:
        gs.declarer = declarer
        gs.set_role(declarer, 0)
        for i in range(4):
            if i != declarer:
                gs.set_role(i, 2)

    if trace['king_call'] is not None:
        kc_player, kc_action = trace['king_call']
        callable_idxs = gs.callable_kings()
        suit_map = {0: 'hearts', 1: 'diamonds', 2: 'clubs', 3: 'spades'}
        chosen_idx = None
        for idx in callable_idxs:
            card = DECK[idx]
            if card.suit and card.suit.value == suit_map.get(kc_action):
                chosen_idx = idx
                break
        if chosen_idx is None and callable_idxs:
            chosen_idx = callable_idxs[0]
        if chosen_idx is not None:
            gs.set_called_king(chosen_idx)
            for p in range(4):
                if p != declarer:
                    hand = gs.hand(p)
                    if chosen_idx in hand:
                        gs.partner = p
                        gs.set_role(p, 1)
                        break

    if trace['talon_pick'] is not None:
        tp_player, tp_group_idx = trace['talon_pick']
        talon_cards_count = _talon_cards(contract)
        talon_idxs = gs.talon()
        groups = _build_talon_groups(talon_idxs, talon_cards_count)
        picked = groups[min(tp_group_idx, len(groups) - 1)]
        for ci in picked:
            gs.add_to_hand(declarer, ci)
            gs.remove_from_talon(ci)
        for ci in trace['put_down']:
            gs.remove_card(declarer, ci)
            gs.add_put_down(ci)

    gs.phase = te.PHASE_TRICK_PLAY
    lead_player = (dealer + 1) % 4
    cards_played = trace['cards_played']
    card_cursor = 0
    for trick_num in range(12):
        gs.start_trick(lead_player)
        for offset in range(4):
            player = (lead_player + offset) % 4
            gs.current_player = player
            tp, ci = cards_played[card_cursor]
            card_cursor += 1
            gs.play_card(player, ci)
        winner, points = gs.finish_trick()
        lead_player = winner
        if contract == 8 and declarer >= 0 and winner == declarer:
            break

    gs.phase = te.PHASE_SCORING
    scores_replay = list(gs.score_game())
    if scores_replay != scores_original:
        print(f"Game {g}: MISMATCH! original={scores_original} replay={scores_replay} contract={contract}")
        all_match = False

print(f"\n{N_GAMES} games tested. All match: {all_match}")
