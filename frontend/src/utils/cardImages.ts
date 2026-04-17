import type { CardData } from '../types/game';

// Mapping of 'card_type-value-suit' to image path.
// Generated from process_cards.py — 37 cards with verified identifications.
const IMAGE_MAP: Record<string, string> = {
  // Taroks (15 of 22)
  'tarok-1-null': '/cards/tarok_01.jpg',
  'tarok-2-null': '/cards/tarok_02.jpg',
  'tarok-3-null': '/cards/tarok_03.jpg',
  'tarok-5-null': '/cards/tarok_05.jpg',
  'tarok-6-null': '/cards/tarok_06.jpg',
  'tarok-7-null': '/cards/tarok_07.jpg',
  'tarok-9-null': '/cards/tarok_09.jpg',
  'tarok-11-null': '/cards/tarok_11.jpg',
  'tarok-12-null': '/cards/tarok_12.jpg',
  'tarok-15-null': '/cards/tarok_15.jpg',
  'tarok-16-null': '/cards/tarok_16.jpg',
  'tarok-17-null': '/cards/tarok_17.jpg',
  'tarok-18-null': '/cards/tarok_18.jpg',
  'tarok-19-null': '/cards/tarok_19.jpg',
  'tarok-20-null': '/cards/tarok_20.jpg',
  // Hearts (6 of 8)
  'suit-1-hearts': '/cards/hearts_pip1.jpg',
  'suit-3-hearts': '/cards/hearts_pip3.jpg',
  'suit-4-hearts': '/cards/hearts_pip4.jpg',
  'suit-5-hearts': '/cards/hearts_jack.jpg',
  'suit-7-hearts': '/cards/hearts_queen.jpg',
  'suit-8-hearts': '/cards/hearts_king.jpg',
  // Diamonds (5 of 8)
  'suit-1-diamonds': '/cards/diamonds_pip1.jpg',
  'suit-4-diamonds': '/cards/diamonds_pip4.jpg',
  'suit-5-diamonds': '/cards/diamonds_jack.jpg',
  'suit-6-diamonds': '/cards/diamonds_knight.jpg',
  'suit-8-diamonds': '/cards/diamonds_king.jpg',
  // Clubs (5 of 8)
  'suit-1-clubs': '/cards/clubs_pip1.jpg',
  'suit-2-clubs': '/cards/clubs_pip2.jpg',
  'suit-3-clubs': '/cards/clubs_pip3.jpg',
  'suit-4-clubs': '/cards/clubs_pip4.jpg',
  'suit-6-clubs': '/cards/clubs_knight.jpg',
  // Spades (6 of 8)
  'suit-1-spades': '/cards/spades_pip1.jpg',
  'suit-2-spades': '/cards/spades_pip2.jpg',
  'suit-3-spades': '/cards/spades_pip3.jpg',
  'suit-6-spades': '/cards/spades_knight.jpg',
  'suit-7-spades': '/cards/spades_queen.jpg',
  'suit-8-spades': '/cards/spades_king.jpg',
};

export function getCardImageUrl(card: CardData): string | null {
  const key = `${card.card_type}-${card.value}-${card.suit ?? 'null'}`;
  return IMAGE_MAP[key] ?? null;
}
