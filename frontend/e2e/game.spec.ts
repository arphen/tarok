import { test, expect, type Page } from '@playwright/test';

/**
 * Plays a full game of Tarok against AI opponents.
 *
 * Strategy: always pass during bidding (AI will win the contract or Klop),
 * then play the first legal card each trick.
 */
test('play a full game against AI and verify it completes', async ({ page }) => {
  // 1. Navigate and start a game
  await page.goto('/');
  await expect(page.locator('text=Slovenian Tarok')).toBeVisible();

  const playBtn = page.locator('button:has-text("Play vs AI")');
  await playBtn.click();

  // Wait for connection and game board
  await expect(page.locator('[data-testid="game-board"]')).toBeVisible({ timeout: 15_000 });
  await expect(page.locator('.connection-status:has-text("Connected")')).toBeVisible({ timeout: 10_000 });

  // Wait for the game to enter a meaningful phase (not waiting/dealing)
  await page.waitForFunction(
    () => {
      const board = document.querySelector('[data-testid="game-board"]');
      const phase = board?.getAttribute('data-phase');
      return phase && phase !== 'waiting' && phase !== 'dealing';
    },
    { timeout: 15_000 },
  );

  // 2. Play through the game — handle each phase until finished
  const maxIterations = 200;
  for (let i = 0; i < maxIterations; i++) {
    const phase = await getPhase(page);
    if (phase === 'finished') break;

    if (phase === 'bidding') {
      // Always pass
      const passBtn = page.locator('[data-testid="bid-pass"]');
      try {
        await passBtn.waitFor({ state: 'visible', timeout: 3_000 });
        await passBtn.click();
      } catch {
        // Not our turn to bid; wait for AI
        await page.waitForTimeout(300);
      }
    } else if (phase === 'king_calling') {
      const kingBtn = page.locator('.king-btn').first();
      try {
        await kingBtn.waitFor({ state: 'visible', timeout: 3_000 });
        await kingBtn.click();
      } catch {
        await page.waitForTimeout(300);
      }
    } else if (phase === 'talon_exchange') {
      const talonGroup = page.locator('.talon-group').first();
      try {
        await talonGroup.waitFor({ state: 'visible', timeout: 3_000 });
        await talonGroup.click();
      } catch {
        await page.waitForTimeout(300);
      }
    } else if (phase === 'trick_play') {
      // Wait for a clickable card (our turn) or phase change
      const playableCard = page.locator('.card-clickable').first();
      try {
        await playableCard.waitFor({ state: 'visible', timeout: 5_000 });
        await playableCard.click();
        // Give backend time to process the trick
        await page.waitForTimeout(600);
      } catch {
        // Not our turn, wait for AI to play
        await page.waitForTimeout(300);
      }
    } else {
      // Unknown/transitional phase, wait briefly
      await page.waitForTimeout(500);
    }
  }

  // 3. Verify game completed
  await expect(page.locator('[data-testid="score-display"]')).toBeVisible({ timeout: 30_000 });
  await expect(page.locator('[data-testid="score-display"]')).toContainText('Game Over');

  // 4. Verify the game log captured meaningful events
  const log = page.locator('[data-testid="game-log"]');
  await expect(log).toBeVisible();
  const logText = await log.innerText();
  expect(logText).toContain('Game started');
  expect(logText).toContain('Cards dealt');
  expect(logText).toContain('bids');
  expect(logText).toContain('plays');
  expect(logText).toContain('wins the trick');
  expect(logText).toContain('Game over');

  // 5. Verify scores are shown for all 4 players
  const scoreDisplay = page.locator('[data-testid="score-display"]');
  const scoreEntries = scoreDisplay.locator('.score-entry');
  await expect(scoreEntries).toHaveCount(4);
});

async function getPhase(page: Page): Promise<string> {
  const board = page.locator('[data-testid="game-board"]');
  return (await board.getAttribute('data-phase')) ?? 'unknown';
}
