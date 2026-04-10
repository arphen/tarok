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

  // Click through the game setup lobby
  const startBtn = page.locator('button:has-text("Start Game")');
  await expect(startBtn).toBeVisible({ timeout: 5_000 });
  await startBtn.click();

  // Wait for connection and game board
  await expect(page.locator('[data-testid="game-board"]')).toBeVisible({ timeout: 15_000 });
  await expect(page.locator('.connection-status:has-text("Connected")')).toBeVisible({ timeout: 10_000 });

  // Disable AI delay for faster tests
  await page.locator('.speed-control input[type="range"]').fill('0');

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
  const maxIterations = 300;
  for (let i = 0; i < maxIterations; i++) {
    const phase = await getPhase(page);
    if (phase === 'finished') break;

    if (phase === 'bidding') {
      await handleBidding(page);
    } else if (phase === 'king_calling') {
      await handleKingCalling(page);
    } else if (phase === 'talon_exchange') {
      await handleTalonExchange(page);
    } else if (phase === 'trick_play') {
      await handleTrickPlay(page);
    } else {
      // Transitional phase, wait briefly
      await page.waitForTimeout(300);
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

test('cards without images show white fallback with text', async ({ page }) => {
  await page.goto('/');
  await page.locator('button:has-text("Play vs AI")').click();
  await page.locator('button:has-text("Start Game")').click();
  await expect(page.locator('[data-testid="game-board"]')).toBeVisible({ timeout: 15_000 });

  // Reduce AI delay for faster tests
  await page.locator('.speed-control input[type="range"]').fill('0.25');

  // Wait until we have cards in hand
  await page.waitForFunction(
    () => {
      const board = document.querySelector('[data-testid="game-board"]');
      const phase = board?.getAttribute('data-phase');
      return phase && phase !== 'waiting' && phase !== 'dealing';
    },
    { timeout: 15_000 },
  );

  // Wait for hand to be rendered
  const handCards = page.locator('.hand-bottom .card:not(.card-back)');
  await expect(handCards.first()).toBeVisible({ timeout: 10_000 });

  const count = await handCards.count();
  expect(count).toBeGreaterThan(0);

  // Check that fallback cards (without images) have white background
  const fallbackCards = page.locator('.hand-bottom .card-fallback');
  const fallbackCount = await fallbackCards.count();

  if (fallbackCount > 0) {
    const bg = await fallbackCards.first().evaluate(
      el => getComputedStyle(el).backgroundColor,
    );
    expect(bg).toBe('rgb(255, 255, 255)');

    // Verify fallback cards have visible text content
    const text = await fallbackCards.first().innerText();
    expect(text.length).toBeGreaterThan(0);
  }

  // Verify cards WITH images render an img element
  const imageCards = page.locator('.hand-bottom .card:not(.card-fallback):not(.card-back) img.card-image');
  const imageCount = await imageCards.count();

  // Every card is either an image card or a fallback card
  expect(imageCount + fallbackCount).toBe(count);
});

// ---------- Helpers ----------

async function getPhase(page: Page): Promise<string> {
  const board = page.locator('[data-testid="game-board"]');
  return (await board.getAttribute('data-phase')) ?? 'unknown';
}

async function handleBidding(page: Page) {
  const passBtn = page.locator('[data-testid="bid-pass"]');
  if (await passBtn.isVisible().catch(() => false)) {
    await passBtn.click();
    await page.waitForTimeout(300);
  } else {
    await waitForPhaseChangeOrTimeout(page, 'bidding', 3_000);
  }
}

async function handleKingCalling(page: Page) {
  const kingBtn = page.locator('.king-btn').first();
  if (await kingBtn.isVisible().catch(() => false)) {
    await kingBtn.click();
    await page.waitForTimeout(300);
  } else {
    await waitForPhaseChangeOrTimeout(page, 'king_calling', 3_000);
  }
}

async function handleTalonExchange(page: Page) {
  // Try choosing a talon group first
  const talonGroup = page.locator('.talon-group').first();
  if (await talonGroup.isVisible().catch(() => false)) {
    await talonGroup.click();
    await page.waitForTimeout(500);
    return;
  }

  // Handle discard phase
  const discardConfirm = page.locator('[data-testid="discard-confirm"]');
  if (await discardConfirm.isVisible().catch(() => false)) {
    const discardCards = page.locator('.discard-card');
    const count = await discardCards.count();
    for (let j = 0; j < count; j++) {
      await discardCards.nth(j).click();
      await page.waitForTimeout(100);
      if (await discardConfirm.isEnabled()) break;
    }
    if (await discardConfirm.isEnabled()) {
      await discardConfirm.click();
      await page.waitForTimeout(500);
    }
    return;
  }

  await waitForPhaseChangeOrTimeout(page, 'talon_exchange', 5_000);
}

async function handleTrickPlay(page: Page) {
  // Wait for either a clickable card or phase change
  try {
    await page.waitForFunction(
      () => {
        const board = document.querySelector('[data-testid="game-board"]');
        if (board?.getAttribute('data-phase') !== 'trick_play') return true;
        return document.querySelectorAll('.card-clickable').length > 0;
      },
      { timeout: 10_000 },
    );
  } catch {
    return;
  }

  const phase = await getPhase(page);
  if (phase !== 'trick_play') return;

  const clickable = page.locator('.card-clickable').first();
  if (await clickable.isVisible().catch(() => false)) {
    await clickable.click();
    await page.waitForTimeout(500);
  }
}

async function waitForPhaseChangeOrTimeout(page: Page, currentPhase: string, timeout: number) {
  try {
    await page.waitForFunction(
      (phase) => {
        const board = document.querySelector('[data-testid="game-board"]');
        return board?.getAttribute('data-phase') !== phase;
      },
      currentPhase,
      { timeout },
    );
  } catch {
    // Timeout is ok — we'll re-enter the main loop
  }
}
