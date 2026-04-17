import { test, expect, type Page } from '@playwright/test';

/**
 * Plays through a complete game making random legal moves at every decision point.
 * This catches issues like king calling, bid selection, talon exchange, and
 * discard that only surface when the human player wins the contract.
 *
 * Runs 3 times with different random seeds to increase coverage.
 */
for (let seed = 1; seed <= 3; seed++) {
  test(`random-move full game (seed ${seed})`, async ({ page }) => {
    test.setTimeout(120_000);

    // Deterministic PRNG so failures are reproducible
    const rng = mulberry32(seed);

    await startGame(page);

    let lastPhase = '';
    let stuckCount = 0;
    const maxIter = 400;
    for (let i = 0; i < maxIter; i++) {
      const phase = await getPhase(page);
      if (phase === 'finished' || phase === 'scoring') break;

      // Detect being stuck in same phase too long
      if (phase === lastPhase) {
        stuckCount++;
        if (stuckCount > 60) {
          throw new Error(`Stuck in phase "${phase}" for ${stuckCount} iterations`);
        }
      } else {
        stuckCount = 0;
        lastPhase = phase;
      }

      if (phase === 'bidding') {
        await handleBidding(page, rng);
      } else if (phase === 'king_calling') {
        await handleKingCalling(page, rng);
      } else if (phase === 'talon_exchange') {
        await handleTalonExchange(page, rng);
      } else if (phase === 'trick_play') {
        await handleTrickPlay(page, rng);
      } else {
        // Unknown/transitional phase — wait for it to change
        await page.waitForTimeout(200);
      }
    }

    // Game must reach "finished"
    await expect(page.locator('[data-testid="score-display"]')).toBeVisible({ timeout: 30_000 });
    await expect(page.locator('[data-testid="score-display"]')).toContainText('Game Over');
  });
}

// ───────────────────────── helpers ─────────────────────────

/** Deterministic 32-bit PRNG (Mulberry32). Returns () => [0, 1). */
function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pickRandom<T>(arr: T[], rng: () => number): T {
  return arr[Math.floor(rng() * arr.length)];
}

async function startGame(page: Page) {
  await page.goto('/');
  await expect(page.locator('text=Slovenian Tarok')).toBeVisible();
  await page.locator('button:has-text("Play vs AI")').click();

  const startBtn = page.locator('button:has-text("Start Game")');
  await expect(startBtn).toBeVisible({ timeout: 5_000 });
  await startBtn.click();

  await expect(page.locator('[data-testid="game-board"]')).toBeVisible({ timeout: 15_000 });
  await expect(page.locator('.connection-status:has-text("Connected")')).toBeVisible({ timeout: 10_000 });

  // Zero AI delay for speed
  await page.locator('.speed-control input[type="range"]').fill('0');

  // Wait past dealing
  await page.waitForFunction(
    () => {
      const board = document.querySelector('[data-testid="game-board"]');
      const p = board?.getAttribute('data-phase');
      return p && p !== 'waiting' && p !== 'dealing';
    },
    { timeout: 15_000 },
  );
}

async function getPhase(page: Page): Promise<string> {
  return (await page.locator('[data-testid="game-board"]').getAttribute('data-phase')) ?? 'unknown';
}

async function handleBidding(page: Page, rng: () => number) {
  // Collect all visible bid buttons (pass + contract bids)
  const allBtns = page.locator('.bid-actions button');
  const count = await allBtns.count().catch(() => 0);
  if (count > 0) {
    const idx = Math.floor(rng() * count);
    await allBtns.nth(idx).click();
    await page.waitForTimeout(300);
  } else {
    await waitForPhaseChange(page, 'bidding', 3_000);
  }
}

async function handleKingCalling(page: Page, rng: () => number) {
  // Wait for king buttons to appear (they may not be for us)
  try {
    await page.locator('.king-btn').first().waitFor({ state: 'visible', timeout: 5_000 });
  } catch {
    await waitForPhaseChange(page, 'king_calling', 5_000);
    return;
  }

  const btns = page.locator('.king-btn');
  const count = await btns.count();
  if (count > 0) {
    const idx = Math.floor(rng() * count);
    await btns.nth(idx).click();
    await page.waitForTimeout(500);
  } else {
    await waitForPhaseChange(page, 'king_calling', 5_000);
  }
}

async function handleTalonExchange(page: Page, rng: () => number) {
  // Phase 1: pick a talon group
  const groups = page.locator('.talon-group');
  const gCount = await groups.count().catch(() => 0);
  if (gCount > 0) {
    const idx = Math.floor(rng() * gCount);
    await groups.nth(idx).click();
    await page.waitForTimeout(500);
    // After picking, we may enter discard sub-phase — fall through
  }

  // Phase 2: discard cards
  const discardConfirm = page.locator('[data-testid="discard-confirm"]');
  if (await discardConfirm.isVisible().catch(() => false)) {
    // Click random discardable cards until confirm is enabled
    for (let attempt = 0; attempt < 20; attempt++) {
      if (await discardConfirm.isEnabled()) break;
      const cards = page.locator('.discard-card:not(.discard-selected)');
      const dCount = await cards.count().catch(() => 0);
      if (dCount === 0) break;
      const idx = Math.floor(rng() * dCount);
      await cards.nth(idx).click();
      await page.waitForTimeout(100);
    }
    if (await discardConfirm.isEnabled()) {
      await discardConfirm.click();
      await page.waitForTimeout(500);
      return;
    }
  }

  // If we're still in talon_exchange (AI is exchanging), wait
  if (await getPhase(page) === 'talon_exchange') {
    await waitForPhaseChange(page, 'talon_exchange', 5_000);
  }
}

async function handleTrickPlay(page: Page, rng: () => number) {
  // Wait for clickable cards or phase change
  try {
    await page.waitForFunction(
      () => {
        const board = document.querySelector('[data-testid="game-board"]');
        const phase = board?.getAttribute('data-phase');
        if (phase !== 'trick_play') return true;
        return document.querySelectorAll('.card-clickable').length > 0;
      },
      { timeout: 5_000 },
    );
  } catch {
    return; // Not our turn yet, let loop retry
  }

  if (await getPhase(page) !== 'trick_play') return;

  const clickable = page.locator('.card-clickable');
  const count = await clickable.count().catch(() => 0);
  if (count > 0) {
    const idx = Math.floor(rng() * count);
    await clickable.nth(idx).click();
    // Wait for the card to be processed before next iteration
    await page.waitForTimeout(300);
  }
}

async function waitForPhaseChange(page: Page, currentPhase: string, timeout: number) {
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
    // Timeout ok — main loop retries
  }
}
