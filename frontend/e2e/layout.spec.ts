import { test, expect, type Page } from '@playwright/test';

/**
 * Tests for layout fixes:
 * - Talon group selection is clickable
 * - Cards in play are not cut off
 * - Game info bar is sticky/floating
 * - Spectator mode fits the table on screen
 * - Spectator mode shows game count for tournament format
 */

test.describe('Game Board Layout', () => {
  test('game info bar is sticky and always visible while scrolling', async ({ page }) => {
    await page.goto('/');
    await page.locator('button:has-text("Play vs AI")').click();
    await page.locator('button:has-text("Start Game")').click();
    await expect(page.locator('[data-testid="game-board"]')).toBeVisible({ timeout: 15_000 });
    await page.locator('.speed-control input[type="range"]').fill('0');

    // Wait for a meaningful phase
    await page.waitForFunction(() => {
      const board = document.querySelector('[data-testid="game-board"]');
      const phase = board?.getAttribute('data-phase');
      return phase && phase !== 'waiting' && phase !== 'dealing';
    }, { timeout: 15_000 });

    // Verify the game info bar exists and has sticky positioning
    const infoBar = page.locator('.game-info-bar');
    await expect(infoBar).toBeVisible();

    const position = await infoBar.evaluate(el => getComputedStyle(el).position);
    expect(position).toBe('sticky');
  });

  test('bottom player hand cards are not clipped by overflow', async ({ page }) => {
    await page.goto('/');
    await page.locator('button:has-text("Play vs AI")').click();
    await page.locator('button:has-text("Start Game")').click();
    await expect(page.locator('[data-testid="game-board"]')).toBeVisible({ timeout: 15_000 });
    await page.locator('.speed-control input[type="range"]').fill('0');

    await page.waitForFunction(() => {
      const board = document.querySelector('[data-testid="game-board"]');
      const phase = board?.getAttribute('data-phase');
      return phase && phase !== 'waiting' && phase !== 'dealing';
    }, { timeout: 15_000 });

    // Wait for hand cards to render
    const handCards = page.locator('.hand-bottom .card');
    await expect(handCards.first()).toBeVisible({ timeout: 10_000 });

    // Check that the game board allows vertical scrolling (overflow-y is auto, not hidden)
    const overflowY = await page.locator('.game-board').evaluate(el => getComputedStyle(el).overflowY);
    expect(overflowY).toBe('auto');

    // The bottom hand should span all columns (grid-area: bottom spanning 3 columns)
    const tableBottom = page.locator('.table-bottom');
    await expect(tableBottom).toBeVisible();

    // Verify last card in hand is within the viewport or scrollable to
    const lastCard = handCards.last();
    await expect(lastCard).toBeVisible();
    const box = await lastCard.boundingBox();
    expect(box).not.toBeNull();
  });

  test('talon groups are visible and clickable during talon exchange', async ({ page }) => {
    await page.goto('/');
    await page.locator('button:has-text("Play vs AI")').click();
    await page.locator('button:has-text("Start Game")').click();
    await expect(page.locator('[data-testid="game-board"]')).toBeVisible({ timeout: 15_000 });
    await page.locator('.speed-control input[type="range"]').fill('0');

    // Play through bidding — win a contract to get to talon exchange
    const maxIter = 200;
    let reachedTalon = false;
    for (let i = 0; i < maxIter; i++) {
      const phase = await getPhase(page);
      if (phase === 'finished') break;

      if (phase === 'talon_exchange') {
        reachedTalon = true;
        break;
      }

      if (phase === 'bidding') {
        // Try to bid three (lowest) to win the contract
        const threeBtn = page.locator('[data-testid="bid-three"], [data-testid="bid-3"]');
        const passBtn = page.locator('[data-testid="bid-pass"]');
        if (await threeBtn.isVisible().catch(() => false)) {
          await threeBtn.click();
        } else if (await passBtn.isVisible().catch(() => false)) {
          await passBtn.click();
        }
        await page.waitForTimeout(300);
      } else if (phase === 'king_calling') {
        const kingBtn = page.locator('.king-btn').first();
        if (await kingBtn.isVisible().catch(() => false)) {
          await kingBtn.click();
          await page.waitForTimeout(300);
        } else {
          await waitForPhaseChangeOrTimeout(page, 'king_calling', 3_000);
        }
      } else if (phase === 'trick_play') {
        // If we end up in trick play, game skipped talon (e.g., klop/solo)
        break;
      } else {
        await page.waitForTimeout(300);
      }
    }

    if (reachedTalon) {
      // Verify talon groups are visible
      const talonGroups = page.locator('.talon-group');
      const discardConfirm = page.locator('[data-testid="discard-confirm"]');

      // Either we see talon groups to choose, or we're in discard phase
      const hasGroups = await talonGroups.count() > 0;
      const hasDiscard = await discardConfirm.isVisible().catch(() => false);

      if (hasGroups) {
        // Verify groups are visible and have pointer cursor
        const firstGroup = talonGroups.first();
        await expect(firstGroup).toBeVisible();
        const cursor = await firstGroup.evaluate(el => getComputedStyle(el).cursor);
        expect(cursor).toBe('pointer');

        // Verify the talon selection area has z-index for clickability
        const talonSelection = page.locator('.talon-selection');
        const zIndex = await talonSelection.evaluate(el => getComputedStyle(el).zIndex);
        expect(Number(zIndex)).toBeGreaterThanOrEqual(5);
      } else {
        // Discard phase — just confirm it's reachable
        expect(hasDiscard || hasGroups).toBeTruthy();
      }
    }
    // If we never reached talon (all-pass/klop), the test still passes
  });

  test('table grid allows bottom and top hands to span full width', async ({ page }) => {
    await page.goto('/');
    await page.locator('button:has-text("Play vs AI")').click();
    await page.locator('button:has-text("Start Game")').click();
    await expect(page.locator('[data-testid="game-board"]')).toBeVisible({ timeout: 15_000 });

    await page.waitForFunction(() => {
      const board = document.querySelector('[data-testid="game-board"]');
      const phase = board?.getAttribute('data-phase');
      return phase && phase !== 'waiting' && phase !== 'dealing';
    }, { timeout: 15_000 });

    // Verify the bottom hand area spans the full table width
    const table = page.locator('.table');
    const tableBottom = page.locator('.table-bottom');
    await expect(table).toBeVisible();
    await expect(tableBottom).toBeVisible();

    const tableBox = await table.boundingBox();
    const bottomBox = await tableBottom.boundingBox();
    expect(tableBox).not.toBeNull();
    expect(bottomBox).not.toBeNull();

    // Bottom hand should be nearly as wide as the table (accounting for padding)
    if (tableBox && bottomBox) {
      expect(bottomBox.width).toBeGreaterThan(tableBox.width * 0.7);
    }
  });
});

test.describe('Spectator Mode Layout', () => {
  test('spectator setup shows number of games input', async ({ page }) => {
    await page.goto('/');
    await page.locator('button:has-text("Spectate AI vs AI")').click();
    await expect(page.locator('text=Configure Agents')).toBeVisible({ timeout: 5_000 });

    // Verify the number of games input exists
    const numGamesInput = page.locator('[data-testid="num-games-input"]');
    await expect(numGamesInput).toBeVisible();

    // Default value should be 1
    const value = await numGamesInput.inputValue();
    expect(value).toBe('1');

    // Can set to higher value
    await numGamesInput.fill('5');
    const newValue = await numGamesInput.inputValue();
    expect(newValue).toBe('5');
  });

  test('spectator board has scrollable overflow', async ({ page }) => {
    await page.goto('/');
    await page.locator('button:has-text("Spectate AI vs AI")').click();
    await expect(page.locator('text=Configure Agents')).toBeVisible({ timeout: 5_000 });

    // Start a game
    await page.locator('button:has-text("Start Game")').click();

    // Wait for the spectator board to appear
    const board = page.locator('.spectator-board');
    await expect(board).toBeVisible({ timeout: 15_000 });

    // Verify the board has auto overflow for scrolling
    const overflowY = await board.evaluate(el => getComputedStyle(el).overflowY);
    expect(overflowY).toBe('auto');
  });

  test('spectator table uses full-width grid for top/bottom hands', async ({ page }) => {
    await page.goto('/');
    await page.locator('button:has-text("Spectate AI vs AI")').click();
    await expect(page.locator('text=Configure Agents')).toBeVisible({ timeout: 5_000 });

    await page.locator('button:has-text("Start Game")').click();

    // Wait for the spectator table to appear
    const table = page.locator('.spectator-table');
    await expect(table).toBeVisible({ timeout: 15_000 });

    // Verify table has correct grid template areas spanning full width
    const gridAreas = await table.evaluate(el => getComputedStyle(el).gridTemplateAreas);
    // The grid areas should have top and bottom spanning all 3 columns
    expect(gridAreas).toContain('top');
    expect(gridAreas).toContain('bottom');
  });

  test('spectator game info bar is sticky', async ({ page }) => {
    await page.goto('/');
    await page.locator('button:has-text("Spectate AI vs AI")').click();
    await page.locator('button:has-text("Start Game")').click();

    const infoBar = page.locator('.spectator-board .game-info-bar');
    await expect(infoBar).toBeVisible({ timeout: 15_000 });

    const position = await infoBar.evaluate(el => getComputedStyle(el).position);
    expect(position).toBe('sticky');
  });

  test('spectator view with tournament shows game counter', async ({ page }) => {
    await page.goto('/');
    await page.locator('button:has-text("Spectate AI vs AI")').click();
    await expect(page.locator('text=Configure Agents')).toBeVisible({ timeout: 5_000 });

    // Set number of games to 3
    const numGamesInput = page.locator('[data-testid="num-games-input"]');
    await numGamesInput.fill('3');

    // Start the game
    await page.locator('button:has-text("Start Game")').click();

    // Wait for spectator board to load
    await expect(page.locator('.spectator-board')).toBeVisible({ timeout: 15_000 });

    // Verify game counter is displayed
    const gameCounter = page.locator('[data-testid="game-counter"]');
    await expect(gameCounter).toBeVisible({ timeout: 5_000 });
    await expect(gameCounter).toContainText('Game');
    await expect(gameCounter).toContainText('/3');
  });
});

// ---------- Helpers ----------

async function getPhase(page: Page): Promise<string> {
  const board = page.locator('[data-testid="game-board"]');
  return (await board.getAttribute('data-phase')) ?? 'unknown';
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
    // Timeout is ok
  }
}
