import { test, expect } from '@playwright/test';

/**
 * E2E tests for the tournament feature:
 * - Setup screen: add models, select checkpoints
 * - Single tournament bracket: auto-run all games
 * - Multi-tournament simulation: start, track progress, view standings
 */

test.describe('Tournament', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.locator('button:has-text("Tournament")').click();
    await expect(page.locator('text=Double Elimination Tournament')).toBeVisible({ timeout: 5_000 });
  });

  test('shows tournament setup screen', async ({ page }) => {
    await expect(page.locator('text=Models (0/8)')).toBeVisible();
    await expect(page.locator('text=Games per round')).toBeVisible();
  });

  test('can add random agents', async ({ page }) => {
    // Select random type
    const typeSelect = page.locator('.add-entry-form select').first();
    await typeSelect.selectOption('random');

    // Add 4 random agents
    for (let i = 0; i < 4; i++) {
      await page.locator('button:has-text("+ Add")').click();
    }

    await expect(page.locator('text=Models (4/8)')).toBeVisible();
    await expect(page.locator('button:has-text("Start Tournament")')).toBeEnabled();
  });

  test('start button disabled with fewer than 4 models', async ({ page }) => {
    const typeSelect = page.locator('.add-entry-form select').first();
    await typeSelect.selectOption('random');

    // Add only 2
    await page.locator('button:has-text("+ Add")').click();
    await page.locator('button:has-text("+ Add")').click();

    await expect(page.locator('button:has-text("Start Tournament")')).toBeDisabled();
  });

  test('can remove an agent', async ({ page }) => {
    const typeSelect = page.locator('.add-entry-form select').first();
    await typeSelect.selectOption('random');
    await page.locator('button:has-text("+ Add")').click();
    await expect(page.locator('text=Models (1/8)')).toBeVisible();

    await page.locator('.btn-danger').click();
    await expect(page.locator('text=Models (0/8)')).toBeVisible();
  });

  test('single tournament bracket runs all games', async ({ page }) => {
    // Add 4 random agents
    const typeSelect = page.locator('.add-entry-form select').first();
    await typeSelect.selectOption('random');
    for (let i = 0; i < 4; i++) {
      await page.locator('button:has-text("+ Add")').click();
    }

    // Start single tournament
    await page.locator('button:has-text("Start Tournament")').click();
    await expect(page.locator('text=Winners Bracket')).toBeVisible({ timeout: 5_000 });

    // Auto-run all matches
    await page.locator('button:has-text("Auto-Run All")').click();

    // Wait for champion banner
    await expect(page.locator('.tournament-champion-banner')).toBeVisible({ timeout: 120_000 });
    await expect(page.locator('.champion-text')).toContainText('Champion:');
  });

  test('mode toggle shows multi-tournament options', async ({ page }) => {
    // Switch to multi mode
    await page.locator('label:has-text("Multi-Tournament")').click();
    await expect(page.locator('text=Number of tournaments')).toBeVisible();
  });

  test('multi-tournament simulation starts and shows progress', async ({ page }) => {
    // Add 4 random agents
    const typeSelect = page.locator('.add-entry-form select').first();
    await typeSelect.selectOption('random');
    for (let i = 0; i < 4; i++) {
      await page.locator('button:has-text("+ Add")').click();
    }

    // Switch to multi mode
    await page.locator('label:has-text("Multi-Tournament")').click();

    // Set 1 tournament for speed
    const numInput = page.locator('input[type="number"]').last();
    await numInput.fill('1');

    // Set 1 game per round for speed
    const gamesInput = page.locator('input[type="number"]').first();
    await gamesInput.fill('1');

    // Start simulation
    await page.locator('button:has-text("Simulate")').click();

    // Wait for progress view
    await expect(page.locator('[data-testid="multi-tournament-progress"]')).toBeVisible({ timeout: 10_000 });

    // Wait for completion
    await expect(page.locator('[data-testid="multi-tournament-counter"]')).toContainText('Completed', { timeout: 120_000 });

    // Standings table should be visible
    await expect(page.locator('[data-testid="multi-tournament-standings"]')).toBeVisible();

    // Should have rows in the table
    const rows = page.locator('[data-testid="multi-tournament-standings"] tbody tr');
    await expect(rows).not.toHaveCount(0);
  });

  test('checkpoint dropdown shows for RL type', async ({ page }) => {
    // Default type is RL
    const selects = page.locator('.add-entry-form select');
    // Should have 2 selects: type + checkpoint
    await expect(selects).toHaveCount(2);
  });

  test('checkpoint dropdown hidden for random type', async ({ page }) => {
    const typeSelect = page.locator('.add-entry-form select').first();
    await typeSelect.selectOption('random');
    // Only type select visible
    const selects = page.locator('.add-entry-form select');
    await expect(selects).toHaveCount(1);
  });

  test('multi-tournament runs to completion with multiple tournaments', async ({ page }) => {
    // Add 4 random agents
    const typeSelect = page.locator('.add-entry-form select').first();
    await typeSelect.selectOption('random');
    for (let i = 0; i < 4; i++) {
      await page.locator('button:has-text("+ Add")').click();
    }

    // Switch to multi mode
    await page.locator('label:has-text("Multi-Tournament")').click();

    // Set 3 tournaments
    const numInput = page.locator('input[type="number"]').last();
    await numInput.fill('3');

    // Set 1 game per round for speed
    const gamesInput = page.locator('input[type="number"]').first();
    await gamesInput.fill('1');

    // Start simulation
    await page.locator('button:has-text("Simulate")').click();

    // Wait for completion — counter should show "3 / 3"
    await expect(page.locator('[data-testid="multi-tournament-counter"]')).toContainText('3 / 3', { timeout: 180_000 });

    // Standings should be visible and reset button available
    await expect(page.locator('[data-testid="multi-tournament-standings"]')).toBeVisible();
    await expect(page.locator('button:has-text("Reset")')).toBeVisible();
  });
});
