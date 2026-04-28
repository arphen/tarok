import { test, expect } from '@playwright/test';

/**
 * Smoke tests for the 3-player Tarok UI surfaces.
 *
 * Validates that the variant radio buttons in both the Play-vs-AI lobby
 * and the Bot Arena correctly resize the seat-selection grids. Does NOT
 * start a full game (which would require a 3p model checkpoint to be
 * shipped with the repo).
 */

test.describe('3-player variant UI', () => {
  test('Play-vs-AI lobby switches between 3 and 2 opponent slots', async ({ page }) => {
    await page.goto('/');
    await page.locator('button:has-text("Play vs AI")').click();
    await expect(page.locator('button:has-text("Start Game")')).toBeVisible({ timeout: 5_000 });

    // Default = 4-player → 3 AI dropdowns labelled AI-1 / AI-2 / AI-3
    await expect(page.locator('.lobby-label:has-text("AI-1")')).toBeVisible();
    await expect(page.locator('.lobby-label:has-text("AI-2")')).toBeVisible();
    await expect(page.locator('.lobby-label:has-text("AI-3")')).toBeVisible();

    // Switch to 3-player
    await page.locator('input[type="radio"][value="three_player"]').check();

    // Now only 2 AI dropdowns
    await expect(page.locator('.lobby-label:has-text("AI-1")')).toBeVisible();
    await expect(page.locator('.lobby-label:has-text("AI-2")')).toBeVisible();
    await expect(page.locator('.lobby-label:has-text("AI-3")')).toHaveCount(0);

    // Variant choice persists across navigation (localStorage)
    await page.locator('button:has-text("← Back")').click();
    await page.locator('button:has-text("Play vs AI")').click();
    await expect(page.locator('input[type="radio"][value="three_player"]')).toBeChecked();
    await expect(page.locator('.lobby-label:has-text("AI-3")')).toHaveCount(0);
  });

  test('Bot Arena switches between 4 and 3 seat cards', async ({ page }) => {
    await page.goto('/');
    await page.locator('button:has-text("Bot Arena")').click();
    await expect(page.locator('text=Agents')).toBeVisible({ timeout: 5_000 });

    // Default = 4 seats
    await expect(page.locator('.arena-agent-seat')).toHaveCount(4);

    // Switch to 3-player
    await page.locator('input[name="arena-variant"][value="three_player"]').check();
    await expect(page.locator('.arena-agent-seat')).toHaveCount(3);

    // Switch back to 4-player
    await page.locator('input[name="arena-variant"][value="four_player"]').check();
    await expect(page.locator('.arena-agent-seat')).toHaveCount(4);
  });
});
