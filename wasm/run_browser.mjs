import puppeteer from 'puppeteer-core';
import fs from 'node:fs/promises';
if (!process.env.CHROME_PATH) throw Error('Set CHROME_PATH to a Chrome executable');
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_PATH, headless: true, protocolTimeout: 600000,
});
try {
  const page = await browser.newPage();
  page.on('console', message => console.log(message.text()));
  page.on('pageerror', error => console.error(String(error)));
  await page.goto(process.argv[2] ?? 'http://127.0.0.1:8768');
  await page.waitForFunction(() => window.probeDone, {timeout: 600000});
  const result = await page.evaluate(() => ({
    passed: window.probePassed, error: window.probeError, output: window.output,
  }));
  await fs.writeFile(process.argv[3] ?? 'probe-output.json', JSON.stringify(result, null, 2));
  if (!result.passed || result.error) process.exitCode = 1;
} finally {
  await browser.close();
}
