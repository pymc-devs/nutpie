// Exercise the actual Rust WASM module and JS memory bridge with a Gaussian.
// A real browser Numba function-table callback additionally needs a Xeus runtime.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {sample} from './bridge.mjs';
const bytes = await readFile(process.argv[2] ??
  new URL('./adapter/target/wasm32-unknown-unknown/release/nuts_browser_adapter.wasm', import.meta.url));
const memory = new WebAssembly.Memory({initial: 1});
let calls = 0, progress = 0;
const runtime = {wasmMemory: memory, wasmTable: {get(pointer) {
  assert.equal(pointer, 7);
  return (xp, gp) => {
    // Exercise view invalidation when Emscripten grows memory during a callback.
    if (++calls === 1) memory.grow(1);
    const x = new Float64Array(memory.buffer, xp, 2);
    const g = new Float64Array(memory.buffer, gp, 2);
    g[0] = -x[0]; g[1] = -x[1];
    return -(x[0] ** 2 + x[1] ** 2) / 2;
  };
}}};
const model = {initial: [0.1, 0.2], x_pointer: 0, g_pointer: 16,
  callback_pointer: 7, layout: []};
const options = {bytes, runtime, model, onProgress: () => progress++};
const result = await sample(options);
assert.equal(result.divergences, 0);
assert.equal(result.samples.length, 2);
assert.ok(result.samples.every(c => c.length === 500 && c.every(x => x.length === 2)));
assert.equal(calls, result.logp_evaluations);
assert.ok(progress > 0);
for (let j = 0; j < 2; j++) {
  const xs = result.samples.flat().map(x => x[j]);
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  const second = xs.reduce((a, b) => a + b * b, 0) / xs.length;
  assert.ok(Math.abs(mean) < 0.2, `Gaussian mean ${mean}`);
  assert.ok(Math.abs(second - 1) < 0.25, `Gaussian second moment ${second}`);
}
await assert.rejects(sample({...options, draws: 0}), /Invalid draws/);
await assert.rejects(sample({...options, model: {...model, initial: [NaN, 0]}}), /finite/);
const badRuntime = {...runtime, wasmTable: {get: () => () => NaN}};
await assert.rejects(sample({...options, runtime: badRuntime}));
console.log('WASM Gaussian posterior, memory growth, progress and error handling passed');
