/** Experimental adapter between independent Rust and Emscripten WASM memories.
 * Run in a worker: sampling is synchronous and blocks that worker until done.
 * Keep the Python BrowserModel alive. Do not run concurrent fits in this runtime.
 */
export async function sample({bytes, runtime, model, chains = 2, tune = 750,
    draws = 500, seed = 42, onProgress = () => {}}) {
  for (const [name, value] of Object.entries({chains, tune, draws, seed})) {
    if (!Number.isInteger(value) || value < (name === 'seed' ? 0 : 1)
        || value > 0xffffffff) throw Error(`Invalid ${name}`);
  }
  if (tune + draws > 0xffffffff) throw Error('Too many iterations');
  if (!model.initial.length || !model.initial.every(Number.isFinite)) {
    throw Error('A finite, nonempty initial position is required');
  }
  const unexpected = () => {throw Error('Unexpected wasm-bindgen runtime call');};
  let ex;
  const {instance} = await WebAssembly.instantiate(bytes, {
    env: {
      model_logp(x, g, n) {
        new Float64Array(runtime.wasmMemory.buffer, model.x_pointer, n)
          .set(new Float64Array(ex.memory.buffer, x, n));
        const lp = runtime.wasmTable.get(model.callback_pointer)(model.x_pointer, model.g_pointer);
        // Reacquire views: model evaluation may grow Emscripten memory.
        new Float64Array(ex.memory.buffer, g, n)
          .set(new Float64Array(runtime.wasmMemory.buffer, model.g_pointer, n));
        return lp;
      },
      report_progress(chain, index, tuning) {onProgress({chain, index, tuning: !!tuning});},
    },
    // These are retained by the pinned dependency build, but are not used by
    // the explicitly seeded sequential sampler. Fail if that assumption changes.
    __wbindgen_placeholder__: {__wbindgen_describe: unexpected},
    __wbindgen_externref_xform__: {
      __wbindgen_externref_table_set_null: unexpected,
      __wbindgen_externref_table_grow: unexpected,
    },
  });
  ex = instance.exports;
  const n = model.initial.length, p = ex.alloc_f64(n);
  try {
    new Float64Array(ex.memory.buffer, p, n).set(model.initial);
    const started = performance.now();
    const status = ex.run(n, chains, tune, draws, seed, p);
    const samplingSeconds = (performance.now() - started) / 1000;
    const result = JSON.parse(new TextDecoder().decode(
      new Uint8Array(ex.memory.buffer, ex.result_ptr(), ex.result_len())));
    if (status) throw Error(result.error);
    return {...result, sampling_seconds: samplingSeconds, layout: model.layout};
  } finally {
    ex.free_f64(p, n);
  }
}
