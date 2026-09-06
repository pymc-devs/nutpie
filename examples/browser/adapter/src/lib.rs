use nuts_rs::{
    Chain, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HasDims, LogpError, Settings,
};
use rand::{rngs::StdRng, SeedableRng};
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
};
thread_local! {
    static RESULT: RefCell<Vec<u8>> = RefCell::new(Vec::new());
    static CALLBACK: Cell<usize> = const { Cell::new(0) };
    static EVALUATIONS: Cell<u64> = const { Cell::new(0) };
}
#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "env")]
extern "C" {
    fn model_logp(x: *const f64, g: *mut f64, n: usize) -> f64;
    fn report_progress(chain: u32, index: u32, tuning: u32);
}
#[derive(Debug, thiserror::Error)]
#[error("Non-finite model log density or gradient")]
struct DensityError;
impl LogpError for DensityError {
    fn is_recoverable(&self) -> bool {
        true
    }
}
struct Density(usize);
impl HasDims for Density {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::new()
    }
}
impl CpuLogpFunc for Density {
    type LogpError = DensityError;
    type FlowParameters = ();
    type ExpandedVector = ();
    fn dim(&self) -> usize {
        self.0
    }
    fn logp(&mut self, x: &[f64], g: &mut [f64]) -> Result<f64, DensityError> {
        EVALUATIONS.with(|v| v.set(v.get() + 1));
        #[cfg(target_arch = "wasm32")]
        let lp = unsafe { model_logp(x.as_ptr(), g.as_mut_ptr(), self.0) };
        #[cfg(not(target_arch = "wasm32"))]
        let lp = CALLBACK.with(|p| unsafe {
            let f: extern "C" fn(*const f64, *mut f64) -> f64 = std::mem::transmute(p.get());
            f(x.as_ptr(), g.as_mut_ptr())
        });
        if lp.is_finite() && g.iter().all(|x| x.is_finite()) {
            Ok(lp)
        } else {
            Err(DensityError)
        }
    }
    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self,
        _: &mut R,
        _: &[f64],
    ) -> Result<(), CpuMathError> {
        Ok(())
    }
}
#[no_mangle]
pub extern "C" fn set_callback(p: usize) {
    CALLBACK.with(|v| v.set(p));
}
#[no_mangle]
pub extern "C" fn alloc_f64(n: usize) -> *mut f64 {
    Box::into_raw(vec![0.; n].into_boxed_slice()) as *mut f64
}
#[no_mangle]
pub unsafe extern "C" fn free_f64(p: *mut f64, n: usize) {
    drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(p, n)));
}
#[no_mangle]
pub extern "C" fn result_ptr() -> *const u8 {
    RESULT.with(|r| r.borrow().as_ptr())
}
#[no_mangle]
pub extern "C" fn result_len() -> usize {
    RESULT.with(|r| r.borrow().len())
}
#[no_mangle]
pub unsafe extern "C" fn run(
    n: usize,
    chains: u32,
    tune: u32,
    draws: u32,
    seed: u32,
    start: *const f64,
) -> i32 {
    // The caller owns a live, aligned buffer containing n doubles for this call.
    let initial = std::slice::from_raw_parts(start, n);
    let work = || -> Result<serde_json::Value, String> {
        if n == 0 || chains == 0 || tune == 0 || draws == 0 || tune.checked_add(draws).is_none() {
            return Err(
                "Dimensions, chains, warmup and draws must be positive and fit in u32".into(),
            );
        }
        if !initial.iter().all(|x| x.is_finite()) {
            return Err("Initial position must be finite".into());
        }
        #[cfg(not(target_arch = "wasm32"))]
        if CALLBACK.with(|p| p.get() == 0) {
            return Err("Set a model callback before sampling".into());
        }
        let mut samples = Vec::new();
        let mut divergences = 0;
        let mut leapfrogs = 0u64;
        let mut all_stats = Vec::new();
        EVALUATIONS.with(|v| v.set(0));
        for c in 0..chains {
            let mut settings = DiagNutsSettings::default();
            settings.num_tune = tune.into();
            settings.num_draws = draws.into();
            settings.maxdepth = 10;
            settings.adapt_options.step_size_settings.target_accept = 0.9;
            let mut rng = StdRng::seed_from_u64(seed as u64 + c as u64);
            let mut sampler = settings.new_chain(c.into(), CpuMath::new(Density(n)), &mut rng);
            sampler.set_position(initial).map_err(|e| e.to_string())?;
            let mut chain = Vec::new();
            let mut stats = Vec::new();
            for _i in 0..tune + draws {
                let (x, p) = sampler.draw().map_err(|e| e.to_string())?;
                leapfrogs += p.num_steps;
                if !p.tuning {
                    divergences += u32::from(p.diverging);
                    chain.push(x.to_vec());
                    stats.push(serde_json::json!({"diverging":p.diverging,"n_steps":p.num_steps,"step_size":p.step_size}));
                }
                #[cfg(target_arch = "wasm32")]
                if _i % 100 == 0 || _i + 1 == tune + draws {
                    report_progress(c, _i, u32::from(p.tuning));
                }
            }
            samples.push(chain);
            all_stats.push(stats);
        }
        Ok(
            serde_json::json!({"samples":samples,"stats":all_stats,"divergences":divergences,"leapfrog_steps":leapfrogs,"logp_evaluations":EVALUATIONS.with(|v|v.get())}),
        )
    };
    let (value, status) = match work() {
        Ok(v) => (v, 0),
        Err(e) => (serde_json::json!({"error":e}), 1),
    };
    RESULT.with(|r| *r.borrow_mut() = serde_json::to_vec(&value).unwrap());
    status
}
