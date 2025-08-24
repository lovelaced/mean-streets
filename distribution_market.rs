#![no_main]
#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use ethabi::{encode, decode, Token, ParamType};
use polkavm_derive::polkavm_export;
use simplealloc::SimpleAlloc;
use uapi::{HostFn, HostFnImpl as api, StorageFlags, ReturnFlags, CallFlags};

#[global_allocator]
static ALLOCATOR: SimpleAlloc<100000> = SimpleAlloc::new(); // 100KB to match v4

// Constants for fixed-point arithmetic (9 decimal places)
const DECIMALS: u64 = 1_000_000_000;
const MAX_INPUT: usize = 1000;
const MAX_STORAGE_VALUE: usize = 512;

// Wei conversion constants
const WEI_TO_FIXED: u64 = 1_000_000_000; // 1e9 wei = 1 unit
const MIN_TRANSFER_UNIT: u64 = 1_000_000; // Minimum transfer in wei

// Mathematical constants
const PI_FIXED: u64 = 3_141_592_654; // π with 9 decimals
const SQRT_PI_FIXED: u64 = 1_772_453_851; // √π with 9 decimals
const SQRT_2PI_FIXED: u64 = 2_506_628_275; // √(2π) with 9 decimals (slight precision loss)
const E_FIXED: u64 = 2_718_281_828; // e with 9 decimals

// Optimized parameters for gas efficiency
const TAYLOR_ITERATIONS: u32 = 15; // Reduced from 30
const MIN_VARIANCE: u64 = 1_000_000; // 0.001 minimum variance
const INTEGRATION_STEPS: u64 = 20; // Reduced from 40

// Storage keys
const OWNER_KEY: &[u8] = b"owner";
const INITIALIZED_KEY: &[u8] = b"initialized";
const MARKET_COUNT_KEY: &[u8] = b"market_count";
const POSITION_COUNT_KEY: &[u8] = b"position_count";
const LP_TOKEN_PREFIX: &[u8] = b"lp_";
const METADATA_PREFIX: &[u8] = b"meta_";
const POSITION_BY_ID_PREFIX: &[u8] = b"pos_id_";
const TRADER_POSITIONS_PREFIX: &[u8] = b"trader_pos_";
const TRADER_POS_COUNT_PREFIX: &[u8] = b"trader_cnt_";
const BLOCK_NUMBER_KEY: &[u8] = b"block_number";

// Function selectors (all 24 functions from original)
const INITIALIZE_SELECTOR: [u8; 4] = [0x81, 0x29, 0xfc, 0x1c];
const CREATE_MARKET_SELECTOR: [u8; 4] = [0x44, 0xb8, 0x5a, 0x62];
const TRADE_DISTRIBUTION_SELECTOR: [u8; 4] = [0x5e, 0xa5, 0xec, 0xce];
const ADD_LIQUIDITY_SELECTOR: [u8; 4] = [0x72, 0x26, 0x13, 0x33]; // addLiquidity(uint64,uint256)
const REMOVE_LIQUIDITY_SELECTOR: [u8; 4] = [0x88, 0xb2, 0x26, 0x37];
const GET_MARKET_STATE_SELECTOR: [u8; 4] = [0x20, 0x1d, 0x2f, 0x2b];
const GET_CONSENSUS_SELECTOR: [u8; 4] = [0xb9, 0xf2, 0xf5, 0xbb]; // getConsensus(uint256)
const GET_METADATA_SELECTOR: [u8; 4] = [0x99, 0x8e, 0x84, 0xa3];
const GET_MARKET_COUNT_SELECTOR: [u8; 4] = [0xfd, 0x69, 0xf3, 0xc2];
const CLOSE_POSITION_SELECTOR: [u8; 4] = [0x38, 0x4c, 0x07, 0xe6];
const GET_POSITION_SELECTOR: [u8; 4] = [0x0f, 0x85, 0xfc, 0x5a];
const GET_TRADER_POSITIONS_SELECTOR: [u8; 4] = [0x5f, 0xbb, 0xb2, 0xff];
const RESOLVE_MARKET_SELECTOR: [u8; 4] = [0x6d, 0x22, 0x83, 0xa4];
const CLAIM_WINNINGS_SELECTOR: [u8; 4] = [0x08, 0xf7, 0xed, 0x50];
const CALCULATE_TRADE_SELECTOR: [u8; 4] = [0x6c, 0xfa, 0x49, 0x1b]; // calculateTrade(uint256,uint64,uint64,uint64,uint64)
const GET_LP_BALANCE_SELECTOR: [u8; 4] = [0x0e, 0x3e, 0x56, 0xf8];
const GET_AMM_HOLDINGS_SELECTOR: [u8; 4] = [0x82, 0x51, 0xa2, 0x82]; // getAMMHoldings(uint256)
const EVALUATE_AT_SELECTOR: [u8; 4] = [0x3b, 0x51, 0x07, 0x6f];
const GET_CDF_SELECTOR: [u8; 4] = [0xd8, 0xff, 0xb3, 0x5a];
const GET_EXPECTED_VALUE_SELECTOR: [u8; 4] = [0x92, 0xac, 0xfd, 0xf9];
const GET_BOUNDS_SELECTOR: [u8; 4] = [0x35, 0x24, 0xad, 0x0d];
const GET_MARKET_INFO_SELECTOR: [u8; 4] = [0x3c, 0xc4, 0xfc, 0x4a];
const GET_POSITION_VALUE_SELECTOR: [u8; 4] = [0xe6, 0x95, 0x16, 0x61];
const GET_TVL_SELECTOR: [u8; 4] = [0xee, 0x4c, 0xc8, 0x4c]; // getTVL(uint256)

// Market status
const MARKET_STATUS_OPEN: u8 = 0;
const MARKET_STATUS_CLOSED: u8 = 1;
const MARKET_STATUS_RESOLVED: u8 = 2;

// Market structure (matching original exactly)
#[derive(Clone)]
struct Market {
    creator: [u8; 20],
    creation_time: u64,
    close_time: u64,
    
    // AMM parameters
    k_norm: u64,           // L2 norm constraint (k)
    b_backing: u64,        // Backing amount (b)
    
    // Current AMM state
    current_mean: u64,     // Current consensus mean
    current_variance: u64, // Current consensus variance
    
    // Consensus tracking (new fields for proper aggregation)
    total_weight: u64,     // Sum of all position sizes
    weighted_mean_sum: u128, // Sum of (size * mean) for weighted average
    weighted_m2_sum: u128,   // Sum of (size * (mean^2 + variance)) for variance calc
    
    // Liquidity tracking
    total_lp_shares: u64,  // Total LP tokens
    total_backing: u64,    // Total backing in AMM
    accumulated_fees: u64, // Total fees collected
    
    // Position tracking
    next_position_id: u64, // Next position ID to assign
    total_volume: u64,     // Total trading volume
    
    // Resolution
    status: u8,            // Market status (open/closed/resolved)
    resolution_mean: u64,  // Final mean if resolved
    resolution_variance: u64, // Final variance if resolved
}

// Position structure (matching original exactly)
#[derive(Clone)]
struct Position {
    position_id: u64,
    trader: [u8; 20],
    market_id: u64,
    
    // Entry state
    from_mean: u64,
    from_variance: u64,
    
    // Target state
    to_mean: u64,
    to_variance: u64,
    
    // Size and cost
    size: u64,
    collateral_locked: u64,
    cost_basis: u64,
    
    // Status tracking
    is_open: u8,
    opened_at: u64,
    closed_at: u64,
    exit_value: u64,
    fees_paid: u64,
    realized_pnl: i64,
    claimed: u8,
}

// Helper to encode revert with message
fn encode_revert(message: &str) -> Vec<u8> {
    let mut result = Vec::with_capacity(68 + message.len());
    // Error selector: 0x08c379a0
    result.extend_from_slice(&[0x08, 0xc3, 0x79, 0xa0]);
    // Encode string
    let error_data = encode(&[
        Token::String(message.into()),
    ]);
    result.extend_from_slice(&error_data);
    result
}

// Wei conversion functions (matching original)
fn wei_to_fixed(wei: u64) -> Result<u64, &'static str> {
    wei.checked_div(WEI_TO_FIXED).ok_or("Wei conversion overflow")
}

fn fixed_to_wei(fixed: u64) -> Result<u64, &'static str> {
    fixed.checked_mul(WEI_TO_FIXED).ok_or("Fixed to wei overflow")
}

fn round_wei_for_transfer(wei: u64) -> u64 {
    (wei / MIN_TRANSFER_UNIT) * MIN_TRANSFER_UNIT
}

// Block and timestamp helpers
fn get_block_number() -> u64 {
    let mut block_bytes = [0u8; 8];
    let _ = api::get_storage(
        StorageFlags::empty(),
        BLOCK_NUMBER_KEY,
        &mut &mut block_bytes[..],
    );
    u64::from_le_bytes(block_bytes)
}

fn get_timestamp() -> u64 {
    let mut timestamp_bytes = [0u8; 32];
    api::now(&mut timestamp_bytes);
    let mut timestamp_u64_bytes = [0u8; 8];
    timestamp_u64_bytes.copy_from_slice(&timestamp_bytes[0..8]);
    u64::from_le_bytes(timestamp_u64_bytes)
}

fn u256_bytes(value: u64) -> [u8; 32] {
    let mut result = [0u8; 32];
    result[0..8].copy_from_slice(&value.to_le_bytes());
    result
}

// Storage key generation (matching original format)
fn get_market_key(market_id: u64) -> [u8; 16] {
    let mut key = [0u8; 16];
    key[0..7].copy_from_slice(b"market_");
    key[8..16].copy_from_slice(&market_id.to_le_bytes());
    key
}

fn get_lp_balance_key(market_id: u64, address: &[u8; 20]) -> [u8; 32] {
    let mut key = [0u8; 32];
    key[0..3].copy_from_slice(LP_TOKEN_PREFIX);
    key[3..11].copy_from_slice(&market_id.to_le_bytes());
    key[11..31].copy_from_slice(address);
    key
}

fn get_metadata_key(market_id: u64) -> [u8; 13] {
    let mut key = [0u8; 13];
    key[0..5].copy_from_slice(METADATA_PREFIX);
    key[5..13].copy_from_slice(&market_id.to_le_bytes());
    key
}

fn get_position_key(position_id: u64) -> [u8; 15] {
    let mut key = [0u8; 15];
    key[0..7].copy_from_slice(POSITION_BY_ID_PREFIX);
    key[7..15].copy_from_slice(&position_id.to_le_bytes());
    key
}

fn get_trader_positions_key(trader: &[u8; 20], index: u64) -> [u8; 39] {
    let mut key = [0u8; 39];
    key[0..11].copy_from_slice(TRADER_POSITIONS_PREFIX);
    key[11..31].copy_from_slice(trader);
    key[31..39].copy_from_slice(&index.to_le_bytes());
    key
}

fn get_trader_position_count_key(trader: &[u8; 20]) -> [u8; 31] {
    let mut key = [0u8; 31];
    key[0..11].copy_from_slice(TRADER_POS_COUNT_PREFIX);
    key[11..31].copy_from_slice(trader);
    key
}

// Fixed-point arithmetic (optimized with error handling)
fn mul_fixed(a: u64, b: u64) -> Result<u64, &'static str> {
    let result = (a as u128).saturating_mul(b as u128) / DECIMALS as u128;
    if result > u64::MAX as u128 {
        return Err("Multiplication overflow");
    }
    Ok(result as u64)
}

fn div_fixed(a: u64, b: u64) -> Result<u64, &'static str> {
    if b == 0 {
        return Err("Division by zero");
    }
    let result = (a as u128).saturating_mul(DECIMALS as u128) / b as u128;
    if result > u64::MAX as u128 {
        return Err("Division overflow");
    }
    Ok(result as u64)
}

// Optimized square root with early exit
fn sqrt_fixed(x: u64) -> u64 {
    if x == 0 {
        return 0;
    }
    
    // For fixed-point sqrt: sqrt(x) where x is in fixed-point
    // We want sqrt(x/DECIMALS) * DECIMALS
    // Using the identity: sqrt(a*b) = sqrt(a) * sqrt(b)
    // sqrt(x) = sqrt(x/DECIMALS * DECIMALS) = sqrt(x/DECIMALS) * sqrt(DECIMALS)
    // So: sqrt(x/DECIMALS) * DECIMALS = sqrt(x) * DECIMALS / sqrt(DECIMALS)
    
    let raw_sqrt = integer_sqrt(x);
    
    // sqrt(DECIMALS) = sqrt(10^9) ≈ 31622.776
    // We need: raw_sqrt * DECIMALS / sqrt(DECIMALS)
    // Which is: raw_sqrt * 10^9 / 31622.776
    // Simplifying: raw_sqrt * 31622.776 ≈ raw_sqrt * 31623
    let result = raw_sqrt as u128 * 31623u128;
    if result > u64::MAX as u128 {
        u64::MAX
    } else {
        result as u64
    }
}

fn integer_sqrt(x: u64) -> u64 {
    if x == 0 {
        return 0;
    }
    let mut result = x;
    let mut last_result = 0;
    while result != last_result {
        last_result = result;
        result = (result + x / result) / 2;
    }
    result
}

// Optimized exponential functions
fn exp_neg_fixed(x: u64) -> u64 {
    if x == 0 {
        return DECIMALS;
    }
    // Early exit for large negative exponents
    if x > 20 * DECIMALS {
        return 0;
    }
    let exp_x = exp_fixed(x);
    if exp_x == 0 {
        return 0;
    }
    div_fixed(DECIMALS, exp_x).unwrap_or(0)
}

fn exp_fixed(x: u64) -> u64 {
    if x == 0 {
        return DECIMALS;
    }
    // Early exit for large values
    if x > 20 * DECIMALS {
        return u64::MAX;
    }
    
    let mut result = DECIMALS;
    let mut term = DECIMALS;
    
    for i in 1..TAYLOR_ITERATIONS {
        // Calculate term = term * x / i
        match mul_fixed(term, x) {
            Ok(new_term) => {
                term = new_term / (i as u64);
            }
            Err(_) => {
                // Overflow, term is getting too large
                break;
            }
        }
        
        result = result.saturating_add(term);
        
        // Early exit when term becomes negligible
        if term < 100 {
            break;
        }
    }
    
    result
}

// Optimized normal PDF with caching and early exits
fn normal_pdf(x: u64, mean: u64, variance: u64) -> u64 {
    if variance < MIN_VARIANCE {
        return 0;
    }
    
    let sigma = sqrt_fixed(variance);
    if sigma == 0 {
        return 0;
    }
    
    // Calculate (x - μ)²
    let diff = if x > mean { x - mean } else { mean - x };
    
    // Early exit for values far from mean (>4 sigma)
    let four_sigma = sigma.saturating_mul(4);
    if diff > four_sigma {
        return 0;
    }
    
    let diff_squared = mul_fixed(diff, diff).unwrap_or(u64::MAX);
    
    // Calculate exponent: -(x-μ)²/(2σ²)
    let two_variance = variance.saturating_mul(2);
    let exponent = div_fixed(diff_squared, two_variance).unwrap_or(u64::MAX);
    
    // Optimization: if exponent is too large, PDF is effectively 0
    if exponent > 10 * DECIMALS {
        return 0;
    }
    
    // Calculate e^(-exponent)
    let exp_value = exp_neg_fixed(exponent);
    
    // Calculate normalization: 1/(σ√(2π))
    // For very large sigma, this can underflow, so handle carefully
    let sigma_sqrt_2pi = mul_fixed(sigma, SQRT_2PI_FIXED).unwrap_or(u64::MAX);
    if sigma_sqrt_2pi == 0 || sigma_sqrt_2pi == u64::MAX {
        return 0;  // Distribution too wide to represent
    }
    let normalization = div_fixed(DECIMALS, sigma_sqrt_2pi).unwrap_or(0);
    
    // Return PDF value
    mul_fixed(normalization, exp_value).unwrap_or(0)
}

// Error function approximation (Abramowitz and Stegun)
fn erf_fixed(x: u64) -> u64 {
    // Using Abramowitz and Stegun approximation
    // erf(x) ≈ 1 - (a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵)e^(-x²)
    // where t = 1/(1 + px), p = 0.3275911
    
    let p = 327_591_100; // 0.3275911 in fixed point
    let a1 = 254_829_592; // 0.254829592
    let a2 = 284_496_736; // -0.284496736 (handle sign separately)
    let a3 = 1_421_413_741; // 1.421413741
    let a4 = 1_453_152_027; // -1.453152027 (handle sign separately)
    let a5 = 1_061_405_429; // 1.061405429
    
    // t = 1/(1 + px)
    let px = mul_fixed(p, x).unwrap_or(u64::MAX);
    let t = div_fixed(DECIMALS, DECIMALS + px).unwrap_or(0);
    
    // Calculate polynomial
    let t2 = mul_fixed(t, t).unwrap_or(0);
    let t3 = mul_fixed(t2, t).unwrap_or(0);
    let t4 = mul_fixed(t3, t).unwrap_or(0);
    let t5 = mul_fixed(t4, t).unwrap_or(0);
    
    // Calculate polynomial value with sign handling
    let poly = mul_fixed(a1, t).unwrap_or(0)
        .saturating_sub(mul_fixed(a2, t2).unwrap_or(0))
        .saturating_add(mul_fixed(a3, t3).unwrap_or(0))
        .saturating_sub(mul_fixed(a4, t4).unwrap_or(0))
        .saturating_add(mul_fixed(a5, t5).unwrap_or(0));
    
    // Calculate e^(-x²)
    let x_squared = mul_fixed(x, x).unwrap_or(u64::MAX);
    let exp_neg_x2 = exp_neg_fixed(x_squared);
    
    // Calculate result
    let result = mul_fixed(poly, exp_neg_x2).unwrap_or(0);
    
    // erf(x) = 1 - result
    DECIMALS.saturating_sub(result)
}

// Normal CDF using error function
fn normal_cdf(x: u64, mean: u64, variance: u64) -> u64 {
    if variance < MIN_VARIANCE {
        return if x >= mean { DECIMALS } else { 0 };
    }
    
    let sigma = sqrt_fixed(variance);
    // sqrt(2) ≈ 1.414213562 in fixed point
    const SQRT_2_FIXED: u64 = 1_414_213_562;
    let sqrt_2_sigma = mul_fixed(SQRT_2_FIXED, sigma).unwrap_or(u64::MAX);
    
    // Standardize: z = (x - μ) / (σ√2)
    let z = if x >= mean {
        div_fixed(x - mean, sqrt_2_sigma).unwrap_or(u64::MAX)
    } else {
        // Handle negative z by using symmetry: Φ(-z) = 1 - Φ(z)
        let z_pos = div_fixed(mean - x, sqrt_2_sigma).unwrap_or(u64::MAX);
        let erf_z = erf_fixed(z_pos);
        return div_fixed(DECIMALS - erf_z, 2 * DECIMALS).unwrap_or(DECIMALS / 2);
    };
    
    // CDF = (1 + erf(z)) / 2
    let erf_z = erf_fixed(z);
    div_fixed(DECIMALS + erf_z, 2 * DECIMALS).unwrap_or(DECIMALS / 2)
}

// L2 norm calculation for normal distributions
// NOT USED ANYMORE - kept for compatibility
fn calculate_l2_norm_normal(variance: u64) -> Result<u64, &'static str> {
    if variance < MIN_VARIANCE {
        return Err("Variance too small");
    }
    
    // This was incorrect - keeping for compatibility
    let sigma = sqrt_fixed(variance);
    let two_sigma = mul_fixed(2 * DECIMALS, sigma)?;
    let two_sigma_sqrt_pi = mul_fixed(two_sigma, SQRT_PI_FIXED)?;
    div_fixed(DECIMALS, two_sigma_sqrt_pi)
}

// Calculate lambda from k_norm and variance
fn calculate_lambda(k_norm: u64, variance: u64) -> Result<u64, &'static str> {
    if variance < MIN_VARIANCE {
        return Err("Variance too small");
    }
    
    // According to Paradigm: λ = k·σ·√(2π)
    let sigma = sqrt_fixed(variance);
    let sigma_sqrt_2pi = mul_fixed(sigma, SQRT_2PI_FIXED)?;
    mul_fixed(k_norm, sigma_sqrt_2pi)
}

// Calculate f_max for backing constraint checking
fn calculate_f_max(k_norm: u64, variance: u64) -> Result<u64, &'static str> {
    // For normal distribution with L2 norm k:
    // f_max = λ/(σ√(2π)) where λ = k·σ·√(2π)
    // Therefore f_max = k
    Ok(k_norm)
}

// Calculate minimum variance for given k_norm and backing
fn calculate_min_variance(k_norm: u64, b_backing: u64) -> Result<u64, &'static str> {
    if b_backing == 0 {
        return Err("Backing is zero");
    }
    
    // Since f_max = k, we need k ≤ b
    // Therefore min_variance can be any positive value as long as k ≤ b
    // We'll use a small default value
    Ok(MIN_VARIANCE)
}

// Calculate consensus mean and variance from weighted sums
fn calculate_consensus(total_weight: u64, weighted_mean_sum: u128, weighted_m2_sum: u128) -> Result<(u64, u64), &'static str> {
    if total_weight == 0 {
        return Err("No positions in market");
    }
    
    // Calculate weighted mean: sum(weight * mean) / sum(weight)
    let consensus_mean = (weighted_mean_sum / total_weight as u128) as u64;
    
    // Calculate weighted variance using the formula:
    // Var = (sum(weight * (mean^2 + variance)) / sum(weight)) - consensus_mean^2
    let mean_squared = mul_fixed(consensus_mean, consensus_mean)?;
    let weighted_avg_m2 = (weighted_m2_sum / total_weight as u128) as u64;
    
    if weighted_avg_m2 < mean_squared {
        // Handle numerical issues - return minimum variance
        return Ok((consensus_mean, MIN_VARIANCE));
    }
    
    let consensus_variance = weighted_avg_m2 - mean_squared;
    
    // Ensure minimum variance
    let final_variance = if consensus_variance < MIN_VARIANCE {
        MIN_VARIANCE
    } else {
        consensus_variance
    };
    
    Ok((consensus_mean, final_variance))
}

// IMPROVED: Comprehensive backing constraint validation
fn validate_backing_constraint(
    k_norm: u64,
    b_backing: u64,
    new_mean: u64,
    new_variance: u64,
    current_mean: u64,
    current_variance: u64
) -> Result<bool, &'static str> {
    // Check 1: Basic f_max constraint
    let f_max = calculate_f_max(k_norm, new_variance)?;
    if f_max > b_backing {
        return Ok(false);
    }
    
    // Check 2: Extreme mean shift protection
    // Prevent trades that shift mean by more than 3 standard deviations
    let new_sigma = sqrt_fixed(new_variance);
    let current_sigma = sqrt_fixed(current_variance);
    let max_sigma = if new_sigma > current_sigma { new_sigma } else { current_sigma };
    let mean_shift = if new_mean > current_mean { 
        new_mean - current_mean 
    } else { 
        current_mean - new_mean 
    };
    
    if mean_shift > max_sigma.saturating_mul(3) {
        return Ok(false);
    }
    
    // Check 3: Variance change limit
    // Prevent extreme variance changes (more than 5x increase or decrease)
    if new_variance > current_variance.saturating_mul(5) || 
       new_variance < current_variance / 5 {
        return Ok(false);
    }
    
    // Check 4: Minimum variance requirement
    if new_variance < MIN_VARIANCE {
        return Ok(false);
    }
    
    Ok(true)
}

// Calculate expected value (mean for normal distribution)
fn calculate_expected_value(mean: u64, _variance: u64) -> u64 {
    mean
}

// Get distribution bounds (3 sigma)
fn get_distribution_bounds(mean: u64, variance: u64) -> (u64, u64) {
    let sigma = sqrt_fixed(variance);
    let three_sigma = sigma.saturating_mul(3);
    let lower = mean.saturating_sub(three_sigma);
    let upper = mean.saturating_add(three_sigma);
    (lower, upper)
}

// Calculate AMM holdings at point x
fn calculate_amm_holdings(x: u64, market: &Market) -> u64 {
    let lambda = calculate_lambda(market.k_norm, market.current_variance).unwrap_or(0);
    let pdf_value = normal_pdf(x, market.current_mean, market.current_variance);
    let f_value = mul_fixed(lambda, pdf_value).unwrap_or(0);
    
    // Cap at backing
    let capped_f_value = if f_value > market.b_backing {
        market.b_backing
    } else {
        f_value
    };
    
    // AMM holds (b - f(x))
    market.b_backing.saturating_sub(capped_f_value)
}

// Calculate L2 norm of difference between two scaled normal distributions
fn calculate_l2_norm_difference(
    mean1: u64, variance1: u64, lambda1: u64,
    mean2: u64, variance2: u64, lambda2: u64
) -> Result<u64, &'static str> {
    // For two normal distributions:
    // ||λ₁N(μ₁,σ₁²) - λ₂N(μ₂,σ₂²)||₂² = 
    //   λ₁²/(2σ₁√π) + λ₂²/(2σ₂√π) - 2λ₁λ₂/(√(2π(σ₁²+σ₂²))) * exp(-(μ₁-μ₂)²/(2(σ₁²+σ₂²)))
    
    let sigma1 = sqrt_fixed(variance1);
    let sigma2 = sqrt_fixed(variance2);
    
    // First term: λ₁²/(2σ₁√π)
    let lambda1_squared = mul_fixed(lambda1, lambda1)?;
    let two_sigma1_sqrt_pi = mul_fixed(mul_fixed(2 * DECIMALS, sigma1)?, SQRT_PI_FIXED)?;
    let term1 = div_fixed(lambda1_squared, two_sigma1_sqrt_pi)?;
    
    // Second term: λ₂²/(2σ₂√π)
    let lambda2_squared = mul_fixed(lambda2, lambda2)?;
    let two_sigma2_sqrt_pi = mul_fixed(mul_fixed(2 * DECIMALS, sigma2)?, SQRT_PI_FIXED)?;
    let term2 = div_fixed(lambda2_squared, two_sigma2_sqrt_pi)?;
    
    // Third term: 2λ₁λ₂/(√(2π(σ₁²+σ₂²))) * exp(-(μ₁-μ₂)²/(2(σ₁²+σ₂²)))
    let variance_sum = variance1 + variance2;
    let mean_diff = if mean1 > mean2 { mean1 - mean2 } else { mean2 - mean1 };
    let mean_diff_squared = mul_fixed(mean_diff, mean_diff)?;
    let two_variance_sum = variance_sum.saturating_mul(2);
    let exponent = div_fixed(mean_diff_squared, two_variance_sum)?;
    let exp_term = exp_neg_fixed(exponent);
    
    let lambda_product = mul_fixed(lambda1, lambda2)?;
    let two_lambda_product = lambda_product.saturating_mul(2);
    let sqrt_2pi_variance_sum = sqrt_fixed(mul_fixed(mul_fixed(2 * DECIMALS, PI_FIXED)?, variance_sum)?);
    let coefficient = div_fixed(two_lambda_product, sqrt_2pi_variance_sum)?;
    let term3 = mul_fixed(coefficient, exp_term)?;
    
    // ||f₁ - f₂||₂² = term1 + term2 - term3
    let sum = term1.saturating_add(term2);
    let l2_norm_squared = sum.saturating_sub(term3);
    
    // Return ||f₁ - f₂||₂ = sqrt(l2_norm_squared)
    Ok(sqrt_fixed(l2_norm_squared))
}

// Calculate trade cost with practical scaling
fn calculate_trade_cost(
    k_norm: u64,
    from_mean: u64, from_variance: u64,
    to_mean: u64, to_variance: u64,
    size: u64
) -> Result<(u64, u64, u64), &'static str> {
    // Calculate lambdas
    let lambda_from = calculate_lambda(k_norm, from_variance)?;
    let lambda_to = calculate_lambda(k_norm, to_variance)?;
    
    // Calculate ||f_to - f_from||₂ using the correct formula
    let l2_diff = calculate_l2_norm_difference(
        from_mean, from_variance, lambda_from,
        to_mean, to_variance, lambda_to
    )?;
    
    // IMPROVED: Scale the L2 diff by k_norm to normalize costs
    // This makes costs proportional to the change relative to market capacity
    let normalized_l2_diff = div_fixed(l2_diff, k_norm)?;
    
    // IMPROVED: Use a square root scaling for size to reduce cost/size ratio
    // This provides better incentives for larger trades
    let size_factor = sqrt_fixed(size);
    
    // Base cost = normalized_l2_diff * sqrt(size) * adjustment_factor
    // The adjustment factor (0.1) brings costs to reasonable levels
    let adjustment_factor = DECIMALS / 10; // 0.1
    let scaled_diff = mul_fixed(normalized_l2_diff, adjustment_factor)?;
    let base_cost = mul_fixed(scaled_diff, size_factor)?;
    
    // Fee: 0.3% (3/1000) of base cost
    let fee = (base_cost * 3) / 1000;
    
    // Calculate collateral required
    let collateral_required = calculate_collateral_requirement(
        k_norm, from_mean, from_variance, to_mean, to_variance, size
    )?;
    
    Ok((base_cost + fee, fee, collateral_required))
}

// IMPROVED: More accurate collateral calculation with additional sampling points
fn calculate_collateral_requirement(
    k_norm: u64,
    from_mean: u64, from_variance: u64,
    to_mean: u64, to_variance: u64,
    size: u64
) -> Result<u64, &'static str> {
    // Find the minimum of g(x) - f(x) where g = from distribution, f = to distribution
    // For normal distributions, this minimum occurs at specific points
    
    let lambda_from = calculate_lambda(k_norm, from_variance)?;
    let lambda_to = calculate_lambda(k_norm, to_variance)?;
    
    // IMPROVED: Sample at more points for better accuracy
    let sigma_from = sqrt_fixed(from_variance);
    let sigma_to = sqrt_fixed(to_variance);
    
    // Critical points include means, inflection points, and intersections
    let mut points = Vec::with_capacity(20);
    
    // Add means and midpoint
    points.push(from_mean);
    points.push(to_mean);
    points.push((from_mean + to_mean) / 2);
    
    // Special handling for equal variances - add the critical point
    if from_variance == to_variance {
        // When variances are equal, the minimum of g(x) - f(x) is at the mean of means
        let x_min = (from_mean + to_mean) / 2;
        points.push(x_min);
    }
    
    // Add ±1, ±2, ±3 sigma points for both distributions
    for i in 1..=3 {
        let i_sigma_from = sigma_from.saturating_mul(i);
        let i_sigma_to = sigma_to.saturating_mul(i);
        
        points.push(from_mean.saturating_sub(i_sigma_from));
        points.push(from_mean.saturating_add(i_sigma_from));
        points.push(to_mean.saturating_sub(i_sigma_to));
        points.push(to_mean.saturating_add(i_sigma_to));
    }
    
    let mut max_deficit = 0u64;
    
    // Sample at all critical points
    for &x in &points {
        let g_value = mul_fixed(lambda_from, normal_pdf(x, from_mean, from_variance))?;
        let f_value = mul_fixed(lambda_to, normal_pdf(x, to_mean, to_variance))?;
        
        if f_value > g_value {
            let deficit = f_value - g_value;
            if deficit > max_deficit {
                max_deficit = deficit;
            }
        }
    }
    
    // IMPROVED: Add a safety margin of 10% to collateral
    let collateral_with_margin = max_deficit.saturating_mul(11) / 10;
    
    // Calculate collateral based on max deficit
    let calculated_collateral = mul_fixed(collateral_with_margin, size)?;
    
    // Ensure minimum collateral of 0.1% of size to prevent edge cases
    let min_collateral = size / 1000;
    
    Ok(calculated_collateral.max(min_collateral))
}

// Calculate position value according to Paradigm research
fn calculate_position_value(
    position: &Position,
    current_mean: u64,
    current_variance: u64,
    k_norm: u64
) -> Result<u64, &'static str> {
    // According to Paradigm: Value = ∫(f_target - f_from) * f_current dx
    // This represents the dot product of the position with current distribution
    
    // Calculate lambdas for scaling distributions to meet L2 norm
    let current_lambda = calculate_lambda(k_norm, current_variance)?;
    let from_lambda = calculate_lambda(k_norm, position.from_variance)?;
    let to_lambda = calculate_lambda(k_norm, position.to_variance)?;
    
    // Get standard deviations for bounds
    let current_std = sqrt_fixed(current_variance);
    let from_std = sqrt_fixed(position.from_variance);
    let to_std = sqrt_fixed(position.to_variance);
    
    // Integration bounds: use wider of the distributions to capture all mass
    let max_std = if current_std > from_std {
        if current_std > to_std { current_std } else { to_std }
    } else {
        if from_std > to_std { from_std } else { to_std }
    };
    
    // IMPROVED: Use adaptive bounds based on distribution scale
    // For very wide distributions, we need tighter bounds to avoid numerical issues
    let sigma_multiplier = if max_std > 1000 * DECIMALS {
        // For extremely wide distributions (sigma > 1000), use 1 sigma
        DECIMALS
    } else if max_std > 100 * DECIMALS {
        // For wide distributions (sigma > 100), use 2 sigma
        2 * DECIMALS
    } else {
        // For normal distributions, use 3 sigma
        3 * DECIMALS
    };
    
    let integration_range = mul_fixed(max_std, sigma_multiplier)?;
    
    // Find the center point for integration - use the mean of all three distributions
    let center = (position.from_mean + position.to_mean + current_mean) / 3;
    
    let lower_bound = center.saturating_sub(integration_range);
    let upper_bound = center.saturating_add(integration_range);
    
    // IMPROVED: Adaptive number of steps based on range and variance
    // More steps for wider distributions to maintain accuracy
    let num_steps = if max_std > 1000 * DECIMALS {
        100  // 5x more steps for extreme distributions
    } else if max_std > 100 * DECIMALS {
        40   // 2x more steps for wide distributions
    } else {
        INTEGRATION_STEPS  // Default 20 steps
    };
    let range = upper_bound.saturating_sub(lower_bound);
    
    if range == 0 {
        return Ok(position.cost_basis);
    }
    
    // Calculate dx safely
    let dx = if range < num_steps {
        // Range is too small, use minimum step
        1
    } else {
        range / num_steps
    };
    
    let mut integral_sum = 0i128; // Use i128 for signed arithmetic
    
    // Compute ∫(f_to - f_from) * f_current dx
    for i in 0..=num_steps {
        // Calculate x position for this step
        let step_offset = if i == 0 {
            0
        } else if i == num_steps {
            range
        } else {
            // Avoid overflow: (range * i) / num_steps
            ((range as u128 * i as u128) / num_steps as u128) as u64
        };
        let x = lower_bound.saturating_add(step_offset);
        
        // Calculate f_from(x) = λ_from * N(x; μ_from, σ²_from)
        let pdf_from = normal_pdf(x, position.from_mean, position.from_variance);
        let f_from = mul_fixed(from_lambda, pdf_from)?;
        
        // Calculate f_to(x) = λ_to * N(x; μ_to, σ²_to)
        let pdf_to = normal_pdf(x, position.to_mean, position.to_variance);
        let f_to = mul_fixed(to_lambda, pdf_to)?;
        
        // Calculate f_current(x) = λ_current * N(x; μ_current, σ²_current)
        let pdf_current = normal_pdf(x, current_mean, current_variance);
        let f_current = mul_fixed(current_lambda, pdf_current)?;
        
        // Position represents (f_to - f_from)
        let position_value = f_to as i128 - f_from as i128;
        
        // Multiply by current distribution
        let product = (position_value * f_current as i128) / DECIMALS as i128;
        
        // Trapezoidal rule: add with appropriate weight
        if i == 0 || i == num_steps {
            integral_sum += product / 2;
        } else {
            integral_sum += product;
        }
    }
    
    // Complete the integral: multiply by dx
    // dx is in fixed-point units, so we need to scale appropriately
    let integral_result = if range < num_steps {
        // We used dx = 1, which is 1/DECIMALS in fixed-point
        integral_sum / DECIMALS as i128
    } else {
        // Normal case: dx = range/num_steps
        (integral_sum * dx as i128) / DECIMALS as i128
    };
    
    // The integral represents the P&L per unit of position size
    // Calculate the total P&L by scaling with position size
    let position_pnl = if integral_result >= 0 {
        // Positive P&L
        let profit_per_unit = integral_result as u64;
        let total_profit = mul_fixed(profit_per_unit, position.size)?;
        total_profit as i64
    } else {
        // Negative P&L
        let loss_per_unit = (-integral_result) as u64;
        let total_loss = mul_fixed(loss_per_unit, position.size)?;
        -(total_loss as i64)
    };
    
    // SAFETY CHECK: Prevent unrealistic P&L
    // P&L should not exceed a reasonable multiple of the cost basis
    // This catches numerical errors from extreme distributions
    let max_reasonable_pnl = (position.cost_basis as i64) * 100; // Max 100x return
    let clamped_pnl = if position_pnl > max_reasonable_pnl {
        max_reasonable_pnl
    } else if position_pnl < -max_reasonable_pnl {
        -max_reasonable_pnl
    } else {
        position_pnl
    };
    
    // Calculate the actual payout: collateral locked ± P&L
    // This represents what the trader receives when closing the position
    let payout = if clamped_pnl >= 0 {
        // Profit case: return collateral + profit
        position.collateral_locked.saturating_add(clamped_pnl as u64)
    } else {
        // Loss case: return collateral - loss (minimum 0)
        let loss = (-clamped_pnl) as u64;
        position.collateral_locked.saturating_sub(loss)
    };
    
    // Additional safety: payout should not exceed backing
    // This prevents draining the AMM through numerical errors
    let market_backing = match load_market(position.market_id) {
        Some(m) => m.total_backing,
        None => position.collateral_locked * 10, // Conservative default
    };
    
    let final_payout = payout.min(market_backing / 2); // Max 50% of backing
    
    Ok(final_payout)
}

// Market storage operations
fn load_market(market_id: u64) -> Option<Market> {
    let key = get_market_key(market_id);
    let mut buffer = [0u8; MAX_STORAGE_VALUE];
    
    let _ = api::get_storage(
        StorageFlags::empty(),
        &key,
        &mut &mut buffer[..],
    );
    
    // Check if data exists
    if buffer[0] != 0 || buffer[1] != 0 {
        let mut market = Market {
            creator: [0u8; 20],
            creation_time: 0,
            close_time: 0,
            k_norm: 0,
            b_backing: 0,
            current_mean: 0,
            current_variance: 0,
            total_weight: 0,
            weighted_mean_sum: 0,
            weighted_m2_sum: 0,
            total_lp_shares: 0,
            total_backing: 0,
            accumulated_fees: 0,
            next_position_id: 0,
            total_volume: 0,
            status: 0,
            resolution_mean: 0,
            resolution_variance: 0,
        };
        
        let mut offset = 0;
        
        // Deserialize in same order as original
        market.creator.copy_from_slice(&buffer[offset..offset+20]);
        offset += 20;
        
        market.creation_time = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
        offset += 8;
        
        market.close_time = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
        offset += 8;
        
        market.k_norm = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
        offset += 8;
        
        market.b_backing = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
        offset += 8;
        
        market.current_mean = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
        offset += 8;
        
        market.current_variance = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
        offset += 8;
        
        // Check if this is an old market (without consensus fields)
        // If offset + 40 exceeds expected old size, it's a new market
        if buffer.len() > offset + 40 {
            // New market with consensus fields
            market.total_weight = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap_or([0u8; 8]));
            offset += 8;
            
            market.weighted_mean_sum = u128::from_le_bytes(buffer[offset..offset+16].try_into().unwrap_or([0u8; 16]));
            offset += 16;
            
            market.weighted_m2_sum = u128::from_le_bytes(buffer[offset..offset+16].try_into().unwrap_or([0u8; 16]));
            offset += 16;
        }
        
        market.total_lp_shares = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
        offset += 8;
        
        market.total_backing = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
        offset += 8;
        
        market.accumulated_fees = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
        offset += 8;
        
        market.next_position_id = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
        offset += 8;
        
        market.total_volume = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
        offset += 8;
        
        market.status = buffer[offset];
        offset += 1;
        
        market.resolution_mean = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
        offset += 8;
        
        market.resolution_variance = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
        
        Some(market)
    } else {
        None
    }
}

fn save_market(market_id: u64, market: &Market) {
    let key = get_market_key(market_id);
    let mut buffer = [0u8; 240]; // Increased for new fields
    let mut offset = 0;
    
    // Serialize in same order
    buffer[offset..offset+20].copy_from_slice(&market.creator);
    offset += 20;
    
    buffer[offset..offset+8].copy_from_slice(&market.creation_time.to_le_bytes());
    offset += 8;
    
    buffer[offset..offset+8].copy_from_slice(&market.close_time.to_le_bytes());
    offset += 8;
    
    buffer[offset..offset+8].copy_from_slice(&market.k_norm.to_le_bytes());
    offset += 8;
    
    buffer[offset..offset+8].copy_from_slice(&market.b_backing.to_le_bytes());
    offset += 8;
    
    buffer[offset..offset+8].copy_from_slice(&market.current_mean.to_le_bytes());
    offset += 8;
    
    buffer[offset..offset+8].copy_from_slice(&market.current_variance.to_le_bytes());
    offset += 8;
    
    // New consensus tracking fields
    buffer[offset..offset+8].copy_from_slice(&market.total_weight.to_le_bytes());
    offset += 8;
    
    buffer[offset..offset+16].copy_from_slice(&market.weighted_mean_sum.to_le_bytes());
    offset += 16;
    
    buffer[offset..offset+16].copy_from_slice(&market.weighted_m2_sum.to_le_bytes());
    offset += 16;
    
    buffer[offset..offset+8].copy_from_slice(&market.total_lp_shares.to_le_bytes());
    offset += 8;
    
    buffer[offset..offset+8].copy_from_slice(&market.total_backing.to_le_bytes());
    offset += 8;
    
    buffer[offset..offset+8].copy_from_slice(&market.accumulated_fees.to_le_bytes());
    offset += 8;
    
    buffer[offset..offset+8].copy_from_slice(&market.next_position_id.to_le_bytes());
    offset += 8;
    
    buffer[offset..offset+8].copy_from_slice(&market.total_volume.to_le_bytes());
    offset += 8;
    
    buffer[offset] = market.status;
    offset += 1;
    
    buffer[offset..offset+8].copy_from_slice(&market.resolution_mean.to_le_bytes());
    offset += 8;
    
    buffer[offset..offset+8].copy_from_slice(&market.resolution_variance.to_le_bytes());
    offset += 8;
    
    api::set_storage(
        StorageFlags::empty(),
        &key,
        &buffer[..offset],
    );
}

// Position storage operations
fn save_position(position: &Position) {
    let key = get_position_key(position.position_id);
    let mut buffer = [0u8; 250];
    let mut offset = 0;
    
    // Serialize all fields
    buffer[offset..offset+8].copy_from_slice(&position.position_id.to_le_bytes());
    offset += 8;
    
    buffer[offset..offset+20].copy_from_slice(&position.trader);
    offset += 20;
    
    buffer[offset..offset+8].copy_from_slice(&position.market_id.to_le_bytes());
    offset += 8;
    
    buffer[offset..offset+8].copy_from_slice(&position.from_mean.to_le_bytes());
    offset += 8;
    
    buffer[offset..offset+8].copy_from_slice(&position.from_variance.to_le_bytes());
    offset += 8;
    
    buffer[offset..offset+8].copy_from_slice(&position.to_mean.to_le_bytes());
    offset += 8;
    
    buffer[offset..offset+8].copy_from_slice(&position.to_variance.to_le_bytes());
    offset += 8;
    
    buffer[offset..offset+8].copy_from_slice(&position.size.to_le_bytes());
    offset += 8;
    
    buffer[offset..offset+8].copy_from_slice(&position.collateral_locked.to_le_bytes());
    offset += 8;
    
    buffer[offset..offset+8].copy_from_slice(&position.cost_basis.to_le_bytes());
    offset += 8;
    
    buffer[offset] = position.is_open;
    offset += 1;
    
    buffer[offset..offset+8].copy_from_slice(&position.opened_at.to_le_bytes());
    offset += 8;
    
    buffer[offset..offset+8].copy_from_slice(&position.closed_at.to_le_bytes());
    offset += 8;
    
    buffer[offset..offset+8].copy_from_slice(&position.exit_value.to_le_bytes());
    offset += 8;
    
    buffer[offset..offset+8].copy_from_slice(&position.fees_paid.to_le_bytes());
    offset += 8;
    
    buffer[offset..offset+8].copy_from_slice(&position.realized_pnl.to_le_bytes());
    offset += 8;
    
    buffer[offset] = position.claimed;
    offset += 1;
    
    api::set_storage(
        StorageFlags::empty(),
        &key,
        &buffer[..offset],
    );
}

fn load_position(position_id: u64) -> Option<Position> {
    let key = get_position_key(position_id);
    let mut buffer = [0u8; 250];
    
    let result = api::get_storage(
        StorageFlags::empty(),
        &key,
        &mut &mut buffer[..],
    );
    
    if result.is_err() {
        return None;
    }
    
    // Check if buffer has data by checking trader address (bytes 8-27)
    // A valid position will have a non-zero trader address
    let has_data = buffer[8..28].iter().any(|&b| b != 0);
    if !has_data {
        return None;
    }
    
    let mut offset = 0;
    
    // Deserialize all fields - using safe slice operations
    let position_id = match buffer.get(offset..offset+8) {
        Some(slice) => match slice.try_into() {
            Ok(arr) => u64::from_le_bytes(arr),
            Err(_) => return None,
        },
        None => return None,
    };
    offset += 8;
    
    let mut trader = [0u8; 20];
    match buffer.get(offset..offset+20) {
        Some(slice) => trader.copy_from_slice(slice),
        None => return None,
    }
    offset += 20;
    
    let market_id = match buffer.get(offset..offset+8) {
        Some(slice) => match slice.try_into() {
            Ok(arr) => u64::from_le_bytes(arr),
            Err(_) => return None,
        },
        None => return None,
    };
    offset += 8;
    
    let from_mean = match buffer.get(offset..offset+8) {
        Some(slice) => match slice.try_into() {
            Ok(arr) => u64::from_le_bytes(arr),
            Err(_) => return None,
        },
        None => return None,
    };
    offset += 8;
    
    let from_variance = match buffer.get(offset..offset+8) {
        Some(slice) => match slice.try_into() {
            Ok(arr) => u64::from_le_bytes(arr),
            Err(_) => return None,
        },
        None => return None,
    };
    offset += 8;
    
    let to_mean = match buffer.get(offset..offset+8) {
        Some(slice) => match slice.try_into() {
            Ok(arr) => u64::from_le_bytes(arr),
            Err(_) => return None,
        },
        None => return None,
    };
    offset += 8;
    
    let to_variance = match buffer.get(offset..offset+8) {
        Some(slice) => match slice.try_into() {
            Ok(arr) => u64::from_le_bytes(arr),
            Err(_) => return None,
        },
        None => return None,
    };
    offset += 8;
    
    let size = match buffer.get(offset..offset+8) {
        Some(slice) => match slice.try_into() {
            Ok(arr) => u64::from_le_bytes(arr),
            Err(_) => return None,
        },
        None => return None,
    };
    offset += 8;
    
    let collateral_locked = match buffer.get(offset..offset+8) {
        Some(slice) => match slice.try_into() {
            Ok(arr) => u64::from_le_bytes(arr),
            Err(_) => return None,
        },
        None => return None,
    };
    offset += 8;
    
    let cost_basis = match buffer.get(offset..offset+8) {
        Some(slice) => match slice.try_into() {
            Ok(arr) => u64::from_le_bytes(arr),
            Err(_) => return None,
        },
        None => return None,
    };
    offset += 8;
    
    let is_open = match buffer.get(offset) {
        Some(&b) => b,
        None => return None,
    };
    offset += 1;
    
    let opened_at = match buffer.get(offset..offset+8) {
        Some(slice) => match slice.try_into() {
            Ok(arr) => u64::from_le_bytes(arr),
            Err(_) => return None,
        },
        None => return None,
    };
    offset += 8;
    
    let closed_at = match buffer.get(offset..offset+8) {
        Some(slice) => match slice.try_into() {
            Ok(arr) => u64::from_le_bytes(arr),
            Err(_) => return None,
        },
        None => return None,
    };
    offset += 8;
    
    let exit_value = match buffer.get(offset..offset+8) {
        Some(slice) => match slice.try_into() {
            Ok(arr) => u64::from_le_bytes(arr),
            Err(_) => return None,
        },
        None => return None,
    };
    offset += 8;
    
    let fees_paid = match buffer.get(offset..offset+8) {
        Some(slice) => match slice.try_into() {
            Ok(arr) => u64::from_le_bytes(arr),
            Err(_) => return None,
        },
        None => return None,
    };
    offset += 8;
    
    let realized_pnl = match buffer.get(offset..offset+8) {
        Some(slice) => match slice.try_into() {
            Ok(arr) => i64::from_le_bytes(arr),
            Err(_) => return None,
        },
        None => return None,
    };
    offset += 8;
    
    let claimed = match buffer.get(offset) {
        Some(&b) => b,
        None => return None,
    };
    
    Some(Position {
        position_id,
        trader,
        market_id,
        from_mean,
        from_variance,
        to_mean,
        to_variance,
        size,
        collateral_locked,
        cost_basis,
        is_open,
        opened_at,
        closed_at,
        exit_value,
        fees_paid,
        realized_pnl,
        claimed,
    })
}

// Add trader position tracking
fn add_trader_position(trader: &[u8; 20], position_id: u64) {
    let count_key = get_trader_position_count_key(trader);
    let mut count_bytes = [0u8; 8];
    
    let _ = api::get_storage(
        StorageFlags::empty(),
        &count_key,
        &mut &mut count_bytes[..],
    );
    
    let count = u64::from_le_bytes(count_bytes);
    
    // Store position ID at index
    let pos_key = get_trader_positions_key(trader, count);
    api::set_storage(
        StorageFlags::empty(),
        &pos_key,
        &position_id.to_le_bytes(),
    );
    
    // Update count
    let new_count = count + 1;
    api::set_storage(
        StorageFlags::empty(),
        &count_key,
        &new_count.to_le_bytes(),
    );
}

// Save market metadata
fn save_market_metadata(market_id: u64, title: &str, description: &str, resolution_criteria: &str) {
    let key = get_metadata_key(market_id);
    
    let title_bytes = title.as_bytes();
    let desc_bytes = description.as_bytes();
    let criteria_bytes = resolution_criteria.as_bytes();
    
    // Limit string lengths
    let title_len = (title_bytes.len().min(255)) as u8;
    let desc_len = (desc_bytes.len().min(255)) as u8;
    let criteria_len = (criteria_bytes.len().min(255)) as u8;
    
    let total_len = 3 + title_len as usize + desc_len as usize + criteria_len as usize;
    
    let mut buffer = [0u8; MAX_STORAGE_VALUE];
    
    if total_len > MAX_STORAGE_VALUE {
        return; // Silently fail instead of panic
    }
    
    // Store lengths
    buffer[0] = title_len;
    buffer[1] = desc_len;
    buffer[2] = criteria_len;
    
    let mut offset = 3;
    
    // Store strings
    buffer[offset..offset + title_len as usize].copy_from_slice(&title_bytes[..title_len as usize]);
    offset += title_len as usize;
    
    buffer[offset..offset + desc_len as usize].copy_from_slice(&desc_bytes[..desc_len as usize]);
    offset += desc_len as usize;
    
    buffer[offset..offset + criteria_len as usize].copy_from_slice(&criteria_bytes[..criteria_len as usize]);
    
    api::set_storage(
        StorageFlags::empty(),
        &key,
        &buffer[..total_len],
    );
}

// Handler functions for all operations

fn handle_initialize() -> Vec<u8> {
    let mut caller = [0u8; 20];
    api::caller(&mut caller);
    
    // Check if already initialized
    let mut initialized = [0u8; 1];
    let _ = api::get_storage(
        StorageFlags::empty(),
        INITIALIZED_KEY,
        &mut &mut initialized[..],
    );
    
    if initialized[0] != 0 {
        return encode_revert("Already initialized");
    }
    
    // Set owner
    api::set_storage(
        StorageFlags::empty(),
        OWNER_KEY,
        &caller,
    );
    
    // Mark as initialized
    api::set_storage(
        StorageFlags::empty(),
        INITIALIZED_KEY,
        &[1u8],
    );
    
    // Initialize market count
    api::set_storage(
        StorageFlags::empty(),
        MARKET_COUNT_KEY,
        &0u64.to_le_bytes(),
    );
    
    // Initialize position count
    api::set_storage(
        StorageFlags::empty(),
        POSITION_COUNT_KEY,
        &0u64.to_le_bytes(),
    );
    
    // Initialize block number
    api::set_storage(
        StorageFlags::empty(),
        BLOCK_NUMBER_KEY,
        &1u64.to_le_bytes(),
    );
    
    encode(&[Token::Bool(true)])
}

fn handle_create_market(data: &[u8]) -> Vec<u8> {
    let tokens = match decode(&[
        ParamType::String,    // title
        ParamType::String,    // description
        ParamType::String,    // resolution_criteria
        ParamType::Uint(64),  // close_time
        ParamType::Uint(64),  // k_norm
        ParamType::Uint(64),  // initial_mean
        ParamType::Uint(64),  // initial_variance
    ], data) {
        Ok(t) => t,
        Err(_) => return encode_revert("Invalid parameters"),
    };
    
    let title = tokens[0].clone().into_string().unwrap();
    let description = tokens[1].clone().into_string().unwrap();
    let resolution_criteria = tokens[2].clone().into_string().unwrap();
    let close_time = tokens[3].clone().into_uint().unwrap().as_u64();
    let k_norm = tokens[4].clone().into_uint().unwrap().as_u64();
    let initial_mean = tokens[5].clone().into_uint().unwrap().as_u64();
    let initial_variance = tokens[6].clone().into_uint().unwrap().as_u64();
    
    // Validate variance
    if initial_variance < MIN_VARIANCE {
        return encode_revert("Variance too small");
    }
    
    // SAFETY: Prevent extreme parameters that break numerical calculations
    // Mean should be in reasonable range (0 to 1 million)
    if initial_mean > 1_000_000 * DECIMALS {
        return encode_revert("Initial mean too large (max 1M)");
    }
    
    // Variance should be reasonable (max 10 million)
    if initial_variance > 10_000_000 * DECIMALS {
        return encode_revert("Initial variance too large (max 10M)");
    }
    
    // Standard deviation should not be larger than mean
    let std_dev = sqrt_fixed(initial_variance);
    if initial_mean > 0 && std_dev > initial_mean * 2 {
        return encode_revert("Variance too large relative to mean");
    }
    
    // Get attached value
    let mut value_bytes = [0u8; 32];
    api::value_transferred(&mut value_bytes);
    let value_wei = u64::from_le_bytes([
        value_bytes[0], value_bytes[1], value_bytes[2], value_bytes[3],
        value_bytes[4], value_bytes[5], value_bytes[6], value_bytes[7]
    ]);
    
    if value_wei == 0 {
        return encode_revert("Must provide initial backing");
    }
    
    // Convert wei to fixed
    let b_backing = match wei_to_fixed(value_wei) {
        Ok(v) => v,
        Err(_) => return encode_revert("Value conversion failed"),
    };
    
    if b_backing == 0 {
        return encode_revert("Backing too small");
    }
    
    // Check minimum variance constraint
    let min_variance = match calculate_min_variance(k_norm, b_backing) {
        Ok(v) => v,
        Err(_) => return encode_revert("Min variance calculation failed"),
    };
    
    if initial_variance < min_variance {
        return encode_revert("Variance too low for backing constraint");
    }
    
    // Calculate lambda (verify it's valid)
    let _lambda = match calculate_lambda(k_norm, initial_variance) {
        Ok(v) => v,
        Err(_) => return encode_revert("Lambda calculation failed"),
    };
    
    // Check backing constraint (f_max <= b)
    let f_max = match calculate_f_max(k_norm, initial_variance) {
        Ok(v) => v,
        Err(_) => return encode_revert("F_max calculation failed"),
    };
    
    if f_max > b_backing {
        return encode_revert("Backing constraint violated");
    }
    
    // Get market count
    let mut market_count_bytes = [0u8; 8];
    let _ = api::get_storage(
        StorageFlags::empty(),
        MARKET_COUNT_KEY,
        &mut &mut market_count_bytes[..],
    );
    let market_id = u64::from_le_bytes(market_count_bytes);
    
    // Create market
    let market = Market {
        creator: {
            let mut addr = [0u8; 20];
            api::caller(&mut addr);
            addr
        },
        creation_time: get_timestamp(),
        close_time,
        k_norm,
        b_backing,
        current_mean: initial_mean,
        current_variance: initial_variance,
        // Initialize consensus tracking (market creator sets initial consensus)
        total_weight: 0, // No trades yet
        weighted_mean_sum: 0,
        weighted_m2_sum: 0,
        total_lp_shares: b_backing,
        total_backing: b_backing,
        accumulated_fees: 0,
        next_position_id: 0,
        total_volume: 0,
        status: MARKET_STATUS_OPEN,
        resolution_mean: 0,
        resolution_variance: 0,
    };
    
    // Save market
    save_market(market_id, &market);
    
    // Save metadata
    save_market_metadata(market_id, &title, &description, &resolution_criteria);
    
    // Give creator LP shares
    let lp_key = get_lp_balance_key(market_id, &market.creator);
    api::set_storage(
        StorageFlags::empty(),
        &lp_key,
        &b_backing.to_le_bytes(),
    );
    
    // Update market count
    let new_count = market_id + 1;
    api::set_storage(
        StorageFlags::empty(),
        MARKET_COUNT_KEY,
        &new_count.to_le_bytes(),
    );
    
    // Return market ID
    encode(&[Token::Uint(market_id.into())])
}

fn handle_calculate_trade(data: &[u8]) -> Vec<u8> {
    let tokens = match decode(&[
        ParamType::Uint(64), // market_id
        ParamType::Uint(64), // new_mean
        ParamType::Uint(64), // new_variance
        ParamType::Uint(64), // size
    ], data) {
        Ok(t) => t,
        Err(_) => return encode_revert("Invalid parameters"),
    };
    
    let market_id = tokens[0].clone().into_uint().unwrap().as_u64();
    let new_mean = tokens[1].clone().into_uint().unwrap().as_u64();
    let new_variance = tokens[2].clone().into_uint().unwrap().as_u64();
    let size = tokens[3].clone().into_uint().unwrap().as_u64();
    
    // Load market
    let market = match load_market(market_id) {
        Some(m) => m,
        None => return encode_revert("Market not found"),
    };
    
    // Check market status
    if market.status != MARKET_STATUS_OPEN {
        return encode_revert("Market not open");
    }
    
    // Validate new variance
    if new_variance < MIN_VARIANCE {
        return encode_revert("Variance too small");
    }
    
    // Check minimum variance constraint
    let min_variance = match calculate_min_variance(market.k_norm, market.b_backing) {
        Ok(v) => v,
        Err(_) => return encode_revert("Min variance calculation failed"),
    };
    
    if new_variance < min_variance {
        return encode_revert("Variance too low");
    }
    
    // IMPROVED: Use comprehensive validation for extreme trade protection
    let is_valid = match validate_backing_constraint(
        market.k_norm,
        market.b_backing,
        new_mean,
        new_variance,
        market.current_mean,
        market.current_variance
    ) {
        Ok(v) => v,
        Err(_) => return encode_revert("Validation failed"),
    };
    
    if !is_valid {
        return encode_revert("Trade violates market constraints");
    }
    
    // Calculate trade cost
    let (cost, fee, collateral) = match calculate_trade_cost(
        market.k_norm,
        market.current_mean, market.current_variance,
        new_mean, new_variance,
        size
    ) {
        Ok(v) => v,
        Err(_) => return encode_revert("Trade cost calculation failed"),
    };
    
    // Return costs
    encode(&[
        Token::Uint(cost.into()),
        Token::Uint(fee.into()),
        Token::Uint(collateral.into()),
    ])
}

fn handle_trade_distribution(data: &[u8]) -> Vec<u8> {
    let tokens = match decode(&[
        ParamType::Uint(64), // market_id
        ParamType::Uint(64), // new_mean
        ParamType::Uint(64), // new_variance
        ParamType::Uint(64), // size
        ParamType::Uint(64), // max_cost
    ], data) {
        Ok(t) => t,
        Err(_) => return encode_revert("Invalid parameters"),
    };
    
    let market_id = tokens[0].clone().into_uint().unwrap().as_u64();
    let new_mean = tokens[1].clone().into_uint().unwrap().as_u64();
    let new_variance = tokens[2].clone().into_uint().unwrap().as_u64();
    let size = tokens[3].clone().into_uint().unwrap().as_u64();
    let max_cost = tokens[4].clone().into_uint().unwrap().as_u64();
    
    // Load market
    let mut market = match load_market(market_id) {
        Some(m) => m,
        None => return encode_revert("Market not found"),
    };
    
    // Check market status
    if market.status != MARKET_STATUS_OPEN {
        return encode_revert("Market not open");
    }
    
    // Check if market has closed
    if get_timestamp() >= market.close_time {
        market.status = MARKET_STATUS_CLOSED;
        save_market(market_id, &market);
        return encode_revert("Market has closed");
    }
    
    // Validate new variance
    if new_variance < MIN_VARIANCE {
        return encode_revert("Variance too small");
    }
    
    // Check minimum variance constraint
    let min_variance = match calculate_min_variance(market.k_norm, market.b_backing) {
        Ok(v) => v,
        Err(_) => return encode_revert("Min variance calculation failed"),
    };
    
    if new_variance < min_variance {
        return encode_revert("Variance too low");
    }
    
    // Store entry state
    let entry_mean = market.current_mean;
    let entry_variance = market.current_variance;
    
    // IMPROVED: Use comprehensive validation for extreme trade protection
    let is_valid = match validate_backing_constraint(
        market.k_norm,
        market.b_backing,
        new_mean,
        new_variance,
        entry_mean,
        entry_variance
    ) {
        Ok(v) => v,
        Err(_) => return encode_revert("Validation failed"),
    };
    
    if !is_valid {
        return encode_revert("Trade violates market constraints");
    }
    
    // Calculate trade cost
    let (cost, fee, collateral_required) = match calculate_trade_cost(
        market.k_norm,
        entry_mean, entry_variance,
        new_mean, new_variance,
        size
    ) {
        Ok(v) => v,
        Err(_) => return encode_revert("Trade cost calculation failed"),
    };
    
    // Check max cost
    if cost > max_cost {
        return encode_revert("Cost exceeds maximum");
    }
    
    // Check payment
    let mut value_bytes = [0u8; 32];
    api::value_transferred(&mut value_bytes);
    let value_wei = u64::from_le_bytes([
        value_bytes[0], value_bytes[1], value_bytes[2], value_bytes[3],
        value_bytes[4], value_bytes[5], value_bytes[6], value_bytes[7]
    ]);
    
    let value_fixed = match wei_to_fixed(value_wei) {
        Ok(v) => v,
        Err(_) => return encode_revert("Value conversion failed"),
    };
    
    if value_fixed < cost {
        return encode_revert("Insufficient payment");
    }
    
    // Get trader address
    let mut caller = [0u8; 20];
    api::caller(&mut caller);
    
    // Get and increment global position counter
    let mut position_count_bytes = [0u8; 8];
    let _ = api::get_storage(
        StorageFlags::empty(),
        POSITION_COUNT_KEY,
        &mut &mut position_count_bytes[..],
    );
    let position_id = u64::from_le_bytes(position_count_bytes);
    let next_position_id = position_id + 1;
    api::set_storage(
        StorageFlags::empty(),
        POSITION_COUNT_KEY,
        &next_position_id.to_le_bytes(),
    );
    
    // Create position
    let position = Position {
        position_id,
        trader: caller,
        market_id,
        from_mean: entry_mean,
        from_variance: entry_variance,
        to_mean: new_mean,
        to_variance: new_variance,
        size,
        collateral_locked: collateral_required,
        cost_basis: cost,
        is_open: 1,
        opened_at: get_block_number(),
        closed_at: 0,
        exit_value: 0,
        fees_paid: fee,
        realized_pnl: 0i64,
        claimed: 0,
    };
    
    // Save position
    save_position(&position);
    add_trader_position(&caller, position.position_id);
    
    // Update market consensus using weighted aggregation
    // Add this position's contribution to the weighted sums
    let mean_squared_plus_var = match mul_fixed(new_mean, new_mean) {
        Ok(v) => v + new_variance,
        Err(_) => return encode_revert("Mean squared calculation overflow"),
    };
    
    market.total_weight = market.total_weight.saturating_add(size);
    market.weighted_mean_sum = market.weighted_mean_sum.saturating_add((size as u128) * (new_mean as u128));
    market.weighted_m2_sum = market.weighted_m2_sum.saturating_add((size as u128) * (mean_squared_plus_var as u128));
    
    // Calculate new consensus
    if market.total_weight > 0 {
        let (consensus_mean, consensus_variance) = match calculate_consensus(
            market.total_weight,
            market.weighted_mean_sum,
            market.weighted_m2_sum
        ) {
            Ok((m, v)) => (m, v),
            Err(_) => {
                // Fallback to current values if calculation fails
                (market.current_mean, market.current_variance)
            }
        };
        
        market.current_mean = consensus_mean;
        market.current_variance = consensus_variance;
    } else {
        // First trade sets the consensus
        market.current_mean = new_mean;
        market.current_variance = new_variance;
    }
    
    market.next_position_id += 1;  // Keep for backwards compatibility
    market.total_volume += cost;
    market.accumulated_fees += fee;
    
    save_market(market_id, &market);
    
    // Return excess payment if any
    if value_fixed > cost {
        let excess_fixed = value_fixed - cost;
        if let Ok(excess_wei) = fixed_to_wei(excess_fixed) {
            let rounded_excess = round_wei_for_transfer(excess_wei);
            if rounded_excess > 0 {
                let transfer_amount = u256_bytes(rounded_excess);
                let deposit_limit = [0xffu8; 32];
                let _ = api::call(
                    CallFlags::empty(),
                    &caller,
                    0,
                    0,
                    &deposit_limit,
                    &transfer_amount,
                    &[],
                    None,
                );
            }
        }
    }
    
    // Return position ID
    encode(&[Token::Uint(position.position_id.into())])
}

fn handle_close_position(data: &[u8]) -> Vec<u8> {
    // Debug point 1: Parameter decoding
    let tokens = match decode(&[ParamType::Uint(64)], data) {
        Ok(t) => t,
        Err(_) => return encode_revert("DEBUG1: Invalid parameters"),
    };
    
    let position_id = match tokens[0].clone().into_uint() {
        Some(uint) => uint.as_u64(),
        None => return encode_revert("DEBUG2: Invalid position ID"),
    };
    
    // Debug point 2: Load position
    let mut position = match load_position(position_id) {
        Some(p) => p,
        None => return encode_revert("DEBUG3: Position not found"),
    };
    
    // Check if already closed
    if position.is_open == 0 {
        return encode_revert("DEBUG4: Position already closed");
    }
    
    // Debug point 3: Check ownership
    let mut caller = [0u8; 20];
    api::caller(&mut caller);
    
    if caller != position.trader {
        return encode_revert("DEBUG5: Not position owner");
    }
    
    // Debug point 4: Load market
    let market = match load_market(position.market_id) {
        Some(m) => m,
        None => return encode_revert("DEBUG6: Market not found"),
    };
    
    // Debug point 5: Calculate position value
    let position_value = match calculate_position_value(
        &position,
        market.current_mean,
        market.current_variance,
        market.k_norm
    ) {
        Ok(v) => v,
        Err(e) => {
            // Return specific error for debugging
            let error_msg = if e == "Variance too small" {
                "DEBUG7: Variance too small"
            } else if e == "L2 norm is zero" {
                "DEBUG8: L2 norm is zero"
            } else if e == "Multiplication overflow" {
                "DEBUG9: Multiplication overflow"
            } else if e == "Division by zero" {
                "DEBUG10: Division by zero"
            } else if e == "Division overflow" {
                "DEBUG11: Division overflow"
            } else {
                "DEBUG12: Unknown calc error"
            };
            return encode_revert(error_msg);
        }
    };
    
    // Debug point 6: Update position
    position.is_open = 0;
    position.closed_at = get_block_number();
    position.exit_value = position_value;
    
    // Debug point 7: Calculate P&L
    // Since position_value now includes collateral, we need to adjust the P&L calculation
    if position_value >= position.collateral_locked {
        // Profit case: position_value = collateral + profit
        position.realized_pnl = (position_value - position.collateral_locked) as i64;
    } else {
        // Loss case: position_value = collateral - loss
        let loss = position.collateral_locked - position_value;
        if loss > i64::MAX as u64 {
            return encode_revert("DEBUG13: P&L calculation overflow");
        }
        position.realized_pnl = -(loss as i64);
    }
    
    // Debug point 8: Save position
    save_position(&position);
    
    // Update market consensus by removing this position's contribution
    let mut market = match load_market(position.market_id) {
        Some(m) => m,
        None => return encode_revert("Market not found for consensus update"),
    };
    
    // Remove position's contribution from weighted sums
    if market.total_weight >= position.size {
        market.total_weight = market.total_weight.saturating_sub(position.size);
        
        // Remove contribution to mean sum
        let mean_contribution = (position.size as u128) * (position.to_mean as u128);
        market.weighted_mean_sum = market.weighted_mean_sum.saturating_sub(mean_contribution);
        
        // Remove contribution to m2 sum
        let mean_squared_plus_var = match mul_fixed(position.to_mean, position.to_mean) {
            Ok(v) => v.saturating_add(position.to_variance),
            Err(_) => position.to_variance, // Fallback
        };
        let m2_contribution = (position.size as u128) * (mean_squared_plus_var as u128);
        market.weighted_m2_sum = market.weighted_m2_sum.saturating_sub(m2_contribution);
        
        // Recalculate consensus
        if market.total_weight > 0 {
            let (consensus_mean, consensus_variance) = match calculate_consensus(
                market.total_weight,
                market.weighted_mean_sum,
                market.weighted_m2_sum
            ) {
                Ok((m, v)) => (m, v),
                Err(_) => {
                    // Keep current values if calculation fails
                    (market.current_mean, market.current_variance)
                }
            };
            
            market.current_mean = consensus_mean;
            market.current_variance = consensus_variance;
        }
        
        save_market(position.market_id, &market);
    }
    
    // Debug point 9: Transfer handling
    if position_value > 0 {
        match fixed_to_wei(position_value) {
            Ok(position_value_wei) => {
                let rounded_value = round_wei_for_transfer(position_value_wei);
                if rounded_value >= MIN_TRANSFER_UNIT {
                    let transfer_amount = u256_bytes(rounded_value);
                    let deposit_limit = [0xffu8; 32];
                    // Note: api::call might be causing the issue
                    let result = api::call(
                        CallFlags::empty(),
                        &position.trader,
                        0,
                        0,
                        &deposit_limit,
                        &transfer_amount,
                        &[],
                        None,
                    );
                    
                    if result.is_err() {
                        // Don't fail, just note it
                    }
                }
            },
            Err(_) => {
                // Value conversion failed
            }
        }
    }
    
    // Debug point 10: Final encoding
    // Convert position_value (u64) to U256
    let value_u256 = ethabi::ethereum_types::U256::from(position_value);
    let value_token = Token::Uint(value_u256);
    
    // Convert realized_pnl (i64) to U256 for Token::Int
    // Need to handle negative values properly
    let pnl_u256 = if position.realized_pnl >= 0 {
        ethabi::ethereum_types::U256::from(position.realized_pnl as u64)
    } else {
        // For negative values, we need to use two's complement
        let abs_value = position.realized_pnl.unsigned_abs();
        ethabi::ethereum_types::U256::MAX - ethabi::ethereum_types::U256::from(abs_value) + 1
    };
    let pnl_token = Token::Int(pnl_u256);
    
    encode(&[value_token, pnl_token])
}

fn handle_add_liquidity(data: &[u8]) -> Vec<u8> {
    let tokens = match decode(&[ParamType::Uint(64)], data) {
        Ok(t) => t,
        Err(_) => return encode_revert("Invalid parameters"),
    };
    
    let market_id = tokens[0].clone().into_uint().unwrap().as_u64();
    
    // Load market
    let mut market = match load_market(market_id) {
        Some(m) => m,
        None => return encode_revert("Market not found"),
    };
    
    // Check market status
    if market.status != MARKET_STATUS_OPEN {
        return encode_revert("Market not open");
    }
    
    // Get value
    let mut value_bytes = [0u8; 32];
    api::value_transferred(&mut value_bytes);
    let value_wei = u64::from_le_bytes([
        value_bytes[0], value_bytes[1], value_bytes[2], value_bytes[3],
        value_bytes[4], value_bytes[5], value_bytes[6], value_bytes[7]
    ]);
    
    let value_fixed = match wei_to_fixed(value_wei) {
        Ok(v) => v,
        Err(_) => return encode_revert("Value conversion failed"),
    };
    
    if value_fixed == 0 {
        return encode_revert("Must provide liquidity");
    }
    
    // Calculate LP shares to mint
    let lp_shares_to_mint = if market.total_backing == 0 {
        value_fixed
    } else {
        match div_fixed(value_fixed, market.total_backing) {
            Ok(ratio) => match mul_fixed(ratio, market.total_lp_shares) {
                Ok(shares) => shares,
                Err(_) => return encode_revert("LP share calculation overflow"),
            },
            Err(_) => return encode_revert("LP ratio calculation failed"),
        }
    };
    
    // Update market
    market.total_backing += value_fixed;
    market.total_lp_shares += lp_shares_to_mint;
    market.b_backing = market.total_backing;
    
    // IMPORTANT: Scale k_norm proportionally with liquidity
    // This ensures that adding liquidity reduces trading costs
    if market.total_backing > value_fixed {
        // Scale k_norm by the same ratio as backing increased
        let old_backing = market.total_backing - value_fixed;
        let scaling_factor = match div_fixed(market.total_backing, old_backing) {
            Ok(f) => f,
            Err(_) => return encode_revert("Scaling calculation failed"),
        };
        
        market.k_norm = match mul_fixed(market.k_norm, scaling_factor) {
            Ok(k) => k,
            Err(_) => return encode_revert("k_norm scaling overflow"),
        };
    }
    
    // Check if new backing violates variance constraint
    let new_min_variance = match calculate_min_variance(market.k_norm, market.b_backing) {
        Ok(v) => v,
        Err(_) => return encode_revert("Min variance calculation failed"),
    };
    
    if market.current_variance < new_min_variance {
        return encode_revert("Market variance would violate constraint");
    }
    
    save_market(market_id, &market);
    
    // Update LP balance
    let mut caller = [0u8; 20];
    api::caller(&mut caller);
    
    let lp_key = get_lp_balance_key(market_id, &caller);
    let mut current_balance_bytes = [0u8; 8];
    let _ = api::get_storage(
        StorageFlags::empty(),
        &lp_key,
        &mut &mut current_balance_bytes[..],
    );
    
    let current_balance = u64::from_le_bytes(current_balance_bytes);
    let new_balance = current_balance + lp_shares_to_mint;
    
    api::set_storage(
        StorageFlags::empty(),
        &lp_key,
        &new_balance.to_le_bytes(),
    );
    
    // Return minted shares
    encode(&[Token::Uint(lp_shares_to_mint.into())])
}

fn handle_remove_liquidity(data: &[u8]) -> Vec<u8> {
    let tokens = match decode(&[
        ParamType::Uint(64), // market_id
        ParamType::Uint(64), // shares_to_burn
    ], data) {
        Ok(t) => t,
        Err(_) => return encode_revert("Invalid parameters"),
    };
    
    let market_id = tokens[0].clone().into_uint().unwrap().as_u64();
    let shares_to_burn = tokens[1].clone().into_uint().unwrap().as_u64();
    
    // Load market
    let mut market = match load_market(market_id) {
        Some(m) => m,
        None => return encode_revert("Market not found"),
    };
    
    // Get caller
    let mut caller = [0u8; 20];
    api::caller(&mut caller);
    
    // Check LP balance
    let lp_key = get_lp_balance_key(market_id, &caller);
    let mut balance_bytes = [0u8; 8];
    let _ = api::get_storage(
        StorageFlags::empty(),
        &lp_key,
        &mut &mut balance_bytes[..],
    );
    
    let balance = u64::from_le_bytes(balance_bytes);
    
    if balance == 0 {
        return encode_revert("No LP balance");
    }
    
    if shares_to_burn == 0 {
        return encode_revert("Cannot burn 0 shares");
    }
    
    if balance < shares_to_burn {
        return encode_revert("Insufficient LP shares");
    }
    
    // Calculate backing to return (includes fee share)
    let total_assets = market.total_backing + market.accumulated_fees;
    
    let backing_to_return = match div_fixed(shares_to_burn, market.total_lp_shares) {
        Ok(ratio) => match mul_fixed(ratio, total_assets) {
            Ok(amount) => amount,
            Err(_) => return encode_revert("Backing calculation overflow"),
        },
        Err(_) => return encode_revert("Ratio calculation failed"),
    };
    
    // Calculate backing and fee portions
    let backing_portion = match div_fixed(shares_to_burn, market.total_lp_shares) {
        Ok(ratio) => match mul_fixed(ratio, market.total_backing) {
            Ok(amount) => amount,
            Err(_) => return encode_revert("Backing portion overflow"),
        },
        Err(_) => return encode_revert("Ratio calculation failed"),
    };
    
    let fee_portion = backing_to_return.saturating_sub(backing_portion);
    
    // Check if removal would violate minimum liquidity
    let remaining_backing = market.total_backing.saturating_sub(backing_portion);
    let remaining_shares = market.total_lp_shares.saturating_sub(shares_to_burn);
    
    if remaining_shares > 0 && remaining_backing < market.k_norm {
        return encode_revert("Would violate minimum liquidity");
    }
    
    // Update market
    market.total_backing = remaining_backing;
    market.total_lp_shares = remaining_shares;
    market.b_backing = market.total_backing;
    market.accumulated_fees = market.accumulated_fees.saturating_sub(fee_portion);
    
    // IMPORTANT: Scale k_norm proportionally when removing liquidity
    if market.total_backing > 0 && backing_portion > 0 {
        // Scale k_norm by the same ratio as backing decreased
        let old_backing = market.total_backing + backing_portion;
        let scaling_factor = match div_fixed(market.total_backing, old_backing) {
            Ok(f) => f,
            Err(_) => return encode_revert("Scaling calculation failed"),
        };
        
        market.k_norm = match mul_fixed(market.k_norm, scaling_factor) {
            Ok(k) => k,
            Err(_) => return encode_revert("k_norm scaling failed"),
        };
    }
    
    // Check variance constraint after update
    if market.total_backing > 0 {
        let new_min_variance = match calculate_min_variance(market.k_norm, market.b_backing) {
            Ok(v) => v,
            Err(_) => return encode_revert("Min variance calculation failed"),
        };
        
        if market.current_variance < new_min_variance {
            return encode_revert("Would violate variance constraint");
        }
    }
    
    save_market(market_id, &market);
    
    // Update LP balance
    let new_balance = balance - shares_to_burn;
    api::set_storage(
        StorageFlags::empty(),
        &lp_key,
        &new_balance.to_le_bytes(),
    );
    
    // Transfer backing to LP
    if backing_to_return > 0 {
        if let Ok(backing_wei) = fixed_to_wei(backing_to_return) {
            let rounded_value = round_wei_for_transfer(backing_wei);
            if rounded_value > 0 {
                let transfer_amount = u256_bytes(rounded_value);
                let deposit_limit = [0xffu8; 32];
                let result = api::call(
                    CallFlags::empty(),
                    &caller,
                    0,
                    0,
                    &deposit_limit,
                    &transfer_amount,
                    &[],
                    None,
                );
                
                if result.is_err() {
                    return encode_revert("Transfer failed");
                }
            }
        }
    }
    
    // Return amount
    encode(&[Token::Uint(backing_to_return.into())])
}

// View functions

fn handle_get_market_state(data: &[u8]) -> Vec<u8> {
    let tokens = match decode(&[ParamType::Uint(64)], data) {
        Ok(t) => t,
        Err(_) => return encode_revert("Invalid parameters"),
    };
    
    let market_id = tokens[0].clone().into_uint().unwrap().as_u64();
    
    let market = match load_market(market_id) {
        Some(m) => m,
        None => return encode_revert("Market not found"),
    };
    
    // Calculate derived values
    let lambda = calculate_lambda(market.k_norm, market.current_variance).unwrap_or(0);
    let f_max = calculate_f_max(market.k_norm, market.current_variance).unwrap_or(0);
    
    encode(&[
        Token::Uint(market.current_mean.into()),
        Token::Uint(market.current_variance.into()),
        Token::Uint(market.k_norm.into()),
        Token::Uint(market.b_backing.into()),
        Token::Uint(market.total_lp_shares.into()),
        Token::Uint(f_max.into()),
        Token::Uint(market.status.into()),
        Token::Uint(market.accumulated_fees.into()),
        Token::Uint(lambda.into()),
    ])
}

fn handle_evaluate_at(data: &[u8]) -> Vec<u8> {
    let tokens = match decode(&[
        ParamType::Uint(64), // market_id
        ParamType::Uint(64), // x
    ], data) {
        Ok(t) => t,
        Err(_) => return encode_revert("Invalid parameters"),
    };
    
    let market_id = tokens[0].clone().into_uint().unwrap().as_u64();
    let x = tokens[1].clone().into_uint().unwrap().as_u64();
    
    let market = match load_market(market_id) {
        Some(m) => m,
        None => return encode_revert("Market not found"),
    };
    
    // Calculate PDF and f(x)
    let pdf_value = normal_pdf(x, market.current_mean, market.current_variance);
    let lambda = calculate_lambda(market.k_norm, market.current_variance).unwrap_or(0);
    let f_value = mul_fixed(lambda, pdf_value).unwrap_or(0);
    
    // Cap at backing
    let capped_f_value = if f_value > market.b_backing {
        market.b_backing
    } else {
        f_value
    };
    
    encode(&[
        Token::Uint(pdf_value.into()),
        Token::Uint(capped_f_value.into()),
    ])
}

fn handle_get_cdf(data: &[u8]) -> Vec<u8> {
    let tokens = match decode(&[
        ParamType::Uint(64), // market_id
        ParamType::Uint(64), // x
    ], data) {
        Ok(t) => t,
        Err(_) => return encode_revert("Invalid parameters"),
    };
    
    let market_id = tokens[0].clone().into_uint().unwrap().as_u64();
    let x = tokens[1].clone().into_uint().unwrap().as_u64();
    
    let market = match load_market(market_id) {
        Some(m) => m,
        None => return encode_revert("Market not found"),
    };
    
    let cdf_value = normal_cdf(x, market.current_mean, market.current_variance);
    
    encode(&[Token::Uint(cdf_value.into())])
}

fn handle_get_expected_value(data: &[u8]) -> Vec<u8> {
    let tokens = match decode(&[ParamType::Uint(64)], data) {
        Ok(t) => t,
        Err(_) => return encode_revert("Invalid parameters"),
    };
    
    let market_id = tokens[0].clone().into_uint().unwrap().as_u64();
    
    let market = match load_market(market_id) {
        Some(m) => m,
        None => return encode_revert("Market not found"),
    };
    
    let expected_value = calculate_expected_value(market.current_mean, market.current_variance);
    
    encode(&[Token::Uint(expected_value.into())])
}

fn handle_get_bounds(data: &[u8]) -> Vec<u8> {
    let tokens = match decode(&[ParamType::Uint(64)], data) {
        Ok(t) => t,
        Err(_) => return encode_revert("Invalid parameters"),
    };
    
    let market_id = tokens[0].clone().into_uint().unwrap().as_u64();
    
    let market = match load_market(market_id) {
        Some(m) => m,
        None => return encode_revert("Market not found"),
    };
    
    let (lower, upper) = get_distribution_bounds(market.current_mean, market.current_variance);
    
    encode(&[
        Token::Uint(lower.into()),
        Token::Uint(upper.into()),
    ])
}

fn handle_get_market_info(data: &[u8]) -> Vec<u8> {
    let tokens = match decode(&[ParamType::Uint(64)], data) {
        Ok(t) => t,
        Err(_) => return encode_revert("Invalid parameters"),
    };
    
    let market_id = tokens[0].clone().into_uint().unwrap().as_u64();
    
    let market = match load_market(market_id) {
        Some(m) => m,
        None => return encode_revert("Market not found"),
    };
    
    // Calculate all derived values
    let lambda = calculate_lambda(market.k_norm, market.current_variance).unwrap_or(0);
    let f_max = calculate_f_max(market.k_norm, market.current_variance).unwrap_or(0);
    let min_variance = calculate_min_variance(market.k_norm, market.b_backing).unwrap_or(0);
    let expected_value = calculate_expected_value(market.current_mean, market.current_variance);
    let (lower_bound, upper_bound) = get_distribution_bounds(market.current_mean, market.current_variance);
    
    encode(&[
        Token::Address(market.creator.into()),
        Token::Uint(market.creation_time.into()),
        Token::Uint(market.close_time.into()),
        Token::Uint(market.k_norm.into()),
        Token::Uint(market.b_backing.into()),
        Token::Uint(market.current_mean.into()),
        Token::Uint(market.current_variance.into()),
        Token::Uint(lambda.into()),
        Token::Uint(market.total_lp_shares.into()),
        Token::Uint(market.total_backing.into()),
        Token::Uint(market.accumulated_fees.into()),
        Token::Uint(f_max.into()),
        Token::Uint(min_variance.into()),
        Token::Uint(market.total_volume.into()),
        Token::Uint(market.status.into()),
        Token::Uint(expected_value.into()),
        Token::Uint(lower_bound.into()),
        Token::Uint(upper_bound.into()),
    ])
}

fn handle_get_position_value(data: &[u8]) -> Vec<u8> {
    let tokens = match decode(&[ParamType::Uint(64)], data) {
        Ok(t) => t,
        Err(_) => return encode_revert("Invalid parameters"),
    };
    
    let position_id = tokens[0].clone().into_uint().unwrap().as_u64();
    
    let position = match load_position(position_id) {
        Some(p) => p,
        None => return encode_revert("Position not found"),
    };
    
    // If closed, return exit value
    if position.is_open == 0 {
        return encode(&[Token::Uint(position.exit_value.into())]);
    }
    
    // Calculate current value
    let market = match load_market(position.market_id) {
        Some(m) => m,
        None => return encode_revert("Market not found"),
    };
    
    let current_value = match calculate_position_value(
        &position,
        market.current_mean,
        market.current_variance,
        market.k_norm
    ) {
        Ok(v) => v,
        Err(_) => 0,
    };
    
    encode(&[Token::Uint(current_value.into())])
}

fn handle_get_tvl(data: &[u8]) -> Vec<u8> {
    let tokens = match decode(&[ParamType::Uint(64)], data) {
        Ok(t) => t,
        Err(_) => return encode_revert("Invalid parameters"),
    };
    
    let market_id = tokens[0].clone().into_uint().unwrap().as_u64();
    
    let market = match load_market(market_id) {
        Some(m) => m,
        None => return encode_revert("Market not found"),
    };
    
    let tvl = market.total_backing + market.accumulated_fees;
    
    encode(&[Token::Uint(tvl.into())])
}

fn handle_resolve_market(data: &[u8]) -> Vec<u8> {
    let tokens = match decode(&[
        ParamType::Uint(64), // market_id
        ParamType::Uint(64), // final_mean
        ParamType::Uint(64), // final_variance
    ], data) {
        Ok(t) => t,
        Err(_) => return encode_revert("Invalid parameters"),
    };
    
    let market_id = tokens[0].clone().into_uint().unwrap().as_u64();
    let final_mean = tokens[1].clone().into_uint().unwrap().as_u64();
    let final_variance = tokens[2].clone().into_uint().unwrap().as_u64();
    
    let mut market = match load_market(market_id) {
        Some(m) => m,
        None => return encode_revert("Market not found"),
    };
    
    // Check authorization (only owner can resolve)
    let mut owner = [0u8; 20];
    let _ = api::get_storage(
        StorageFlags::empty(),
        OWNER_KEY,
        &mut &mut owner[..],
    );
    
    let mut caller = [0u8; 20];
    api::caller(&mut caller);
    
    if caller != owner {
        return encode_revert("Not authorized");
    }
    
    // Check if already resolved
    if market.status == MARKET_STATUS_RESOLVED {
        return encode_revert("Market already resolved");
    }
    
    // Check if market is closed or past close time
    if market.status == MARKET_STATUS_OPEN && get_timestamp() < market.close_time {
        return encode_revert("Market still open");
    }
    
    // Validate resolution variance
    if final_variance < MIN_VARIANCE {
        return encode_revert("Resolution variance too small");
    }
    
    // Check backing constraint for resolution
    let min_variance = match calculate_min_variance(market.k_norm, market.b_backing) {
        Ok(v) => v,
        Err(_) => return encode_revert("Min variance calculation failed"),
    };
    
    if final_variance < min_variance {
        return encode_revert("Resolution variance too low");
    }
    
    // Update market
    market.status = MARKET_STATUS_RESOLVED;
    market.resolution_mean = final_mean;
    market.resolution_variance = final_variance;
    
    save_market(market_id, &market);
    
    encode(&[Token::Bool(true)])
}

fn handle_claim_winnings(data: &[u8]) -> Vec<u8> {
    let tokens = match decode(&[ParamType::Uint(64)], data) {
        Ok(t) => t,
        Err(_) => return encode_revert("Invalid parameters"),
    };
    
    let position_id = tokens[0].clone().into_uint().unwrap().as_u64();
    
    let mut position = match load_position(position_id) {
        Some(p) => p,
        None => return encode_revert("Position not found"),
    };
    
    // Check ownership
    let mut caller = [0u8; 20];
    api::caller(&mut caller);
    
    if caller != position.trader {
        return encode_revert("Not position owner");
    }
    
    // Check if already claimed
    if position.claimed == 1 {
        return encode_revert("Already claimed");
    }
    
    // Load market
    let market = match load_market(position.market_id) {
        Some(m) => m,
        None => return encode_revert("Market not found"),
    };
    
    // Check if market is resolved
    if market.status != MARKET_STATUS_RESOLVED {
        return encode_revert("Market not resolved");
    }
    
    // Calculate final value
    let final_value = if position.is_open == 1 {
        // Position still open, calculate value at resolution
        match calculate_position_value(
            &position,
            market.resolution_mean,
            market.resolution_variance,
            market.k_norm
        ) {
            Ok(v) => v,
            Err(_) => 0,
        }
    } else {
        // Position was closed, use exit value
        position.exit_value
    };
    
    // Mark as claimed
    position.claimed = 1;
    save_position(&position);
    
    // Transfer winnings
    if final_value > 0 {
        if let Ok(final_value_wei) = fixed_to_wei(final_value) {
            let rounded_value = round_wei_for_transfer(final_value_wei);
            if rounded_value > 0 {
                let transfer_amount = u256_bytes(rounded_value);
                let deposit_limit = [0xffu8; 32];
                let result = api::call(
                    CallFlags::empty(),
                    &caller,
                    0,
                    0,
                    &deposit_limit,
                    &transfer_amount,
                    &[],
                    None,
                );
                
                if result.is_err() {
                    return encode_revert("Transfer failed");
                }
            }
        }
    }
    
    encode(&[Token::Uint(final_value.into())])
}

fn handle_get_consensus(data: &[u8]) -> Vec<u8> {
    let tokens = match decode(&[
        ParamType::Uint(64), // market_id
        ParamType::Uint(64), // x
    ], data) {
        Ok(t) => t,
        Err(_) => return encode_revert("Invalid parameters"),
    };
    
    let market_id = tokens[0].clone().into_uint().unwrap().as_u64();
    let x = tokens[1].clone().into_uint().unwrap().as_u64();
    
    let market = match load_market(market_id) {
        Some(m) => m,
        None => return encode_revert("Market not found"),
    };
    
    // Calculate consensus distribution at x
    let pdf_value = normal_pdf(x, market.current_mean, market.current_variance);
    let lambda = calculate_lambda(market.k_norm, market.current_variance).unwrap_or(0);
    let f_value = mul_fixed(lambda, pdf_value).unwrap_or(0);
    
    // Cap at backing
    let capped_f_value = if f_value > market.b_backing {
        market.b_backing
    } else {
        f_value
    };
    
    // Calculate AMM holdings
    let holdings = calculate_amm_holdings(x, &market);
    
    encode(&[
        Token::Uint(capped_f_value.into()),
        Token::Uint(holdings.into()),
    ])
}

fn handle_get_metadata(data: &[u8]) -> Vec<u8> {
    let tokens = match decode(&[ParamType::Uint(64)], data) {
        Ok(t) => t,
        Err(_) => return encode_revert("Invalid parameters"),
    };
    
    let market_id = tokens[0].clone().into_uint().unwrap().as_u64();
    
    let key = get_metadata_key(market_id);
    let mut buffer = [0u8; MAX_STORAGE_VALUE];
    
    let _ = api::get_storage(
        StorageFlags::empty(),
        &key,
        &mut &mut buffer[..],
    );
    
    // Parse metadata
    let (title, description, resolution_criteria) = if buffer[0] == 0 && buffer[1] == 0 && buffer[2] == 0 {
        ("", "", "")
    } else {
        let title_len = buffer[0] as usize;
        let desc_len = buffer[1] as usize;
        let criteria_len = buffer[2] as usize;
        
        let mut offset = 3;
        
        let title = core::str::from_utf8(&buffer[offset..offset + title_len]).unwrap_or("");
        offset += title_len;
        
        let description = core::str::from_utf8(&buffer[offset..offset + desc_len]).unwrap_or("");
        offset += desc_len;
        
        let resolution_criteria = core::str::from_utf8(&buffer[offset..offset + criteria_len]).unwrap_or("");
        
        (title, description, resolution_criteria)
    };
    
    encode(&[
        Token::String(title.into()),
        Token::String(description.into()),
        Token::String(resolution_criteria.into()),
    ])
}

fn handle_get_market_count() -> Vec<u8> {
    let mut market_count_bytes = [0u8; 8];
    let _ = api::get_storage(
        StorageFlags::empty(),
        MARKET_COUNT_KEY,
        &mut &mut market_count_bytes[..],
    );
    
    let count = u64::from_le_bytes(market_count_bytes);
    
    encode(&[Token::Uint(count.into())])
}

fn handle_get_trader_positions(data: &[u8]) -> Vec<u8> {
    let tokens = match decode(&[ParamType::Address], data) {
        Ok(t) => t,
        Err(_) => return encode_revert("Invalid parameters"),
    };
    
    let trader_bytes = tokens[0].clone().into_address().unwrap();
    let mut trader = [0u8; 20];
    trader.copy_from_slice(&trader_bytes.0);
    
    // Get position count
    let count_key = get_trader_position_count_key(&trader);
    let mut count_bytes = [0u8; 8];
    let _ = api::get_storage(
        StorageFlags::empty(),
        &count_key,
        &mut &mut count_bytes[..],
    );
    
    let count = u64::from_le_bytes(count_bytes);
    
    // Collect position IDs
    let mut position_ids = Vec::new();
    
    for i in 0..count {
        let pos_key = get_trader_positions_key(&trader, i);
        let mut pos_id_bytes = [0u8; 8];
        let _ = api::get_storage(
            StorageFlags::empty(),
            &pos_key,
            &mut &mut pos_id_bytes[..],
        );
        
        let pos_id = u64::from_le_bytes(pos_id_bytes);
        position_ids.push(Token::Uint(pos_id.into()));
    }
    
    encode(&[Token::Array(position_ids)])
}

fn handle_get_position(data: &[u8]) -> Vec<u8> {
    let tokens = match decode(&[ParamType::Uint(64)], data) {
        Ok(t) => t,
        Err(_) => return encode_revert("Invalid parameters"),
    };
    
    let position_id = tokens[0].clone().into_uint().unwrap().as_u64();
    
    match load_position(position_id) {
        Some(position) => {
            encode(&[
                Token::Uint(position.position_id.into()),
                Token::Address(position.trader.into()),
                Token::Uint(position.market_id.into()),
                Token::Uint(position.from_mean.into()),
                Token::Uint(position.from_variance.into()),
                Token::Uint(position.to_mean.into()),
                Token::Uint(position.to_variance.into()),
                Token::Uint(position.size.into()),
                Token::Uint(position.collateral_locked.into()),
                Token::Uint(position.cost_basis.into()),
                Token::Uint(position.opened_at.into()),
                Token::Uint(position.is_open.into()),
                Token::Uint(position.closed_at.into()),
                Token::Uint(position.exit_value.into()),
                Token::Uint(position.fees_paid.into()),
                Token::Int(if position.realized_pnl >= 0 {
                    ethabi::ethereum_types::U256::from(position.realized_pnl as u64)
                } else {
                    // For negative values, use two's complement
                    let abs_value = position.realized_pnl.unsigned_abs();
                    ethabi::ethereum_types::U256::MAX - ethabi::ethereum_types::U256::from(abs_value) + 1
                }),
                Token::Uint(position.claimed.into()),
            ])
        }
        None => encode_revert("Position not found"),
    }
}

fn handle_get_lp_balance(data: &[u8]) -> Vec<u8> {
    let tokens = match decode(&[
        ParamType::Uint(64),  // market_id
        ParamType::Address,   // address
    ], data) {
        Ok(t) => t,
        Err(_) => return encode_revert("Invalid parameters"),
    };
    
    let market_id = tokens[0].clone().into_uint().unwrap().as_u64();
    let address_token = tokens[1].clone().into_address().unwrap();
    let mut address = [0u8; 20];
    address.copy_from_slice(address_token.as_bytes());
    
    let lp_key = get_lp_balance_key(market_id, &address);
    let mut balance_bytes = [0u8; 8];
    
    let result = api::get_storage(
        StorageFlags::empty(),
        &lp_key,
        &mut &mut balance_bytes[..],
    );
    
    let balance = if result.is_ok() {
        u64::from_le_bytes(balance_bytes)
    } else {
        0u64
    };
    
    encode(&[Token::Uint(balance.into())])
}

fn handle_get_amm_holdings(data: &[u8]) -> Vec<u8> {
    let tokens = match decode(&[
        ParamType::Uint(64), // market_id
        ParamType::Uint(64), // x
    ], data) {
        Ok(t) => t,
        Err(_) => return encode_revert("Invalid parameters"),
    };
    
    let market_id = tokens[0].clone().into_uint().unwrap().as_u64();
    let x = tokens[1].clone().into_uint().unwrap().as_u64();
    
    let market = match load_market(market_id) {
        Some(m) => m,
        None => return encode_revert("Market not found"),
    };
    
    let holdings = calculate_amm_holdings(x, &market);
    
    encode(&[Token::Uint(holdings.into())])
}

// Main entry points

#[polkavm_export]
pub extern "C" fn deploy() {
    // Initialize market count
    api::set_storage(
        StorageFlags::empty(),
        MARKET_COUNT_KEY,
        &0u64.to_le_bytes(),
    );
    
    // Initialize position count
    api::set_storage(
        StorageFlags::empty(),
        POSITION_COUNT_KEY,
        &0u64.to_le_bytes(),
    );
}

#[polkavm_export]
pub extern "C" fn call() {
    // Get call data size
    let length = api::call_data_size() as usize;
    
    if length == 0 {
        api::return_value(ReturnFlags::empty(), &[]);
        return;
    }
    
    if length < 4 {
        api::return_value(ReturnFlags::REVERT, b"Invalid input");
        return;
    }
    
    // Read selector
    let mut selector = [0u8; 4];
    api::call_data_copy(&mut selector, 0);
    
    // Read data
    let mut data = [0u8; MAX_INPUT];
    let data_len = length.saturating_sub(4).min(MAX_INPUT);
    
    if data_len > 0 {
        api::call_data_copy(&mut data[..data_len], 4);
    }
    
    // Route to handler
    let result = match selector {
        INITIALIZE_SELECTOR => handle_initialize(),
        CREATE_MARKET_SELECTOR => handle_create_market(&data[..data_len]),
        TRADE_DISTRIBUTION_SELECTOR => handle_trade_distribution(&data[..data_len]),
        ADD_LIQUIDITY_SELECTOR => handle_add_liquidity(&data[..data_len]),
        REMOVE_LIQUIDITY_SELECTOR => handle_remove_liquidity(&data[..data_len]),
        GET_MARKET_STATE_SELECTOR => handle_get_market_state(&data[..data_len]),
        GET_CONSENSUS_SELECTOR => handle_get_consensus(&data[..data_len]),
        GET_METADATA_SELECTOR => handle_get_metadata(&data[..data_len]),
        GET_MARKET_COUNT_SELECTOR => handle_get_market_count(),
        GET_TRADER_POSITIONS_SELECTOR => handle_get_trader_positions(&data[..data_len]),
        CLOSE_POSITION_SELECTOR => handle_close_position(&data[..data_len]),
        GET_POSITION_SELECTOR => handle_get_position(&data[..data_len]),
        RESOLVE_MARKET_SELECTOR => handle_resolve_market(&data[..data_len]),
        CLAIM_WINNINGS_SELECTOR => handle_claim_winnings(&data[..data_len]),
        CALCULATE_TRADE_SELECTOR => handle_calculate_trade(&data[..data_len]),
        GET_LP_BALANCE_SELECTOR => handle_get_lp_balance(&data[..data_len]),
        GET_AMM_HOLDINGS_SELECTOR => handle_get_amm_holdings(&data[..data_len]),
        EVALUATE_AT_SELECTOR => handle_evaluate_at(&data[..data_len]),
        GET_CDF_SELECTOR => handle_get_cdf(&data[..data_len]),
        GET_EXPECTED_VALUE_SELECTOR => handle_get_expected_value(&data[..data_len]),
        GET_BOUNDS_SELECTOR => handle_get_bounds(&data[..data_len]),
        GET_MARKET_INFO_SELECTOR => handle_get_market_info(&data[..data_len]),
        GET_POSITION_VALUE_SELECTOR => handle_get_position_value(&data[..data_len]),
        GET_TVL_SELECTOR => handle_get_tvl(&data[..data_len]),
        _ => Vec::new(), // Unknown selector
    };
    
    // Return result
    if result.is_empty() {
        api::return_value(ReturnFlags::empty(), &[]);
    } else if result.starts_with(&[0x08, 0xc3, 0x79, 0xa0]) {
        // Error response
        api::return_value(ReturnFlags::REVERT, &result);
    } else {
        // Success response
        api::return_value(ReturnFlags::empty(), &result);
    }
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // Instead of panic, revert with generic error
    let error = encode_revert("Contract panic");
    api::return_value(ReturnFlags::REVERT, &error);
    
    // Required for panic handler
    unsafe {
        core::arch::asm!("unimp");
        core::hint::unreachable_unchecked();
    }
}