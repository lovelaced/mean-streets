#![no_main]
#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use ethabi::{encode, decode, Token, ParamType};
use polkavm_derive::polkavm_export;
use simplealloc::SimpleAlloc;
use uapi::{HostFn, HostFnImpl as api, StorageFlags, ReturnFlags, CallFlags, u64_output};

#[global_allocator]
static ALLOCATOR: SimpleAlloc<100000> = SimpleAlloc::new();

// Constants for fixed-point arithmetic (9 decimal places)
const DECIMALS: u64 = 1_000_000_000;
const MAX_INPUT: usize = 1000;
const MAX_STORAGE_VALUE: usize = 512;

// Mathematical constants
const PI_FIXED: u64 = 3_141_592_654; // π with 9 decimals
const SQRT_PI_FIXED: u64 = 1_772_453_851; // √π with 9 decimals
const SQRT_2PI_FIXED: u64 = 2_506_628_275; // √(2π) with 9 decimals

// Storage keys
const OWNER_KEY: &[u8] = b"owner";
const INITIALIZED_KEY: &[u8] = b"initialized";
const MARKET_COUNT_KEY: &[u8] = b"market_count";
const LP_TOKEN_PREFIX: &[u8] = b"lp_";
const POSITION_PREFIX: &[u8] = b"pos_";
const METADATA_PREFIX: &[u8] = b"meta_";
const POSITION_BY_ID_PREFIX: &[u8] = b"pos_id_";
const TRADER_POSITIONS_PREFIX: &[u8] = b"trader_pos_";
const TRADER_POS_COUNT_PREFIX: &[u8] = b"trader_cnt_";

// Function selectors
const INITIALIZE_SELECTOR: [u8; 4] = [0x81, 0x29, 0xfc, 0x1c]; // initialize()
const CREATE_MARKET_SELECTOR: [u8; 4] = [0x44, 0xb8, 0x5a, 0x62]; // createMarket
const TRADE_DISTRIBUTION_SELECTOR: [u8; 4] = [0xd2, 0xb1, 0x5c, 0xe6]; // tradeDistribution
const ADD_LIQUIDITY_SELECTOR: [u8; 4] = [0x82, 0x0f, 0x89, 0xeb]; // addLiquidity
const GET_MARKET_STATE_SELECTOR: [u8; 4] = [0x20, 0x1d, 0x2f, 0x2b]; // getMarketState
const GET_CONSENSUS_SELECTOR: [u8; 4] = [0x5a, 0xa6, 0x48, 0x00]; // getConsensusAt
const GET_METADATA_SELECTOR: [u8; 4] = [0x99, 0x8e, 0x84, 0xa3]; // getMetadata
const GET_MARKET_COUNT_SELECTOR: [u8; 4] = [0xfd, 0x69, 0xf3, 0xc2]; // getMarketCount
const CLOSE_POSITION_SELECTOR: [u8; 4] = [0x38, 0x4c, 0x07, 0xe6]; // closePosition
const GET_POSITION_SELECTOR: [u8; 4] = [0xac, 0x60, 0x5f, 0x39]; // getPosition
const GET_TRADER_POSITIONS_SELECTOR: [u8; 4] = [0x5f, 0xbb, 0xb2, 0xff]; // getTraderPositions

// Market structure
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
    lambda: u64,           // Scaling factor λ
    
    // Liquidity tracking
    total_lp_shares: u64,  // Total LP tokens
    total_backing: u64,    // Total backing in AMM
    
    // Trading state
    f_max: u64,           // Current max{f}
    min_variance: u64,    // Minimum variance allowed (σ² ≥ k²/b²π)
    
    // Position tracking
    next_position_id: u64, // Next position ID to assign
    total_volume: u64,     // Total trading volume
}

// Position structure - simplified without signed integers
#[derive(Clone)]
struct Position {
    // Identity
    position_id: u64,
    trader: [u8; 20],
    market_id: u64,
    
    // Entry state (when position opened)
    entry_mean: u64,
    entry_variance: u64,
    entry_lambda: u64,
    entry_f_max: u64,
    
    // Current state (for tracking)
    current_mean: u64,
    current_variance: u64,
    current_lambda: u64,
    current_f_max: u64,
    
    // Position details
    size: u64,
    collateral_locked: u64,
    fees_paid: u64,
    is_open: u8, // 1 = open, 0 = closed
    opened_at: u64,
    closed_at: u64,
    realized_pnl: u64, // simplified: 0 = loss, >0 = profit (offset by DECIMALS)
}

// Helper functions
fn get_timestamp() -> u64 {
    // For now, return a placeholder timestamp
    // In production, this would use block timestamp
    0
}

fn u256_bytes(value: u64) -> [u8; 32] {
    let mut result = [0u8; 32];
    result[0..8].copy_from_slice(&value.to_le_bytes());
    result
}

fn get_market_key(market_id: u64) -> [u8; 16] {
    let mut key = [0u8; 16];
    key[0..7].copy_from_slice(b"market_");
    key[8..16].copy_from_slice(&market_id.to_le_bytes());
    key
}

fn get_lp_balance_key(market_id: u64, address: &[u8; 20]) -> [u8; 44] {
    let mut key = [0u8; 44];
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

// Fixed-point math functions
fn mul_fixed(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) / DECIMALS as u128) as u64
}

fn div_fixed(a: u64, b: u64) -> u64 {
    if b == 0 {
        panic!("Division by zero");
    }
    ((a as u128 * DECIMALS as u128) / b as u128) as u64
}

fn sqrt_fixed(x: u64) -> u64 {
    if x == 0 {
        return 0;
    }
    
    // For fixed-point sqrt, if input x represents value X (in fixed-point with DECIMALS)
    // We want output to represent sqrt(X) (in fixed-point with DECIMALS)
    let x_scaled = (x as u128) * (DECIMALS as u128);
    
    // Integer square root using Newton's method
    let mut result = x_scaled;
    let mut last_result: u128 = 0;
    
    // Better initial guess
    if x_scaled > (1u128 << 32) {
        result = 1u128 << ((128 - x_scaled.leading_zeros()) / 2);
    }
    
    while result != last_result {
        last_result = result;
        if result > 0 {
            result = (result + x_scaled / result) / 2;
        }
    }
    
    result as u64
}

// Calculate e^(-x) using Taylor series for small x
fn exp_neg_fixed(x: u64) -> u64 {
    if x > 10 * DECIMALS {
        return 0; // e^(-10) ≈ 0
    }
    
    let mut result = DECIMALS;
    let mut term = DECIMALS;
    
    for i in 1..20 {
        term = mul_fixed(term, x) / i as u64;
        if i % 2 == 1 {
            result = result.saturating_sub(term);
        } else {
            result = result + term;
        }
        
        if term < 1000 { // Stop when term is negligible
            break;
        }
    }
    
    result
}

// Calculate Normal PDF value at x
fn normal_pdf(x: u64, mean: u64, variance: u64) -> u64 {
    let sigma = sqrt_fixed(variance);
    if sigma == 0 {
        return 0;
    }
    
    // Calculate (x - μ)²
    let diff = if x > mean { x - mean } else { mean - x };
    let diff_squared = mul_fixed(diff, diff);
    
    // Calculate exponent: -(x-μ)²/(2σ²)
    // Note: variance is already in fixed-point, so we need to be careful with scaling
    let two_variance = mul_fixed(2 * DECIMALS, variance);
    let exponent = div_fixed(diff_squared, two_variance);
    
    // Calculate e^(exponent)
    let exp_value = exp_neg_fixed(exponent);
    
    // Calculate normalization: 1/(σ√(2π))
    let sigma_sqrt_2pi = mul_fixed(sigma, SQRT_2PI_FIXED);
    let normalization = div_fixed(DECIMALS, sigma_sqrt_2pi);
    
    // Return PDF value
    mul_fixed(normalization, exp_value)
}

// Calculate L2 norm for normal distribution: ||p||₂ = 1/(2σ√π)
fn calculate_l2_norm_normal(variance: u64) -> u64 {
    let sigma = sqrt_fixed(variance);
    if sigma == 0 {
        return 0;
    }
    
    // Calculate 2σ√π
    let two_sigma_sqrt_pi = 2 * mul_fixed(sigma, SQRT_PI_FIXED);
    
    // Return 1/(2σ√π)
    div_fixed(DECIMALS, two_sigma_sqrt_pi)
}

// Calculate lambda for given k and variance
fn calculate_lambda(k_norm: u64, variance: u64) -> u64 {
    let l2_norm = calculate_l2_norm_normal(variance);
    if l2_norm == 0 {
        return 0;
    }
    
    div_fixed(k_norm, l2_norm)
}

// Calculate maximum value of f = λ * max(p) = k/(σ√π)
fn calculate_f_max(k_norm: u64, variance: u64) -> u64 {
    let sigma = sqrt_fixed(variance);
    if sigma == 0 {
        return 0;
    }
    
    let sigma_sqrt_pi = mul_fixed(sigma, SQRT_PI_FIXED);
    div_fixed(k_norm, sigma_sqrt_pi)
}

// Calculate minimum allowed variance: σ² ≥ k²/(b²π)
fn calculate_min_variance(k_norm: u64, b_backing: u64) -> u64 {
    let k_squared = mul_fixed(k_norm, k_norm);
    let b_squared = mul_fixed(b_backing, b_backing);
    let b_squared_pi = mul_fixed(b_squared, PI_FIXED);
    
    div_fixed(k_squared, b_squared_pi)
}

// Simple fee calculation
fn calculate_fees(size: u64) -> u64 {
    mul_fixed(size, 3 * DECIMALS / 1000) // 0.3% fee
}

// Simple collateral calculation
fn calculate_required_collateral(
    _from_mean: u64, _from_variance: u64, from_f_max: u64,
    _to_mean: u64, _to_variance: u64, to_f_max: u64,
    _size: u64
) -> u64 {
    // Simplified: require collateral equal to max possible loss
    from_f_max + to_f_max
}

// Simple P&L calculation
fn calculate_simple_pnl(
    entry_mean: u64, entry_variance: u64,
    exit_mean: u64, exit_variance: u64,
    size: u64
) -> u64 {
    // Simplified: profit if variance decreased or mean moved favorably
    let variance_change = if exit_variance < entry_variance {
        (entry_variance - exit_variance) / 1000 // Scale down
    } else {
        0
    };
    
    let mean_change = if exit_mean > entry_mean {
        (exit_mean - entry_mean) / 1000 // Scale down
    } else if entry_mean > exit_mean {
        (entry_mean - exit_mean) / 1000
    } else {
        0
    };
    
    // Return a simple profit metric (offset by DECIMALS to avoid negative)
    DECIMALS + mul_fixed(size, variance_change + mean_change)
}

// Storage helpers
fn load_market(market_id: u64) -> Option<Market> {
    let key = get_market_key(market_id);
    let mut buffer = [0u8; MAX_STORAGE_VALUE];
    
    let _ = api::get_storage(
        StorageFlags::empty(),
        &key,
        &mut &mut buffer[..],
    );
    
    if buffer[0] != 0 || buffer[1] != 0 {
        let mut market = Market {
            creator: [0u8; 20],
            creation_time: 0,
            close_time: 0,
            k_norm: 0,
            b_backing: 0,
            current_mean: 0,
            current_variance: 0,
            lambda: 0,
            total_lp_shares: 0,
            total_backing: 0,
            f_max: 0,
            min_variance: 0,
            next_position_id: 0,
            total_volume: 0,
        };
        
        market.creator.copy_from_slice(&buffer[0..20]);
        market.creation_time = u64::from_le_bytes(buffer[20..28].try_into().unwrap());
        market.close_time = u64::from_le_bytes(buffer[28..36].try_into().unwrap());
        market.k_norm = u64::from_le_bytes(buffer[36..44].try_into().unwrap());
        market.b_backing = u64::from_le_bytes(buffer[44..52].try_into().unwrap());
        market.current_mean = u64::from_le_bytes(buffer[52..60].try_into().unwrap());
        market.current_variance = u64::from_le_bytes(buffer[60..68].try_into().unwrap());
        market.lambda = u64::from_le_bytes(buffer[68..76].try_into().unwrap());
        market.total_lp_shares = u64::from_le_bytes(buffer[76..84].try_into().unwrap());
        market.total_backing = u64::from_le_bytes(buffer[84..92].try_into().unwrap());
        market.f_max = u64::from_le_bytes(buffer[92..100].try_into().unwrap());
        market.min_variance = u64::from_le_bytes(buffer[100..108].try_into().unwrap());
        market.next_position_id = u64::from_le_bytes(buffer[108..116].try_into().unwrap());
        market.total_volume = u64::from_le_bytes(buffer[116..124].try_into().unwrap());
        
        Some(market)
    } else {
        None
    }
}

fn save_market(market_id: u64, market: &Market) {
    let key = get_market_key(market_id);
    let mut buffer = [0u8; 160];
    
    buffer[0..20].copy_from_slice(&market.creator);
    buffer[20..28].copy_from_slice(&market.creation_time.to_le_bytes());
    buffer[28..36].copy_from_slice(&market.close_time.to_le_bytes());
    buffer[36..44].copy_from_slice(&market.k_norm.to_le_bytes());
    buffer[44..52].copy_from_slice(&market.b_backing.to_le_bytes());
    buffer[52..60].copy_from_slice(&market.current_mean.to_le_bytes());
    buffer[60..68].copy_from_slice(&market.current_variance.to_le_bytes());
    buffer[68..76].copy_from_slice(&market.lambda.to_le_bytes());
    buffer[76..84].copy_from_slice(&market.total_lp_shares.to_le_bytes());
    buffer[84..92].copy_from_slice(&market.total_backing.to_le_bytes());
    buffer[92..100].copy_from_slice(&market.f_max.to_le_bytes());
    buffer[100..108].copy_from_slice(&market.min_variance.to_le_bytes());
    buffer[108..116].copy_from_slice(&market.next_position_id.to_le_bytes());
    buffer[116..124].copy_from_slice(&market.total_volume.to_le_bytes());
    
    api::set_storage(
        StorageFlags::empty(),
        &key,
        &buffer[..124],
    );
}

fn save_position(position: &Position) {
    let key = get_position_key(position.position_id);
    let mut buffer = [0u8; 200];
    
    let mut offset = 0;
    buffer[offset..offset+8].copy_from_slice(&position.position_id.to_le_bytes());
    offset += 8;
    buffer[offset..offset+20].copy_from_slice(&position.trader);
    offset += 20;
    buffer[offset..offset+8].copy_from_slice(&position.market_id.to_le_bytes());
    offset += 8;
    buffer[offset..offset+8].copy_from_slice(&position.entry_mean.to_le_bytes());
    offset += 8;
    buffer[offset..offset+8].copy_from_slice(&position.entry_variance.to_le_bytes());
    offset += 8;
    buffer[offset..offset+8].copy_from_slice(&position.entry_lambda.to_le_bytes());
    offset += 8;
    buffer[offset..offset+8].copy_from_slice(&position.entry_f_max.to_le_bytes());
    offset += 8;
    buffer[offset..offset+8].copy_from_slice(&position.current_mean.to_le_bytes());
    offset += 8;
    buffer[offset..offset+8].copy_from_slice(&position.current_variance.to_le_bytes());
    offset += 8;
    buffer[offset..offset+8].copy_from_slice(&position.current_lambda.to_le_bytes());
    offset += 8;
    buffer[offset..offset+8].copy_from_slice(&position.current_f_max.to_le_bytes());
    offset += 8;
    buffer[offset..offset+8].copy_from_slice(&position.size.to_le_bytes());
    offset += 8;
    buffer[offset..offset+8].copy_from_slice(&position.collateral_locked.to_le_bytes());
    offset += 8;
    buffer[offset..offset+8].copy_from_slice(&position.fees_paid.to_le_bytes());
    offset += 8;
    buffer[offset] = position.is_open;
    offset += 1;
    buffer[offset..offset+8].copy_from_slice(&position.opened_at.to_le_bytes());
    offset += 8;
    buffer[offset..offset+8].copy_from_slice(&position.closed_at.to_le_bytes());
    offset += 8;
    buffer[offset..offset+8].copy_from_slice(&position.realized_pnl.to_le_bytes());
    offset += 8;
    
    api::set_storage(
        StorageFlags::empty(),
        &key,
        &buffer[..offset],
    );
}

fn add_trader_position(trader: &[u8; 20], position_id: u64) {
    // Get current count
    let count_key = get_trader_position_count_key(trader);
    let mut count_bytes = [0u8; 8];
    let _ = api::get_storage(
        StorageFlags::empty(),
        &count_key,
        &mut &mut count_bytes[..],
    );
    let count = u64::from_le_bytes(count_bytes);
    
    // Add position to trader's list
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

fn load_position(position_id: u64) -> Option<Position> {
    let key = get_position_key(position_id);
    let mut buffer = [0u8; 200];
    
    let result = api::get_storage(
        StorageFlags::empty(),
        &key,
        &mut &mut buffer[..],
    );
    
    if result.is_err() {
        return None;
    }
    
    let mut offset = 0;
    let position_id = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
    offset += 8;
    
    let mut trader = [0u8; 20];
    trader.copy_from_slice(&buffer[offset..offset+20]);
    offset += 20;
    
    let market_id = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
    offset += 8;
    
    let entry_mean = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
    offset += 8;
    
    let entry_variance = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
    offset += 8;
    
    let entry_lambda = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
    offset += 8;
    
    let entry_f_max = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
    offset += 8;
    
    let current_mean = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
    offset += 8;
    
    let current_variance = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
    offset += 8;
    
    let current_lambda = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
    offset += 8;
    
    let current_f_max = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
    offset += 8;
    
    let size = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
    offset += 8;
    
    let collateral_locked = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
    offset += 8;
    
    let fees_paid = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
    offset += 8;
    
    let is_open = buffer[offset];
    offset += 1;
    
    let opened_at = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
    offset += 8;
    
    let closed_at = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
    offset += 8;
    
    let realized_pnl = u64::from_le_bytes(buffer[offset..offset+8].try_into().unwrap());
    
    Some(Position {
        position_id,
        trader,
        market_id,
        entry_mean,
        entry_variance,
        entry_lambda,
        entry_f_max,
        current_mean,
        current_variance,
        current_lambda,
        current_f_max,
        size,
        collateral_locked,
        fees_paid,
        is_open,
        opened_at,
        closed_at,
        realized_pnl,
    })
}

fn save_market_metadata(market_id: u64, title: &str, description: &str, resolution_criteria: &str) {
    let key = get_metadata_key(market_id);
    
    let title_bytes = title.as_bytes();
    let desc_bytes = description.as_bytes();
    let criteria_bytes = resolution_criteria.as_bytes();
    
    let title_len = (title_bytes.len().min(255)) as u8;
    let desc_len = (desc_bytes.len().min(255)) as u8;
    let criteria_len = (criteria_bytes.len().min(255)) as u8;
    
    let total_len = 3 + title_len as usize + desc_len as usize + criteria_len as usize;
    let mut buffer = [0u8; MAX_STORAGE_VALUE];
    
    if total_len > MAX_STORAGE_VALUE {
        panic!("Metadata too long");
    }
    
    buffer[0] = title_len;
    buffer[1] = desc_len;
    buffer[2] = criteria_len;
    
    let mut offset = 3;
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

// Contract functions
fn handle_initialize() {
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
        panic!("Already initialized");
    }
    
    // Set owner
    api::set_storage(
        StorageFlags::empty(),
        OWNER_KEY,
        &caller,
    );
    
    // Set initialized
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
}

fn handle_create_market(data: &[u8]) {
    // Decode parameters: title, description, resolution_criteria, close_time, k_norm, initial_mean, initial_variance
    let tokens = match decode(&[
        ParamType::String,
        ParamType::String,
        ParamType::String,
        ParamType::Uint(64),
        ParamType::Uint(64),
        ParamType::Uint(64), 
        ParamType::Uint(64),
    ], data) {
        Ok(t) => t,
        Err(_) => panic!("Invalid parameters"),
    };
    
    let title = tokens[0].clone().into_string().unwrap();
    let description = tokens[1].clone().into_string().unwrap();
    let resolution_criteria = tokens[2].clone().into_string().unwrap();
    let close_time = tokens[3].clone().into_uint().unwrap().as_u64();
    let k_norm = tokens[4].clone().into_uint().unwrap().as_u64();
    let initial_mean = tokens[5].clone().into_uint().unwrap().as_u64();
    let initial_variance = tokens[6].clone().into_uint().unwrap().as_u64();
    
    // Get value transferred (initial backing)
    let value = u64_output!(api::value_transferred,);
    if value == 0 {
        panic!("Must provide initial backing");
    }
    
    let b_backing = value;
    
    // Calculate minimum variance constraint
    let min_variance = calculate_min_variance(k_norm, b_backing);
    if initial_variance < min_variance {
        panic!("Variance too low for backing constraint");
    }
    
    // Calculate initial AMM parameters
    let lambda = calculate_lambda(k_norm, initial_variance);
    let f_max = calculate_f_max(k_norm, initial_variance);
    
    if f_max > b_backing {
        panic!("Backing constraint violated");
    }
    
    // Get market ID
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
        creation_time: 0, // Would use block timestamp in real implementation
        close_time,
        k_norm,
        b_backing,
        current_mean: initial_mean,
        current_variance: initial_variance,
        lambda,
        total_lp_shares: b_backing, // Initial LP gets shares equal to backing
        total_backing: b_backing,
        f_max,
        min_variance,
        next_position_id: 0,
        total_volume: 0,
    };
    
    // Save market
    save_market(market_id, &market);
    
    // Save metadata
    save_market_metadata(market_id, &title, &description, &resolution_criteria);
    
    // Give creator initial LP shares
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
    let result = encode(&[Token::Uint(market_id.into())]);
    api::return_value(ReturnFlags::empty(), &result);
}

fn handle_trade_distribution(data: &[u8]) {
    // Decode parameters: market_id, new_mean, new_variance
    let tokens = match decode(&[
        ParamType::Uint(64),
        ParamType::Uint(64),
        ParamType::Uint(64),
    ], data) {
        Ok(t) => t,
        Err(_) => panic!("Invalid parameters"),
    };
    
    let market_id = tokens[0].clone().into_uint().unwrap().as_u64();
    let new_mean = tokens[1].clone().into_uint().unwrap().as_u64();
    let new_variance = tokens[2].clone().into_uint().unwrap().as_u64();
    
    // Load market
    let mut market = load_market(market_id).expect("Market not found");
    
    // Check variance constraint
    if new_variance < market.min_variance {
        panic!("Variance too low");
    }
    
    // Store entry state
    let entry_mean = market.current_mean;
    let entry_variance = market.current_variance;
    let entry_lambda = market.lambda;
    let entry_f_max = market.f_max;
    
    // Calculate new lambda and f_max
    let new_lambda = calculate_lambda(market.k_norm, new_variance);
    let new_f_max = calculate_f_max(market.k_norm, new_variance);
    
    if new_f_max > market.b_backing {
        panic!("Backing constraint violated");
    }
    
    // Calculate position size and collateral requirement
    let position_size = DECIMALS; // 1.0 for now, could be parameterized
    let collateral_required = calculate_required_collateral(
        entry_mean, entry_variance, entry_f_max,
        new_mean, new_variance, new_f_max,
        position_size
    );
    
    let value = u64_output!(api::value_transferred,);
    if value < collateral_required {
        panic!("Insufficient collateral");
    }
    
    // Calculate fees
    let fees = calculate_fees(position_size);
    
    // Create position
    let mut caller = [0u8; 20];
    api::caller(&mut caller);
    
    let position = Position {
        position_id: market.next_position_id,
        trader: caller,
        market_id,
        entry_mean,
        entry_variance,
        entry_lambda,
        entry_f_max,
        current_mean: new_mean,
        current_variance: new_variance,
        current_lambda: new_lambda,
        current_f_max: new_f_max,
        size: position_size,
        collateral_locked: value,
        fees_paid: fees,
        is_open: 1,
        opened_at: 0, // Would use block number
        closed_at: 0,
        realized_pnl: DECIMALS, // Start at break-even (offset)
    };
    
    // Save position
    save_position(&position);
    add_trader_position(&caller, position.position_id);
    
    // Update market state
    market.current_mean = new_mean;
    market.current_variance = new_variance;
    market.lambda = new_lambda;
    market.f_max = new_f_max;
    market.next_position_id += 1;
    market.total_volume += position_size;
    
    save_market(market_id, &market);
    
    // Return position ID
    let result = encode(&[Token::Uint(position.position_id.into())]);
    api::return_value(ReturnFlags::empty(), &result);
}

fn handle_add_liquidity(data: &[u8]) {
    // Decode parameters: market_id, proportion (in basis points, 10000 = 100%)
    let tokens = match decode(&[
        ParamType::Uint(64),
        ParamType::Uint(64),
    ], data) {
        Ok(t) => t,
        Err(_) => panic!("Invalid parameters"),
    };
    
    let market_id = tokens[0].clone().into_uint().unwrap().as_u64();
    let proportion_bps = tokens[1].clone().into_uint().unwrap().as_u64();
    
    // Load market
    let mut market = load_market(market_id).expect("Market not found");
    
    // Calculate required backing
    let required_backing = (market.total_backing * proportion_bps) / 10000;
    
    let value = u64_output!(api::value_transferred,);
    if value < required_backing {
        panic!("Insufficient liquidity");
    }
    
    // Calculate LP shares to mint
    let lp_shares_to_mint = (market.total_lp_shares * proportion_bps) / 10000;
    
    // Update market
    market.total_backing += required_backing;
    market.total_lp_shares += lp_shares_to_mint;
    market.b_backing = market.total_backing; // Update backing constraint
    
    // Recalculate minimum variance with new backing
    market.min_variance = calculate_min_variance(market.k_norm, market.b_backing);
    
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
}

fn handle_get_market_state(data: &[u8]) {
    let tokens = match decode(&[ParamType::Uint(64)], data) {
        Ok(t) => t,
        Err(_) => panic!("Invalid parameters"),
    };
    
    let market_id = tokens[0].clone().into_uint().unwrap().as_u64();
    let market = load_market(market_id).expect("Market not found");
    
    // Return market state
    let result = encode(&[
        Token::Uint(market.current_mean.into()),
        Token::Uint(market.current_variance.into()),
        Token::Uint(market.k_norm.into()),
        Token::Uint(market.b_backing.into()),
        Token::Uint(market.total_lp_shares.into()),
        Token::Uint(market.f_max.into()),
    ]);
    
    api::return_value(ReturnFlags::empty(), &result);
}

fn handle_get_consensus(data: &[u8]) {
    let tokens = match decode(&[
        ParamType::Uint(64),
        ParamType::Uint(64),
    ], data) {
        Ok(t) => t,
        Err(_) => panic!("Invalid parameters"),
    };
    
    let market_id = tokens[0].clone().into_uint().unwrap().as_u64();
    let x = tokens[1].clone().into_uint().unwrap().as_u64();
    
    let market = load_market(market_id).expect("Market not found");
    
    // Calculate f(x) = λ * p(x)
    let pdf_value = normal_pdf(x, market.current_mean, market.current_variance);
    let f_value = mul_fixed(market.lambda, pdf_value);
    
    // Return the consensus at x
    let result = encode(&[Token::Uint(f_value.into())]);
    api::return_value(ReturnFlags::empty(), &result);
}

fn handle_get_metadata(data: &[u8]) {
    let tokens = match decode(&[ParamType::Uint(64)], data) {
        Ok(t) => t,
        Err(_) => panic!("Invalid parameters"),
    };
    
    let market_id = tokens[0].clone().into_uint().unwrap().as_u64();
    let key = get_metadata_key(market_id);
    let mut buffer = [0u8; MAX_STORAGE_VALUE];
    
    let _ = api::get_storage(
        StorageFlags::empty(),
        &key,
        &mut &mut buffer[..],
    );
    
    let (title, description, resolution_criteria) = if buffer[0] == 0 && buffer[1] == 0 && buffer[2] == 0 {
        // No metadata found
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
    
    let result = encode(&[
        Token::String(title.into()),
        Token::String(description.into()),
        Token::String(resolution_criteria.into()),
    ]);
    
    api::return_value(ReturnFlags::empty(), &result);
}

fn handle_get_market_count() {
    let mut market_count_bytes = [0u8; 8];
    let _ = api::get_storage(
        StorageFlags::empty(),
        MARKET_COUNT_KEY,
        &mut &mut market_count_bytes[..],
    );
    let count = u64::from_le_bytes(market_count_bytes);
    
    let result = encode(&[Token::Uint(count.into())]);
    api::return_value(ReturnFlags::empty(), &result);
}

fn handle_get_trader_positions(data: &[u8]) {
    let tokens = match decode(&[ParamType::Address], data) {
        Ok(t) => t,
        Err(_) => panic!("Invalid parameters"),
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
    
    let result = encode(&[Token::Array(position_ids)]);
    api::return_value(ReturnFlags::empty(), &result);
}

fn handle_get_position(data: &[u8]) {
    let tokens = match decode(&[ParamType::Uint(64)], data) {
        Ok(t) => t,
        Err(_) => panic!("Invalid parameters"),
    };
    
    let position_id = tokens[0].clone().into_uint().unwrap().as_u64();
    
    match load_position(position_id) {
        Some(position) => {
            // Convert realized_pnl to signed integer for output
            // We store it as u64 with DECIMALS as offset for 0
            let realized_pnl_signed = if position.realized_pnl >= DECIMALS {
                (position.realized_pnl - DECIMALS) as i64
            } else {
                -((DECIMALS - position.realized_pnl) as i64)
            };
            
            let result = encode(&[
                Token::Uint(position.position_id.into()),
                Token::Address(position.trader.into()),
                Token::Uint(position.market_id.into()),
                Token::Uint(position.entry_mean.into()),
                Token::Uint(position.entry_variance.into()),
                Token::Uint(position.entry_lambda.into()),
                Token::Uint(position.entry_f_max.into()),
                Token::Uint(position.current_mean.into()),
                Token::Uint(position.current_variance.into()),
                Token::Uint(position.current_lambda.into()),
                Token::Uint(position.current_f_max.into()),
                Token::Uint(position.size.into()),
                Token::Uint(position.collateral_locked.into()),
                Token::Uint(position.fees_paid.into()),
                Token::Uint(position.is_open.into()),
                Token::Uint(position.opened_at.into()),
                Token::Uint(position.closed_at.into()),
                Token::Int(realized_pnl_signed.into()),
            ]);
            
            api::return_value(ReturnFlags::empty(), &result);
        }
        None => {
            panic!("Position not found");
        }
    }
}

fn handle_close_position(data: &[u8]) {
    let tokens = match decode(&[ParamType::Uint(64)], data) {
        Ok(t) => t,
        Err(_) => panic!("Invalid parameters"),
    };
    
    let position_id = tokens[0].clone().into_uint().unwrap().as_u64();
    
    // Load position
    let mut position = load_position(position_id).expect("Position not found");
    
    if position.is_open == 0 {
        panic!("Position already closed");
    }
    
    // Load current market state
    let market = load_market(position.market_id).expect("Market not found");
    
    // Calculate position value at current market state
    // Δs = λ' * f'_max - λ * f_max
    let current_value = mul_fixed(market.lambda, market.f_max);
    let entry_value = mul_fixed(position.entry_lambda, position.entry_f_max);
    
    let position_value = if current_value >= entry_value {
        // Profit
        let profit = current_value - entry_value;
        position.collateral_locked + profit
    } else {
        // Loss
        let loss = entry_value - current_value;
        if loss >= position.collateral_locked {
            // Total loss
            0
        } else {
            position.collateral_locked - loss
        }
    };
    
    // Update position state
    position.is_open = 0;
    position.closed_at = get_timestamp();
    position.current_mean = market.current_mean;
    position.current_variance = market.current_variance;
    position.current_lambda = market.lambda;
    position.current_f_max = market.f_max;
    
    // Calculate realized P&L
    if position_value >= position.collateral_locked {
        // Profit: store as DECIMALS + profit
        position.realized_pnl = DECIMALS + (position_value - position.collateral_locked);
    } else {
        // Loss: store as DECIMALS - loss
        position.realized_pnl = DECIMALS - (position.collateral_locked - position_value);
    }
    
    // Save updated position
    save_position(&position);
    
    // Return funds to trader if any
    if position_value > 0 {
        // Round down to nearest 1,000,000 wei to avoid DecimalPrecisionLoss
        // PolkaVM requires values >= 1,000,000 for transfers
        // This loses at most 0.000000000999999 ETH (999,999 wei)
        const MIN_TRANSFER_UNIT: u64 = 1_000_000;
        let rounded_value = (position_value / MIN_TRANSFER_UNIT) * MIN_TRANSFER_UNIT;
        
        // Only transfer if the rounded value is non-zero
        if rounded_value > 0 {
            // Transfer value back to the trader
            let transfer_amount = u256_bytes(rounded_value);
            let deposit_limit = [0xffu8; 32]; // Max deposit limit
            
            let result = api::call(
                CallFlags::empty(),     // No special flags needed
                &position.trader,       // Recipient
                0,                      // Use all available ref_time
                0,                      // Use all available proof_size
                &deposit_limit,         // Deposit limit
                &transfer_amount,       // Value to transfer
                &[],                    // Empty input data
                None,                   // No output needed
            );
            
            if result.is_err() {
                panic!("Transfer failed");
            }
        }
    }
    
    // Return success with position ID
    let result = encode(&[Token::Uint(position_id.into())]);
    api::return_value(ReturnFlags::empty(), &result);
}

// Main entry points
#[polkavm_export]
pub extern "C" fn deploy() {
    // Constructor - nothing to do
}

#[polkavm_export]
pub extern "C" fn call() {
    let length = api::call_data_size() as usize;
    
    if length == 0 {
        // Fallback function - accept value
        api::return_value(ReturnFlags::empty(), &[]);
    }
    
    if length < 4 {
        api::return_value(ReturnFlags::REVERT, b"Invalid input");
    }
    
    let mut selector = [0u8; 4];
    api::call_data_copy(&mut selector, 0);
    
    let mut data = [0u8; MAX_INPUT];
    let data_len = length.saturating_sub(4).min(MAX_INPUT);
    if data_len > 0 {
        api::call_data_copy(&mut data[..data_len], 4);
    }
    
    match selector {
        INITIALIZE_SELECTOR => handle_initialize(),
        CREATE_MARKET_SELECTOR => handle_create_market(&data[..data_len]),
        TRADE_DISTRIBUTION_SELECTOR => handle_trade_distribution(&data[..data_len]),
        ADD_LIQUIDITY_SELECTOR => handle_add_liquidity(&data[..data_len]),
        GET_MARKET_STATE_SELECTOR => handle_get_market_state(&data[..data_len]),
        GET_CONSENSUS_SELECTOR => handle_get_consensus(&data[..data_len]),
        GET_METADATA_SELECTOR => handle_get_metadata(&data[..data_len]),
        GET_MARKET_COUNT_SELECTOR => handle_get_market_count(),
        GET_TRADER_POSITIONS_SELECTOR => handle_get_trader_positions(&data[..data_len]),
        CLOSE_POSITION_SELECTOR => handle_close_position(&data[..data_len]),
        GET_POSITION_SELECTOR => handle_get_position(&data[..data_len]),
        _ => {
            // Unknown selector - accept as fallback
            api::return_value(ReturnFlags::empty(), &[]);
        }
    }
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe {
        core::arch::asm!("unimp");
        core::hint::unreachable_unchecked();
    }
}