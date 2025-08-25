# PolkaVM Smart Contract Pitfalls and Gotchas: A Comprehensive Guide

## Table of Contents
1. [Memory Management Pitfalls](#memory-management-pitfalls)
2. [Gas and Resource Exhaustion](#gas-and-resource-exhaustion)
3. [Storage Operation Gotchas](#storage-operation-gotchas)
4. [Input/Output Handling Issues](#inputoutput-handling-issues)
5. [Security Vulnerabilities](#security-vulnerabilities)
6. [ABI and Selector Problems](#abi-and-selector-problems)
7. [Inter-Contract Communication Dangers](#inter-contract-communication-dangers)
8. [Compilation and Deployment Issues](#compilation-and-deployment-issues)
9. [Type Safety and Conversion Errors](#type-safety-and-conversion-errors)
10. [State Management Anti-Patterns](#state-management-anti-patterns)
11. [Error Handling Mistakes](#error-handling-mistakes)
12. [Performance Traps](#performance-traps)

## Memory Management Pitfalls

### 1. Fixed Heap Size Limitations

**Problem:**
```rust
#[global_allocator]
static ALLOCATOR: simplealloc::SimpleAlloc<{ 1024 * 50 }> = simplealloc::SimpleAlloc::new();
```

**Issue:** The heap is fixed at compile time. Exceeding this will cause panics.

**Better Approach:**
```rust
// Calculate your actual memory needs
// Use stack allocation where possible
let mut buffer = [0u8; 1024]; // Stack allocation
```

### 2. Stack Overflow with Large Arrays

**Problem:**
```rust
pub extern "C" fn call() {
    let huge_buffer = [0u8; 1_000_000]; // Stack overflow!
}
```

**Solution:**
```rust
// Use smaller buffers or process in chunks
const CHUNK_SIZE: usize = 4096;
let mut chunk = [0u8; CHUNK_SIZE];
```

### 3. Memory Leaks with Dynamic Allocation

**Problem:**
```rust
// In no_std environment, Vec can leak if not properly handled
let mut data = Vec::with_capacity(1000);
// If function returns early, memory isn't freed
```

**Solution:**
- Prefer stack allocation
- Use fixed-size arrays
- Ensure proper cleanup in all code paths

### 4. Out-of-Memory Conditions

**Real Example from PolkaVM:**
```rust
// oom_rw_included.rs - This will fail deployment
static mut BUFFER: [u8; 513 * 1024] = [42; 513 * 1024];
```

**Gotcha:** Even zero-initialized buffers count toward memory limits:
```rust
// oom_rw_trailing.rs - Also fails
static mut BUFFER: [u8; 2 * 1024 * 1024] = [0; 2 * 1024 * 1024];
```

## Gas and Resource Exhaustion

### 1. Infinite Loops

**Problem:**
```rust
pub extern "C" fn call() {
    loop {} // Will exhaust gas and terminate
}
```

**Better Pattern:**
```rust
pub extern "C" fn call() {
    let max_iterations = 1000;
    for i in 0..max_iterations {
        if api::ref_time_left() < MIN_GAS_THRESHOLD {
            break; // Exit before exhaustion
        }
        // Do work
    }
}
```

### 2. Unbounded Storage Iterations

**Problem:**
```rust
// Iterating through all storage keys - dangerous!
for i in 0..u64::MAX {
    let key = get_key(i);
    api::get_storage(StorageFlags::empty(), &key, &mut value);
}
```

**Solution:**
- Implement pagination
- Store collection sizes
- Use bounded loops

### 3. Recursive Calls Without Depth Limits

**Problem:**
```rust
fn recursive_call(depth: u32) {
    api::call(
        CallFlags::ALLOW_REENTRY,
        &self_address,
        u64::MAX, // All gas forwarded!
        u64::MAX,
        &[u8::MAX; 32],
        &[0u8; 32],
        &(depth + 1).to_le_bytes(),
        None,
    );
}
```

**Solution:**
```rust
const MAX_DEPTH: u32 = 10;
if depth >= MAX_DEPTH {
    return; // Prevent stack overflow
}
```

### 4. Gas Calculation Errors

**Common Mistake:**
```rust
// Forwarding exact gas can fail due to overhead
let gas_to_forward = api::ref_time_left();
api::call(..., gas_to_forward, ...); // May fail!
```

**Correct Approach:**
```rust
// Reserve gas for post-call operations
let gas_to_forward = (api::ref_time_left() * 9) / 10; // Forward 90%
```

## Storage Operation Gotchas

### 1. Assuming Storage Exists

**Problem:**
```rust
let mut value = [0u8; 32];
api::get_storage(StorageFlags::empty(), &key, &mut value).unwrap(); // Panics if not found!
```

**Solution:**
```rust
let mut value = [0u8; 32];
match api::get_storage(StorageFlags::empty(), &key, &mut value) {
    Ok(_) => { /* Use value */ },
    Err(_) => { /* Handle missing key */ },
}
// Or use the convenience function:
api::get_storage_or_zero(StorageFlags::empty(), &key, &mut value);
```

### 2. Transient vs Persistent Storage Confusion

**Problem:**
```rust
// Setting in transient storage
api::set_storage(StorageFlags::TRANSIENT, &key, &value);

// Later, trying to read from persistent storage
api::get_storage(StorageFlags::empty(), &key, &mut value); // Won't find it!
```

**Best Practice:**
```rust
// Create helper functions to avoid confusion
fn set_persistent(key: &[u8], value: &[u8]) {
    api::set_storage(StorageFlags::empty(), key, value);
}

fn set_transient(key: &[u8], value: &[u8]) {
    api::set_storage(StorageFlags::TRANSIENT, key, value);
}
```

### 3. Storage Key Collisions

**Problem:**
```rust
// These could collide!
fn get_balance_key(user: &[u8; 20]) -> [u8; 32] {
    let mut key = [0u8; 32];
    key[..20].copy_from_slice(user);
    key
}

fn get_allowance_key(owner: &[u8; 20]) -> [u8; 32] {
    let mut key = [0u8; 32];
    key[..20].copy_from_slice(owner); // Same pattern!
    key
}
```

**Solution:**
```rust
// Use prefixes to namespace storage
fn get_balance_key(user: &[u8; 20]) -> [u8; 32] {
    let mut key = [1u8; 32]; // Prefix: 1 for balances
    key[1..21].copy_from_slice(user);
    key
}

fn get_allowance_key(owner: &[u8; 20], spender: &[u8; 20]) -> [u8; 32] {
    let mut key = [2u8; 32]; // Prefix: 2 for allowances
    // Hash the combination for uniqueness
    let mut data = [0u8; 40];
    data[..20].copy_from_slice(owner);
    data[20..].copy_from_slice(spender);
    api::hash_keccak_256(&data, &mut key[1..]);
    key
}
```

### 4. Zero-Value Storage Behavior

**Gotcha:** Setting 32-byte zero values clears storage (Ethereum compatibility):
```rust
let zero = [0u8; 32];
api::set_storage_or_clear(StorageFlags::empty(), &key, &zero); // Deletes the key!
```

**Important:** This only applies to the specialized 32-byte functions.

## Input/Output Handling Issues

### 1. Fixed Buffer Overflow

**Problem:**
```rust
const MAX_INPUT: usize = 100;
let mut buffer = [0u8; MAX_INPUT];
api::call_data_copy(&mut buffer, 0); // What if input > 100 bytes?
```

**Solution:**
```rust
let input_size = api::call_data_size();
if input_size > MAX_INPUT {
    panic!("Input too large");
}
let mut buffer = [0u8; MAX_INPUT];
api::call_data_copy(&mut buffer[..input_size.min(MAX_INPUT)], 0);
```

### 2. Incorrect Input Parsing

**Problem:**
```rust
input!(
    selector: &[u8; 4],
    amount: u64,        // Assumes exactly 8 bytes follow
    recipient: &[u8; 20], // Assumes exactly 20 bytes follow
);
// Panics if input is shorter!
```

**Better Approach:**
```rust
// Check size first
if api::call_data_size() < 32 { // 4 + 8 + 20
    panic!("Invalid input size");
}
input!(
    selector: &[u8; 4],
    amount: u64,
    recipient: &[u8; 20],
);
```

### 3. Endianness Confusion

**Problem:**
```rust
// Ethereum uses big-endian for U256
let amount_bytes = [0u8; 32];
api::call_data_copy(&mut amount_bytes, 4);
let amount = u64::from_le_bytes(amount_bytes[..8].try_into().unwrap()); // Wrong!
```

**Correct:**
```rust
use primitive_types::U256;
let amount_bytes = [0u8; 32];
api::call_data_copy(&mut amount_bytes, 4);
let amount = U256::from_big_endian(&amount_bytes); // Ethereum standard
```

### 4. Return Data Buffer Sizing

**Problem:**
```rust
let mut output = [0u8; 32]; // Fixed size
api::call(..., Some(&mut output))?;
// What if return data > 32 bytes?
```

**Solution:**
```rust
// First, make the call
api::call(..., None)?; // Don't capture output yet

// Check return size
let return_size = api::return_data_size();
if return_size > MAX_RETURN_SIZE {
    panic!("Return data too large");
}

// Now copy what we need
let mut output = [0u8; 32];
api::return_data_copy(&mut output[..return_size.min(32)], 0);
```

## Security Vulnerabilities

### 1. Reentrancy Attacks

**Vulnerable Pattern:**
```rust
// Update balance AFTER external call - WRONG!
let balance = get_balance(sender);
api::call(
    CallFlags::ALLOW_REENTRY, // Dangerous!
    recipient,
    // ...
)?;
set_balance(sender, balance - amount); // State change after call
```

**Secure Pattern:**
```rust
// Checks-Effects-Interactions pattern
let balance = get_balance(sender);
if balance < amount {
    panic!("Insufficient balance");
}
set_balance(sender, balance - amount); // State change BEFORE call

api::call(
    CallFlags::empty(), // No reentrancy by default
    recipient,
    // ...
)?;
```

### 2. Integer Overflow/Underflow

**Problem:**
```rust
let balance: u64 = get_balance(user);
let new_balance = balance + amount; // Can overflow!
set_balance(user, new_balance);
```

**Solution:**
```rust
let balance: u64 = get_balance(user);
let new_balance = balance.checked_add(amount)
    .expect("Balance overflow");
set_balance(user, new_balance);
```

### 3. Unchecked External Calls

**Problem:**
```rust
// Ignoring call results
let _ = api::call(...); // Return value ignored!
// Continues execution even if call failed
```

**Correct:**
```rust
match api::call(...) {
    Ok(()) => { /* Handle success */ },
    Err(ReturnErrorCode::CalleeReverted) => {
        // Check return data for revert reason
        let size = api::return_data_size();
        // Handle revert
    },
    Err(e) => panic!("Call failed: {:?}", e),
}
```

### 4. Delegate Call Storage Corruption

**Dangerous:**
```rust
// Calling unknown code with your storage context
api::delegate_call(
    CallFlags::empty(),
    &untrusted_address, // Could corrupt your storage!
    // ...
)?;
```

**Safe Approach:**
```rust
// Only delegate to known, audited contracts
const TRUSTED_LOGIC: [u8; 20] = [/* known address */];
if callee != TRUSTED_LOGIC {
    panic!("Unauthorized delegate call");
}
```

### 5. Missing Access Control

**Problem:**
```rust
pub extern "C" fn call() {
    input!(
        selector: &[u8; 4],
    );
    
    match u32::from_le_bytes(*selector) {
        0x12345678 => admin_function(), // No permission check!
        _ => {},
    }
}
```

**Solution:**
```rust
fn admin_function() {
    let mut caller = [0u8; 20];
    api::caller(&mut caller);
    
    let mut owner = [0u8; 20];
    api::get_storage(StorageFlags::empty(), &OWNER_KEY, &mut owner).unwrap();
    
    if caller != owner {
        panic!("Unauthorized");
    }
    // Admin logic
}
```

## ABI and Selector Problems

### 1. Selector Collisions

**Problem:**
```rust
// These could have the same 4-byte selector!
// "transfer(address,uint256)" -> 0xa9059cbb
// "gasprice_bit_ether(int128)" -> 0xa9059cbb (collision!)
```

**Prevention:**
```rust
// Use a compile-time check or external tool
const SELECTORS: &[(u32, &str)] = &[
    (0xa9059cbb, "transfer"),
    (0x23b872dd, "transferFrom"),
    // Add all your functions
];

// Verify uniqueness in tests
#[test]
fn test_no_selector_collisions() {
    let mut seen = std::collections::HashSet::new();
    for (selector, name) in SELECTORS {
        assert!(seen.insert(selector), "Collision for {}", name);
    }
}
```

### 2. Hardcoded Magic Selectors

**Bad Practice:**
```rust
match u32::from_le_bytes(selector) {
    0x12345678 => {}, // What function is this?
    0x87654321 => {}, // No documentation!
}
```

**Good Practice:**
```rust
// Define constants with function signatures
const TRANSFER: u32 = 0xa9059cbb; // transfer(address,uint256)
const APPROVE: u32 = 0x095ea7b3;  // approve(address,uint256)

match u32::from_le_bytes(selector) {
    TRANSFER => transfer(),
    APPROVE => approve(),
    _ => panic!("Unknown function"),
}
```

### 3. Variable-Length Parameter Issues

**Problem:**
```rust
input!(
    selector: &[u8; 4],
    data: [u8], // Consumes ALL remaining bytes
    more_data: u32, // This will never be reached!
);
```

**Correct Order:**
```rust
input!(
    selector: &[u8; 4],
    fixed_param: u32,
    another_fixed: &[u8; 20],
    variable_data: [u8], // Variable-length MUST be last
);
```

### 4. ABI Encoding Mismatches

**Common Mistake:**
```rust
// Contract expects: transfer(address,uint256)
// But decoding as:
input!(
    selector: &[u8; 4],
    recipient: &[u8; 20],
    amount: u64, // Should be u256 (32 bytes)!
);
```

**Correct:**
```rust
input!(
    selector: &[u8; 4],
    recipient: &[u8; 20],
    _padding: &[u8; 12], // Addresses are padded to 32 bytes
    amount: &[u8; 32],   // Full u256
);
```

## Inter-Contract Communication Dangers

### 1. Forwarding Arbitrary Input

**Dangerous:**
```rust
api::call(
    CallFlags::FORWARD_INPUT, // Forwards raw input including selector!
    &target,
    // ...
)?;
```

**Issue:** The target contract receives your selector, not theirs.

**Better:**
```rust
// Skip your selector when forwarding
let input_size = api::call_data_size();
let mut forward_data = vec![0u8; input_size - 4];
api::call_data_copy(&mut forward_data, 4); // Skip first 4 bytes

api::call(
    CallFlags::empty(),
    &target,
    // ...
    &forward_data,
    None,
)?;
```

### 2. Gas Limit Confusion

**Problem:**
```rust
// These are TWO different limits!
api::call(
    CallFlags::empty(),
    &target,
    1_000_000,      // ref_time limit
    500_000,        // proof_size limit - often forgotten!
    // ...
)?;
```

**Best Practice:**
```rust
// Forward proportional gas
let ref_time = (api::ref_time_left() * 90) / 100;
let proof_size = u64::MAX; // Usually not the bottleneck

api::call(
    CallFlags::empty(),
    &target,
    ref_time,
    proof_size,
    // ...
)?;
```

### 3. Value Transfer Failures

**Problem:**
```rust
// This can fail silently in some patterns
let value = U256::from(1000).to_big_endian();
api::call(
    CallFlags::empty(),
    &recipient,
    // ...
    &value,
    // ...
)?; // Call succeeds even if transfer fails internally
```

**Solution:**
```rust
// Check balance before and after
let mut balance_before = [0u8; 32];
api::balance(&mut balance_before);

api::call(/* with value */)?;

let mut balance_after = [0u8; 32];
api::balance(&mut balance_after);

// Verify transfer occurred
let before = U256::from_big_endian(&balance_before);
let after = U256::from_big_endian(&balance_after);
assert!(after < before, "Value transfer failed");
```

## Compilation and Deployment Issues

### 1. Wrong Target Architecture

**Problem:**
```bash
cargo build --release # Uses default target!
```

**Correct:**
```bash
cargo build --release --target riscv64emac-unknown-none-polkavm
```

### 2. Missing Optimization

**Unoptimized Cargo.toml:**
```toml
[profile.release]
# Default settings = larger binary
```

**Optimized:**
```toml
[profile.release]
opt-level = "s"     # Size optimization
lto = "fat"         # Link-time optimization
codegen-units = 1   # Better optimization
strip = true        # Remove symbols
```

### 3. Forgotten Polkatool Step

**Incomplete:**
```bash
cargo build --release --target riscv64emac-unknown-none-polkavm
# Binary isn't ready for deployment!
```

**Complete:**
```bash
cargo build --release --target riscv64emac-unknown-none-polkavm
polkatool link --strip \
    --output contract.polkavm \
    target/riscv64emac-unknown-none-polkavm/release/contract
```

### 4. Deployment with Wrong Constructor Args

**Problem:**
```rust
// Deploy expects: constructor(address,uint256)
// But sending: constructor(uint256,address)
```

**Debugging Tip:**
```rust
pub extern "C" fn deploy() {
    // Log the input for debugging
    let size = api::call_data_size();
    let mut data = vec![0u8; size];
    api::call_data_copy(&mut data, 0);
    api::deposit_event(&[[1u8; 32]], &data); // Log input
    
    // Then parse
    input!(/* ... */);
}
```

## Type Safety and Conversion Errors

### 1. U256 to Primitive Conversions

**Dangerous:**
```rust
let value = U256::from_big_endian(&bytes);
let amount = value.as_u64(); // Truncates if > u64::MAX!
```

**Safe:**
```rust
let value = U256::from_big_endian(&bytes);
if value > U256::from(u64::MAX) {
    panic!("Value too large");
}
let amount = value.as_u64();
```

### 2. Unchecked Array Conversions

**Problem:**
```rust
let data = &bytes[0..8];
let value = u64::from_le_bytes(data.try_into().unwrap()); // Panics if not 8 bytes
```

**Better:**
```rust
if bytes.len() < 8 {
    panic!("Insufficient data");
}
let mut data = [0u8; 8];
data.copy_from_slice(&bytes[0..8]);
let value = u64::from_le_bytes(data);
```

### 3. Mixed Decimal Precision

**Confusing:**
```rust
// Some contracts use 18 decimals (ETH standard)
const DECIMALS_18: u64 = 1_000_000_000_000_000_000;

// Others use 9 decimals
const DECIMALS_9: u64 = 1_000_000_000;

// Mixing them causes errors!
let amount = user_input * DECIMALS_18; // Which one?
```

**Solution:**
```rust
// Be explicit and consistent
const TOKEN_DECIMALS: u8 = 18;
const TOKEN_SCALE: u64 = 10_u64.pow(TOKEN_DECIMALS as u32);

// Document in comments
/// All amounts are in base units (wei equivalent)
fn transfer(amount_in_base_units: U256) { /* ... */ }
```

## State Management Anti-Patterns

### 1. Partial State Updates

**Problem:**
```rust
// Multiple storage operations - not atomic!
api::set_storage(StorageFlags::empty(), &balance_key, &new_balance);
// If this fails, state is inconsistent!
api::set_storage(StorageFlags::empty(), &total_supply_key, &new_supply);
```

**Better Pattern:**
```rust
// Validate everything first
let new_balance = calculate_new_balance()?;
let new_supply = calculate_new_supply()?;

// Then update atomically (all or nothing)
api::set_storage(StorageFlags::empty(), &balance_key, &new_balance);
api::set_storage(StorageFlags::empty(), &total_supply_key, &new_supply);
```

### 2. Missing State Initialization

**Problem:**
```rust
pub extern "C" fn call() {
    // Assumes storage was initialized in deploy()
    let mut config = [0u8; 32];
    api::get_storage(StorageFlags::empty(), &CONFIG_KEY, &mut config)
        .unwrap(); // Panics if not initialized!
}
```

**Robust Pattern:**
```rust
fn get_config() -> Config {
    let mut data = [0u8; 32];
    match api::get_storage(StorageFlags::empty(), &CONFIG_KEY, &mut data) {
        Ok(_) => Config::from_bytes(&data),
        Err(_) => Config::default(), // Fallback to defaults
    }
}
```

### 3. Race Conditions in Initialization

**Problem:**
```rust
pub extern "C" fn deploy() {
    // Check if initialized
    let mut init = [0u8; 1];
    if api::get_storage(StorageFlags::empty(), &INIT_KEY, &mut init).is_ok() {
        panic!("Already initialized");
    }
    // RACE: Another call could initialize here!
    
    // Set initialized
    api::set_storage(StorageFlags::empty(), &INIT_KEY, &[1u8]);
}
```

**Better:**
```rust
pub extern "C" fn deploy() {
    // Atomic check-and-set
    let prev = api::set_storage(StorageFlags::empty(), &INIT_KEY, &[1u8]);
    if prev.is_some() {
        panic!("Already initialized");
    }
    // Now safe to continue initialization
}
```

## Error Handling Mistakes

### 1. Panic vs Revert

**Bad:**
```rust
if condition {
    panic!("Error message"); // User won't see this message!
}
```

**Good:**
```rust
if condition {
    // Revert with data the user can decode
    let error = b"InsufficientBalance";
    api::return_value(ReturnFlags::REVERT, error);
}
```

### 2. Silent Failures

**Problem:**
```rust
let _ = api::set_storage(StorageFlags::empty(), &key, &value); // Ignoring result
```

**Better:**
```rust
api::set_storage(StorageFlags::empty(), &key, &value);
// Storage operations don't typically fail, but be aware of gas costs
```

### 3. Unwrap in Production

**Never Do This:**
```rust
let value = some_option.unwrap(); // Can panic
let result = some_result.unwrap(); // Can panic
```

**Always:**
```rust
let value = some_option.expect("Specific error message");
// Or better:
let value = match some_option {
    Some(v) => v,
    None => {
        api::return_value(ReturnFlags::REVERT, b"ValueNotFound");
    }
};
```

## Performance Traps

### 1. Unnecessary Storage Reads

**Inefficient:**
```rust
for i in 0..100 {
    let mut value = [0u8; 32];
    api::get_storage(StorageFlags::empty(), &KEY, &mut value).unwrap();
    // Same value read 100 times!
}
```

**Efficient:**
```rust
let mut value = [0u8; 32];
api::get_storage(StorageFlags::empty(), &KEY, &mut value).unwrap();
for i in 0..100 {
    // Use cached value
}
```

### 2. Large Event Data

**Problem:**
```rust
let huge_data = [0u8; 100_000];
api::deposit_event(&[], &huge_data); // Expensive!
```

**Better:**
```rust
// Only emit essential data
let summary = create_summary(&huge_data); // 32 bytes
api::deposit_event(&[], &summary);
```

### 3. Redundant Hashing

**Inefficient:**
```rust
fn get_key(a: &[u8], b: &[u8]) -> [u8; 32] {
    let mut result = [0u8; 32];
    let mut data = Vec::new();
    data.extend_from_slice(a);
    data.extend_from_slice(b);
    api::hash_keccak_256(&data, &mut result); // Allocates vector
    result
}
```

**Efficient:**
```rust
fn get_key(a: &[u8], b: &[u8]) -> [u8; 32] {
    let mut result = [0u8; 32];
    // Stack allocation
    let mut data = [0u8; 64]; // Assuming known sizes
    data[..32].copy_from_slice(a);
    data[32..].copy_from_slice(b);
    api::hash_keccak_256(&data, &mut result);
    result
}
```

## Summary of Key Takeaways

1. **Always validate inputs** - Never trust external data
2. **Check all return values** - Don't ignore Results
3. **Use safe math** - Prevent overflows/underflows
4. **Manage gas carefully** - Reserve gas for cleanup
5. **Prevent reentrancy** - Use checks-effects-interactions
6. **Initialize state properly** - Handle missing storage gracefully
7. **Document your ABI** - Prevent selector collisions
8. **Test edge cases** - Empty inputs, max values, gas exhaustion
9. **Optimize for size** - PolkaVM contracts have size limits
10. **Use the type system** - Let Rust help you catch errors

Remember: PolkaVM gives you low-level control, which means you're responsible for implementing safety measures that might be automatic in other environments. When in doubt, be explicit and defensive in your code.