# Comprehensive Guide to Writing Smart Contracts in Rust for PolkaVM

## Table of Contents
1. [Introduction](#introduction)
2. [Environment Setup](#environment-setup)
3. [Contract Structure](#contract-structure)
4. [Entry Points](#entry-points)
5. [Input/Output Handling](#inputoutput-handling)
6. [Storage Operations](#storage-operations)
7. [Inter-Contract Communication](#inter-contract-communication)
8. [Gas and Resource Management](#gas-and-resource-management)
9. [Cryptographic Functions](#cryptographic-functions)
10. [Contract Deployment](#contract-deployment)
11. [Advanced Features](#advanced-features)
12. [Compilation Process](#compilation-process)
13. [Best Practices](#best-practices)
14. [Example Contracts](#example-contracts)

## Introduction

PolkaVM is a RISC-V based virtual machine designed for executing smart contracts on Substrate-based blockchains. The pallet-revive framework provides an Ethereum-compatible execution environment with enhanced features. This guide covers everything you need to know to write, compile, and deploy smart contracts in Rust for PolkaVM.

## Environment Setup

### Prerequisites

1. **Rust Nightly Toolchain**
   ```bash
   rustup toolchain install nightly-2024-11-19
   rustup component add rust-src --toolchain nightly-2024-11-19
   ```

2. **PolkaVM Tools**
   ```bash
   cargo install polkatool
   ```

3. **Project Configuration**

   Create `rust-toolchain.toml`:
   ```toml
   [toolchain]
   channel = "nightly-2024-11-19"
   components = ["rust-src"]
   ```

   Create `.cargo/config.toml`:
   ```toml
   [build]
   target = "riscv64emac-unknown-none-polkavm.json"
   
   [unstable]
   build-std = ["core", "alloc"]
   build-std-features = ["panic_immediate_abort"]
   ```

   Create the custom target file `riscv64emac-unknown-none-polkavm.json`:
   ```json
   {
     "arch": "riscv64",
     "cpu": "generic-rv64",
     "features": "+e,+m,+a,+c,+zbb",
     "llvm-target": "riscv64",
     "panic-strategy": "abort",
     "relocation-model": "pie",
     "target-endian": "little",
     "target-pointer-width": "64",
     "data-layout": "e-m:e-p:64:64-i64:64-i128:128-n32:64-S64",
     "linker-flavor": "gnu",
     "pre-link-args": {
       "gnu": ["-nostartfiles", "--emit-relocs", "--unique", "--relocatable"]
     }
   }
   ```

## Contract Structure

Every PolkaVM contract follows this basic structure:

```rust
#![no_std]  // No standard library
#![no_main] // No main function

// Include the panic handler
include!("path/to/panic_handler.rs")

// Import the UAPI (User API) for blockchain interactions
use uapi::{HostFn, HostFnImpl as api};

// Optional: Initialize a simple allocator for dynamic memory
#[global_allocator]
static ALLOCATOR: simplealloc::SimpleAlloc<4096> = simplealloc::SimpleAlloc::new();

// Contract entry point for deployment (constructor)
#[no_mangle]
#[polkavm_derive::polkavm_export]
pub extern "C" fn deploy() {
    // Constructor logic
}

// Contract entry point for calls (runtime)
#[no_mangle]
#[polkavm_derive::polkavm_export]
pub extern "C" fn call() {
    // Runtime logic
}
```

### Panic Handler

Create a custom panic handler since `no_std` environment doesn't have one:

```rust
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // Use the unimp instruction to trap and revert
    unsafe {
        core::arch::asm!("unimp");
        core::hint::unreachable_unchecked();
    }
}
```

## Entry Points

PolkaVM contracts have two mandatory entry points:

### 1. Deploy Function (Constructor)
- Called once when the contract is instantiated
- Can access constructor arguments via `call_data`
- Can set initial storage values
- Can emit events
- Can even self-destruct

```rust
#[no_mangle]
#[polkavm_derive::polkavm_export]
pub extern "C" fn deploy() {
    // Parse constructor arguments
    input!(
        initial_supply: u64,
        owner: &[u8; 20],
    );
    
    // Set initial storage
    api::set_storage(StorageFlags::empty(), &TOTAL_SUPPLY_KEY, &initial_supply.to_le_bytes());
    api::set_storage(StorageFlags::empty(), &owner_key(owner), &initial_supply.to_le_bytes());
    
    // Emit deployment event
    let topics = [[0x01; 32]]; // Event signature
    api::deposit_event(&topics, &initial_supply.to_le_bytes());
}
```

### 2. Call Function (Runtime)
- Called for every contract interaction after deployment
- Handles all contract methods
- Typically uses a dispatcher pattern

```rust
#[no_mangle]
#[polkavm_derive::polkavm_export]
pub extern "C" fn call() {
    // Read the function selector (first 4 bytes)
    let mut selector = [0u8; 4];
    api::call_data_copy(&mut selector, 0);
    
    // Dispatch based on selector
    match u32::from_le_bytes(selector) {
        0x70a08231 => balance_of(),    // balanceOf(address)
        0xa9059cbb => transfer(),       // transfer(address,uint256)
        0x23b872dd => transfer_from(), // transferFrom(address,address,uint256)
        _ => panic!("Unknown selector"),
    }
}
```

## Input/Output Handling

### Reading Input Data

Use the `input!` macro for structured input parsing:

```rust
// Fixed-size parameters
input!(
    selector: &[u8; 4],      // Read 4 bytes
    address: &[u8; 20],      // Read 20 bytes
    amount: u64,             // Read 8 bytes
);

// Variable-size data
input!(
    data_len: u32,           // Read length first
    data: [u8],              // Read remaining bytes
);

// Mixed parameters
input!(
    100,                     // Expected total size
    param1: u32,
    param2: &[u8; 32],
    remaining: [u8],         // Rest of the data
);
```

### Manual Input Reading

```rust
// Get total input size
let size = api::call_data_size();

// Copy specific range
let mut buffer = [0u8; 32];
api::call_data_copy(&mut buffer, offset);

// Load 32 bytes efficiently
let mut data = [0u8; 32];
api::call_data_load(&mut data, offset);
```

### Returning Output

```rust
// Return data and terminate execution
api::return_value(ReturnFlags::empty(), &output_data);

// Revert with data
api::return_value(ReturnFlags::REVERT, &error_data);
```

## Storage Operations

### Persistent Storage

```rust
// Set storage
let key = [1u8; 32];
let value = [2u8; 32];
let previous_size = api::set_storage(StorageFlags::empty(), &key, &value);

// Get storage
let mut output = [0u8; 32];
match api::get_storage(StorageFlags::empty(), &key, &mut output) {
    Ok(size) => { /* Value found, size bytes written */ },
    Err(_) => { /* Key not found */ },
}

// Clear storage
let removed_size = api::clear_storage(StorageFlags::empty(), &key);

// Check if key exists
match api::contains_storage(StorageFlags::empty(), &key) {
    Some(size) => { /* Key exists with size bytes */ },
    None => { /* Key doesn't exist */ },
}

// Take storage (get and remove atomically)
let mut output = [0u8; 32];
match api::take_storage(StorageFlags::empty(), &key, &mut output) {
    Ok(size) => { /* Value taken */ },
    Err(_) => { /* Key not found */ },
}
```

### Transient Storage

Transient storage is cleared after each transaction:

```rust
// Use the TRANSIENT flag
api::set_storage(StorageFlags::TRANSIENT, &key, &value);
api::get_storage(StorageFlags::TRANSIENT, &key, &mut output);
```

### Optimized 32-byte Operations

For Ethereum compatibility:

```rust
// Automatically clears if value is all zeros
let zero_value = [0u8; 32];
api::set_storage_or_clear(StorageFlags::empty(), &key, &zero_value);

// Returns zeros if key doesn't exist
let mut output = [0u8; 32];
api::get_storage_or_zero(StorageFlags::empty(), &key, &mut output);
```

## Inter-Contract Communication

### Making Contract Calls

```rust
// Basic call
let result = api::call(
    CallFlags::empty(),       // Flags
    &callee_address,         // Contract address [u8; 20]
    u64::MAX,                // ref_time limit (use all available)
    u64::MAX,                // proof_size limit
    &[u8::MAX; 32],         // deposit limit (no limit)
    &value,                  // Value to transfer [u8; 32]
    &input_data,            // Call data
    None,                    // Output buffer (None = ignore output)
);

// Handle result
match result {
    Ok(()) => { /* Success */ },
    Err(code) => {
        match code {
            ReturnErrorCode::CalleeReverted => { /* Contract reverted */ },
            ReturnErrorCode::CalleeTrapped => { /* Contract panicked */ },
            ReturnErrorCode::TransferFailed => { /* Value transfer failed */ },
            _ => { /* Other error */ },
        }
    }
}
```

### Call Flags

```rust
// Forward input to callee (consumes input)
CallFlags::FORWARD_INPUT

// Clone input to callee (preserves input)
CallFlags::CLONE_INPUT

// Return callee's output directly to caller
CallFlags::TAIL_CALL

// Allow reentrancy
CallFlags::ALLOW_REENTRY

// Read-only call (no state changes)
CallFlags::READ_ONLY

// Combine flags
CallFlags::ALLOW_REENTRY | CallFlags::READ_ONLY
```

### Delegate Calls

Execute another contract's code in your storage context:

```rust
api::delegate_call(
    CallFlags::empty(),
    &code_address,          // Address with code to execute
    u64::MAX,               // ref_time limit
    u64::MAX,               // proof_size limit
    &[u8::MAX; 32],        // deposit limit
    &input_data,
    None,
);
```

### Accessing Return Data

```rust
// After a successful call
let return_size = api::return_data_size();
let mut return_buffer = vec![0u8; return_size];
api::return_data_copy(&mut return_buffer, 0);
```

## Gas and Resource Management

### Checking Gas

```rust
// Get remaining computational time
let gas_left = api::ref_time_left();

// Get current gas price
let mut price = [0u8; 32];
api::gas_price(&mut price);

// Get block gas limit
let limit = api::gas_limit();
```

### Making Gas-Limited Calls

```rust
// Calculate gas to forward (e.g., 90% of remaining)
let gas_to_forward = (api::ref_time_left() * 9) / 10;

api::call(
    CallFlags::empty(),
    &callee,
    gas_to_forward,      // Specific gas limit
    u64::MAX,            // proof_size
    &[u8::MAX; 32],
    &[0u8; 32],
    &input,
    None,
)?;
```

## Cryptographic Functions

### Hash Functions

```rust
// Keccak-256 (Ethereum compatible)
let mut hash = [0u8; 32];
api::hash_keccak_256(&input_data, &mut hash);

// Blake2-128
let mut hash = [0u8; 16];
api::hash_blake2_128(&input_data, &mut hash);
```

### Signature Verification

```rust
// SR25519 signature verification
let signature = [0u8; 64];  // 64-byte signature
let message = b"Hello, World!";
let public_key = [0u8; 32]; // 32-byte public key

match api::sr25519_verify(&signature, message, &public_key) {
    Ok(()) => { /* Valid signature */ },
    Err(_) => { /* Invalid signature */ },
}
```

### Address Conversion

```rust
// Convert ECDSA public key to Ethereum address
let compressed_pubkey = [0u8; 33]; // Compressed ECDSA public key
let mut eth_address = [0u8; 20];
api::ecdsa_to_eth_address(&compressed_pubkey, &mut eth_address);
```

## Contract Deployment

### Deploying New Contracts (CREATE1)

```rust
// Deploy without salt (address based on deployer + nonce)
let mut new_address = [0u8; 20];
api::instantiate(
    u64::MAX,               // ref_time limit
    u64::MAX,               // proof_size limit
    &[u8::MAX; 32],        // deposit limit
    &value,                 // Value to transfer
    &code_hash,            // Code hash [u8; 32]
    &constructor_input,     // Constructor arguments
    Some(&mut new_address), // Output for new address
    None,                   // No salt (CREATE1)
)?;
```

### Deploying with Deterministic Address (CREATE2)

```rust
// Deploy with salt (deterministic address)
let salt = [42u8; 32];
let mut new_address = [0u8; 20];
api::instantiate(
    u64::MAX,
    u64::MAX,
    &[u8::MAX; 32],
    &value,
    &code_hash,
    &constructor_input,
    Some(&mut new_address),
    Some(&salt),            // Salt for CREATE2
)?;
```

## Advanced Features

### Events

```rust
// Emit an event
let topics = [
    [0x11; 32],  // Event signature
    [0x22; 32],  // Indexed parameter 1
    [0x33; 32],  // Indexed parameter 2
];
let data = b"Event data";
api::deposit_event(&topics, data);
```

### Immutable Data

Set permanent data during deployment:

```rust
// In deploy() function
let config_data = b"v1.0.0";
api::set_immutable_data(config_data);

// In call() function
let mut buffer = [0u8; 6];
api::get_immutable_data(&mut buffer);
```

### Chain Extensions

Call custom runtime functions:

```rust
let function_id = 1001u32;
let input = b"custom data";
let mut output = [0u8; 32];
let result = api::call_chain_extension(function_id, input, Some(&mut output));
```

### Self Destruct

```rust
// Terminate contract and send remaining balance
let beneficiary = [0u8; 20];
api::terminate(&beneficiary);
```

## Compilation Process

### Build Script

Create `build.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Build the contract
cargo build --release --target riscv64emac-unknown-none-polkavm

# Link to PolkaVM format
polkatool link --strip \
    --output my_contract.polkavm \
    target/riscv64emac-unknown-none-polkavm/release/my_contract
```

### Cargo.toml Configuration

```toml
[package]
name = "my-polkavm-contract"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "my_contract"
path = "src/lib.rs"

[dependencies]
polkavm-derive = { version = "0.17.0" }
simplealloc = { version = "0.0.1", git = "https://github.com/paritytech/polkavm.git" }

[dependencies.uapi]
package = "pallet-revive-uapi"
git = "https://github.com/paritytech/polkadot-sdk.git"
default-features = false
features = ["unstable-hostfn"]

[profile.release]
opt-level = "s"      # Optimize for size
lto = "fat"          # Enable LTO
codegen-units = 1    # Single codegen unit
```

## Best Practices

### 1. Error Handling
- Always handle storage operation failures
- Check return codes from contract calls
- Use panic sparingly (it reverts the transaction)

### 2. Gas Management
- Be mindful of gas consumption in loops
- Forward appropriate gas to called contracts
- Check `ref_time_left()` before expensive operations

### 3. Security
- Validate all external inputs
- Use `CallFlags::ALLOW_REENTRY` carefully
- Implement checks-effects-interactions pattern
- Never trust external contract return data without validation

### 4. Storage Optimization
- Use transient storage for temporary data
- Pack struct fields to minimize storage slots
- Clear storage when no longer needed
- Use the optimized 32-byte functions when possible

### 5. Code Organization
- Keep contracts focused and modular
- Use clear function selectors
- Document your ABI
- Implement standard interfaces (e.g., ERC-20) correctly

## Example Contracts

### Minimal Contract

```rust
#![no_std]
#![no_main]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe {
        core::arch::asm!("unimp");
        core::hint::unreachable_unchecked();
    }
}

use uapi::{HostFn, HostFnImpl as api};

#[no_mangle]
#[polkavm_derive::polkavm_export]
pub extern "C" fn deploy() {
    // Empty constructor
}

#[no_mangle]
#[polkavm_derive::polkavm_export]
pub extern "C" fn call() {
    // Echo call data back
    let size = api::call_data_size();
    let mut buffer = vec![0u8; size];
    api::call_data_copy(&mut buffer, 0);
    api::return_value(uapi::ReturnFlags::empty(), &buffer);
}
```

### Simple Storage Contract

```rust
#![no_std]
#![no_main]

include!("../panic_handler.rs")

use uapi::{HostFn, HostFnImpl as api, StorageFlags};

const STORAGE_KEY: [u8; 32] = [1u8; 32];

#[no_mangle]
#[polkavm_derive::polkavm_export]
pub extern "C" fn deploy() {}

#[no_mangle]
#[polkavm_derive::polkavm_export]
pub extern "C" fn call() {
    input!(
        selector: &[u8; 4],
    );
    
    match u32::from_le_bytes(*selector) {
        0x60fe47b1 => set(),    // set(uint256)
        0x6d4ce63c => get(),    // get()
        _ => panic!("Unknown function"),
    }
}

fn set() {
    input!(
        _selector: &[u8; 4],
        value: &[u8; 32],
    );
    
    api::set_storage(StorageFlags::empty(), &STORAGE_KEY, value);
}

fn get() {
    let mut value = [0u8; 32];
    match api::get_storage(StorageFlags::empty(), &STORAGE_KEY, &mut value) {
        Ok(_) => api::return_value(uapi::ReturnFlags::empty(), &value),
        Err(_) => api::return_value(uapi::ReturnFlags::empty(), &[0u8; 32]),
    }
}
```

### Token Contract (ERC-20 Style)

```rust
#![no_std]
#![no_main]

include!("../panic_handler.rs")

use uapi::{HostFn, HostFnImpl as api, StorageFlags};

// Storage keys
const TOTAL_SUPPLY_KEY: [u8; 32] = [0u8; 32];

fn balance_key(owner: &[u8; 20]) -> [u8; 32] {
    let mut key = [1u8; 32];
    key[..20].copy_from_slice(owner);
    key
}

fn allowance_key(owner: &[u8; 20], spender: &[u8; 20]) -> [u8; 32] {
    let mut key = [2u8; 32];
    key[..20].copy_from_slice(owner);
    key[20..40].copy_from_slice(spender);
    api::hash_keccak_256(&key[..40], &mut key);
    key
}

#[no_mangle]
#[polkavm_derive::polkavm_export]
pub extern "C" fn deploy() {
    input!(
        initial_supply: &[u8; 32],
        owner: &[u8; 20],
    );
    
    // Set total supply
    api::set_storage(StorageFlags::empty(), &TOTAL_SUPPLY_KEY, initial_supply);
    
    // Give initial supply to owner
    let owner_key = balance_key(owner);
    api::set_storage(StorageFlags::empty(), &owner_key, initial_supply);
}

#[no_mangle]
#[polkavm_derive::polkavm_export]
pub extern "C" fn call() {
    input!(selector: &[u8; 4]);
    
    match u32::from_le_bytes(*selector) {
        0x70a08231 => balance_of(),     // balanceOf(address)
        0xa9059cbb => transfer(),        // transfer(address,uint256)
        0x23b872dd => transfer_from(),  // transferFrom(address,address,uint256)
        0xdd62ed3e => allowance(),      // allowance(address,address)
        0x095ea7b3 => approve(),        // approve(address,uint256)
        0x18160ddd => total_supply(),   // totalSupply()
        _ => panic!("Unknown function"),
    }
}

fn balance_of() {
    input!(
        _selector: &[u8; 4],
        owner: &[u8; 20],
    );
    
    let mut balance = [0u8; 32];
    api::get_storage_or_zero(StorageFlags::empty(), &balance_key(owner), &mut balance);
    api::return_value(uapi::ReturnFlags::empty(), &balance);
}

fn transfer() {
    input!(
        _selector: &[u8; 4],
        to: &[u8; 20],
        amount: &[u8; 32],
    );
    
    // Get caller
    let mut from = [0u8; 20];
    api::caller(&mut from);
    
    // Transfer logic would go here...
    // (Implementation omitted for brevity)
}

// Other functions would be implemented similarly...
```

## Conclusion

Writing smart contracts for PolkaVM in Rust provides a powerful and efficient way to build decentralized applications. The combination of Rust's safety guarantees, PolkaVM's performance, and the rich set of host functions makes it an excellent choice for blockchain development.

Key takeaways:
- PolkaVM contracts are `no_std` Rust programs with specific entry points
- The UAPI provides comprehensive blockchain interaction capabilities
- Storage, calls, and cryptographic operations are first-class features
- The compilation process produces optimized RISC-V bytecode
- Following best practices ensures secure and efficient contracts

For more examples and advanced patterns, explore the fixture contracts in the polkadot-sdk repository.