# PolkaVM Rust Smart Contracts: Comprehensive Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Key Differences from EVM](#key-differences-from-evm)
3. [Project Setup](#project-setup)
4. [Contract Structure](#contract-structure)
5. [Storage Patterns](#storage-patterns)
6. [Payable Functions](#payable-functions)
7. [Function Selectors](#function-selectors)
8. [Memory Management](#memory-management)
9. [Gas and Limits](#gas-and-limits)
10. [Deployment and Testing](#deployment-and-testing)
11. [Common Pitfalls](#common-pitfalls)
12. [Best Practices](#best-practices)
13. [Debugging Tips](#debugging-tips)
14. [Working Examples](#working-examples)

## Introduction

PolkaVM is Polkadot's RISC-V-based virtual machine that provides an alternative to the EVM for smart contract execution. The `pallet-revive` provides Ethereum compatibility, allowing contracts to be accessed via Ethereum RPC while running on PolkaVM.

### Key Technologies
- **PolkaVM**: RISC-V based virtual machine
- **pallet-revive**: Ethereum compatibility layer
- **uapi**: Host function interface for contracts
- **polkavm-derive**: Macro for exporting functions

## Key Differences from EVM

### 1. No Dynamic Memory by Default
```rust
// ❌ Avoid
let data = vec![]; // Dynamic allocation

// ✅ Prefer
let data = [0u8; 100]; // Fixed-size array
```

### 2. Payable by Default
```rust
// In Solidity: function foo() payable { }
// In PolkaVM: All functions can receive value unless explicitly rejected
```

### 3. Value Handling
```rust
// ❌ Wrong - Big-endian
let value = U256::from_big_endian(&value_bytes);

// ✅ Correct - Little-endian with macro
let value = u64_output!(api::value_transferred,);
```

### 4. Gas Limits
- PolkaVM gas limits are much higher than EVM
- Don't specify gas limits in deployment scripts
- Let the network estimate gas automatically

## Project Setup

### Cargo.toml
```toml
[package]
name = "my-polkavm-contract"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "my_contract"
path = "src/my_contract.rs"

[dependencies]
polkavm-derive = { version = "0.25.0" }
simplealloc = { version = "0.0.1", git = "https://github.com/paritytech/polkavm.git" }
ethabi = { version = "18.0", default-features = false }

[dependencies.uapi]
package = "pallet-revive-uapi"
git = "https://github.com/paritytech/polkadot-sdk.git"
default-features = false
features = ["unstable-hostfn"]

[profile.release]
opt-level = "s"
lto = "fat"
codegen-units = 1
```

### Build Script
```bash
#!/bin/bash
cargo build --release --target riscv64emac-unknown-none-polkavm --bin $1
polkatool link --strip --output $1.polkavm target/riscv64emac-unknown-none-polkavm/release/$1
```

## Contract Structure

### Basic Template
```rust
#![no_main]
#![no_std]
extern crate alloc;

use simplealloc::SimpleAlloc;

#[global_allocator]
static GLOBAL_ALLOCATOR: SimpleAlloc<{ 1024 * 50 }> = SimpleAlloc::new();

use uapi::{HostFn, HostFnImpl as api, StorageFlags, ReturnFlags, u64_output};
use ethabi::{decode, encode, Token, ParamType, ethereum_types::{U256, H160}};

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe {
        core::arch::asm!("unimp");
        core::hint::unreachable_unchecked();
    }
}

#[no_mangle]
#[polkavm_derive::polkavm_export]
pub extern "C" fn deploy() {
    // Constructor logic here
}

#[no_mangle]
#[polkavm_derive::polkavm_export]
pub extern "C" fn call() {
    // Runtime logic here
}
```

## Storage Patterns

### Storage Keys
```rust
// Simple keys
const KEY_OWNER: &[u8] = b"owner";
const KEY_BALANCE: &[u8] = b"balance";

// Prefixed keys for mappings
const PREFIX_USER: &[u8] = b"user_";
```

### Reading Storage
```rust
// For simple values
let mut buffer = [0u8; 32];
let _ = api::get_storage(StorageFlags::empty(), KEY_BALANCE, &mut &mut buffer[..]);

// For complex types with ethabi
let tokens = decode(&[ParamType::Uint(256)], &buffer).unwrap();
if let [Token::Uint(value)] = &tokens[..] {
    let balance = value.as_u64();
}
```

### Writing Storage
```rust
// Simple value
api::set_storage(StorageFlags::empty(), KEY_INITIALIZED, &[1u8]);

// Complex value with ethabi
let data = encode(&[Token::Uint(U256::from(amount))]);
api::set_storage(StorageFlags::empty(), KEY_BALANCE, &data);
```

### Fixed-Size Storage Keys
```rust
// For mappings, use fixed-size keys
fn make_user_key(user: &[u8; 20]) -> [u8; 24] {
    let mut key = [0u8; 24];
    key[..4].copy_from_slice(PREFIX_USER);
    key[4..24].copy_from_slice(user);
    key
}
```

## Payable Functions

### Reading Value Transferred
```rust
// Correct pattern using u64_output! macro
let value = u64_output!(api::value_transferred,);

if value == 0 {
    panic!("No value sent");
}
```

### Manual Value Reading (if needed)
```rust
let mut value_bytes = [0u8; 32];
api::value_transferred(&mut value_bytes);
// Values are little-endian, first 8 bytes for u64
let value = u64::from_le_bytes(value_bytes[..8].try_into().unwrap());
```

### Making Functions Non-Payable
```rust
let value = u64_output!(api::value_transferred,);
if value > 0 {
    panic!("This function is not payable");
}
```

### Fallback Function
```rust
#[no_mangle]
#[polkavm_derive::polkavm_export]
pub extern "C" fn call() {
    let length = api::call_data_size() as usize;
    
    // Handle plain value transfers (fallback)
    if length == 0 {
        api::return_value(ReturnFlags::empty(), &[]);
        return;
    }
    
    // Handle function calls
    // ...
}
```

## Function Selectors

### Calculating Selectors
```bash
# Using ethers.js
node -e "console.log(require('ethers').id('transfer(address,uint256)').slice(0,10))"
```

### Selector Matching
```rust
const TRANSFER_SELECTOR: [u8; 4] = [0xa9, 0x05, 0x9c, 0xbb];
const BALANCE_OF_SELECTOR: [u8; 4] = [0x70, 0xa0, 0x82, 0x31];

let mut selector = [0u8; 4];
api::call_data_copy(&mut selector, 0);

match selector {
    TRANSFER_SELECTOR => handle_transfer(),
    BALANCE_OF_SELECTOR => handle_balance_of(),
    _ => {
        // Unknown selector - act as fallback
        api::return_value(ReturnFlags::empty(), &[]);
    }
}
```

## Memory Management

### Avoid Dynamic Allocations
```rust
// ❌ Avoid in hot paths
use alloc::vec::Vec;
let mut data = Vec::new();

// ✅ Prefer fixed-size
let mut data = [0u8; 1000];
```

### Buffer Sizes
```rust
// Common buffer sizes
const MAX_INPUT: usize = 1000;
const MAX_STORAGE_VALUE: usize = 256;
```

### SimpleAlloc Configuration
```rust
// Adjust based on contract needs
#[global_allocator]
static GLOBAL_ALLOCATOR: SimpleAlloc<{ 1024 * 50 }> = SimpleAlloc::new();
```

## Gas and Limits

### Key Points
1. PolkaVM gas is different from EVM gas
2. Gas limits are much higher (100M+ is common)
3. Don't specify gas limits in transactions
4. Let the RPC estimate gas

### Example Transaction
```typescript
// ❌ Don't specify gas limit
const tx = await contract.method({ gasLimit: 1000000 });

// ✅ Let it auto-estimate
const tx = await contract.method();
```

## Deployment and Testing

### Building Contracts
```bash
# Build the contract
cargo build --release --target riscv64emac-unknown-none-polkavm --bin my_contract

# Link it
polkatool link --strip --output my_contract.polkavm \
    target/riscv64emac-unknown-none-polkavm/release/my_contract
```

### Deployment Script (TypeScript)
```typescript
import { ethers } from 'ethers';
import * as fs from 'fs';

const RPC_URL = 'https://westend-asset-hub-eth-rpc.polkadot.io';

async function deploy() {
    const provider = new ethers.JsonRpcProvider(RPC_URL);
    const wallet = new ethers.Wallet(privateKey, provider);
    
    const bytecode = '0x' + fs.readFileSync('contract.polkavm').toString('hex');
    const factory = new ethers.ContractFactory(ABI, bytecode, wallet);
    
    // Deploy without gas limit
    const contract = await factory.deploy();
    await contract.waitForDeployment();
    
    console.log('Deployed to:', await contract.getAddress());
}
```

## Common Pitfalls

### 1. Wrong Endianness
```rust
// ❌ Wrong
U256::from_big_endian(&bytes)

// ✅ Correct
U256::from_little_endian(&bytes)
```

### 2. Dynamic Arrays in Parameters
```rust
// ❌ Problematic
fn transfer_batch(recipients: Vec<Address>, amounts: Vec<U256>)

// ✅ Better
fn transfer_batch(recipient: Address, amount: U256)
```

### 3. Incorrect Value Reading
```rust
// ❌ Manual parsing prone to errors
let mut value_bytes = [0u8; 32];
api::value_transferred(&mut value_bytes);
let value = U256::from_big_endian(&value_bytes);

// ✅ Use the macro
let value = u64_output!(api::value_transferred,);
```

### 4. Panic on Unknown Selectors
```rust
// ❌ Will reject valid transfers
match selector {
    KNOWN_SELECTOR => handle(),
    _ => panic!("Unknown selector")
}

// ✅ Accept as fallback
match selector {
    KNOWN_SELECTOR => handle(),
    _ => api::return_value(ReturnFlags::empty(), &[])
}
```

## Best Practices

### 1. Use Fixed-Size Arrays
```rust
// Storage keys
const KEY_BALANCE: &[u8; 7] = b"balance";

// Buffers
let mut buffer = [0u8; 100];
```

### 2. Handle Errors Gracefully
```rust
// Check storage results
if api::get_storage(StorageFlags::empty(), key, &mut buffer).is_err() {
    // Handle missing value
}
```

### 3. Validate Input Lengths
```rust
let length = api::call_data_size() as usize;
if length < 68 {
    panic!("Invalid input length");
}
```

### 4. Use Helper Functions
```rust
fn decode_u256_from_storage(bytes: &[u8]) -> u64 {
    if bytes.is_empty() {
        return 0;
    }
    // Decode logic
}
```

## Debugging Tips

### 1. Check Transaction Errors
```typescript
try {
    const tx = await contract.method();
    await tx.wait();
} catch (error) {
    console.log('Error info:', error.info);
    // Look for "ContractTrapped" messages
}
```

### 2. Use Blockscout
- Westend Asset Hub: https://blockscout-asset-hub.parity-chains-scw.parity.io
- Check contract balance and transactions
- Verify deployment success

### 3. Common Error Messages
- `"ContractTrapped"` - Contract panicked
- `"Invalid Transaction"` - Often gas or nonce issues
- `"Module error [11, 0, 0, 0]"` - Contract execution failed

### 4. Test Incrementally
1. Deploy empty contract
2. Add initialize function
3. Add view functions
4. Add payable functions
5. Add complex logic

## Working Examples

### Simple Token Transfer
```rust
fn handle_transfer(length: usize) {
    if length < 68 {
        panic!("Invalid input");
    }
    
    let mut buffer = [0u8; 68];
    api::call_data_copy(&mut buffer, 0);
    
    let tokens = decode(&[
        ParamType::Address,
        ParamType::Uint(256)
    ], &buffer[4..]).unwrap();
    
    if let [Token::Address(to), Token::Uint(amount)] = &tokens[..] {
        // Transfer logic
        api::return_value(ReturnFlags::empty(), &[1u8]); // success
    }
}
```

### Reading Contract Balance
```rust
fn handle_get_balance() {
    let balance = u64_output!(api::balance,);
    let result = encode(&[Token::Uint(U256::from(balance))]);
    api::return_value(ReturnFlags::empty(), &result);
}
```

### Complete Payable Function
```rust
fn handle_deposit() {
    // Get value sent
    let value = u64_output!(api::value_transferred,);
    
    if value == 0 {
        panic!("No value sent");
    }
    
    // Get caller
    let mut caller = [0u8; 20];
    api::caller(&mut caller);
    
    // Update balance
    let key = make_balance_key(&caller);
    let mut current = [0u8; 32];
    let _ = api::get_storage(StorageFlags::empty(), &key, &mut &mut current[..]);
    
    let balance = decode_u256_from_storage(&current);
    let new_balance = balance + value;
    
    let data = encode(&[Token::Uint(U256::from(new_balance))]);
    api::set_storage(StorageFlags::empty(), &key, &data);
    
    api::return_value(ReturnFlags::empty(), &[]);
}
```

## Resources

- [Polkadot SDK Repository](https://github.com/paritytech/polkadot-sdk)
- [Revive Fixtures](https://github.com/paritytech/polkadot-sdk/tree/master/substrate/frame/revive/fixtures/contracts)
- [Westend Asset Hub RPC](https://westend-asset-hub-eth-rpc.polkadot.io)
- [Blockscout Explorer](https://blockscout-asset-hub.parity-chains-scw.parity.io)

## Summary

PolkaVM provides a powerful alternative to EVM with better performance characteristics. Key success factors:

1. **Use fixed-size arrays** instead of dynamic allocation
2. **Follow the u64_output! pattern** for reading values
3. **Handle unknown selectors gracefully** for fallback behavior
4. **Don't specify gas limits** - let the network estimate
5. **Use little-endian encoding** for values
6. **Test incrementally** to isolate issues

With these patterns, you can build efficient and functional smart contracts on PolkaVM that are accessible via Ethereum tooling.
