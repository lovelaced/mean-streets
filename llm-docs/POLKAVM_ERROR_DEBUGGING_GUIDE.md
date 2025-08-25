# PolkaVM Smart Contract Error Handling and Debugging Guide

A comprehensive guide for implementing robust error handling, debugging strategies, and panic management in PolkaVM smart contracts.

## Table of Contents
1. [Panic Handler Implementation](#panic-handler-implementation)
2. [Error Handling Strategies](#error-handling-strategies)
3. [Debugging Techniques](#debugging-techniques)
4. [Error Code Systems](#error-code-systems)
5. [Testing and Validation](#testing-and-validation)
6. [Production vs Development Patterns](#production-vs-development-patterns)
7. [Common Error Scenarios](#common-error-scenarios)

## Panic Handler Implementation

### Basic Panic Handler (Production)

```rust
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // The unimp instruction causes the VM to trap and revert
    unsafe {
        core::arch::asm!("unimp");
        core::hint::unreachable_unchecked();
    }
}
```

### Debug-Enabled Panic Handler

```rust
#[cfg(feature = "debug")]
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    // Try to emit debug information before panicking
    unsafe {
        // Emit panic location if available
        if let Some(location) = info.location() {
            let line = location.line();
            let topics = [[0xFF; 32]]; // Debug panic topic
            let mut data = [0u8; 8];
            data[0..4].copy_from_slice(&line.to_le_bytes());
            data[4] = b'P'; // Panic marker
            
            // Best effort - ignore result
            let _ = api::deposit_event(&topics, &data);
        }
        
        core::arch::asm!("unimp");
        core::hint::unreachable_unchecked();
    }
}
```

### Shared Panic Handler Pattern

Create a `panic_handler.rs` file:
```rust
// panic_handler.rs
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe {
        core::arch::asm!("unimp");
        core::hint::unreachable_unchecked();
    }
}
```

Include in your contracts:
```rust
#![no_std]
#![no_main]

include!("../panic_handler.rs")
```

## Error Handling Strategies

### 1. Result-Based Error Handling

```rust
#[derive(Debug, Clone, Copy)]
enum ContractError {
    InvalidInput = 1,
    InsufficientBalance = 2,
    Unauthorized = 3,
    MarketNotFound = 4,
    ExceedsLimit = 5,
    TransferFailed = 6,
    StorageError = 7,
    MathOverflow = 8,
}

impl ContractError {
    fn to_bytes(self) -> [u8; 4] {
        (self as u32).to_le_bytes()
    }
}

// Use in functions
fn transfer_tokens(to: &[u8; 20], amount: u64) -> Result<(), ContractError> {
    let balance = get_balance(&caller)?;
    
    if balance < amount {
        return Err(ContractError::InsufficientBalance);
    }
    
    // Perform transfer
    Ok(())
}

// Handle results in main entry point
pub extern "C" fn call() {
    match transfer_tokens(&recipient, amount) {
        Ok(()) => {
            api::return_value(ReturnFlags::empty(), &[0u8; 1]);
        },
        Err(e) => {
            api::return_value(ReturnFlags::REVERT, &e.to_bytes());
        }
    }
}
```

### 2. Panic vs Revert Strategy

```rust
// Use panic for unrecoverable errors (bugs)
fn critical_invariant_check() {
    let total = calculate_total();
    let sum = sum_all_parts();
    
    // This should NEVER happen - indicates a bug
    if total != sum {
        panic!("Critical invariant violated");
    }
}

// Use revert for expected errors
fn user_operation() {
    if amount > MAX_ALLOWED {
        // User error - return meaningful error
        let mut error_data = [0u8; 36];
        error_data[0..4].copy_from_slice(b"E001");
        error_data[4..12].copy_from_slice(&amount.to_le_bytes());
        error_data[12..20].copy_from_slice(&MAX_ALLOWED.to_le_bytes());
        api::return_value(ReturnFlags::REVERT, &error_data);
    }
}
```

### 3. Error Context Pattern

```rust
struct ErrorContext {
    code: u32,
    function: [u8; 4],
    param1: u64,
    param2: u64,
}

impl ErrorContext {
    fn new(code: u32, function: &[u8; 4]) -> Self {
        Self {
            code,
            function: *function,
            param1: 0,
            param2: 0,
        }
    }
    
    fn with_params(mut self, p1: u64, p2: u64) -> Self {
        self.param1 = p1;
        self.param2 = p2;
        self
    }
    
    fn to_bytes(&self) -> [u8; 28] {
        let mut bytes = [0u8; 28];
        bytes[0..4].copy_from_slice(&self.code.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.function);
        bytes[8..16].copy_from_slice(&self.param1.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.param2.to_le_bytes());
        bytes
    }
}

// Usage
fn handle_trade() {
    let context = ErrorContext::new(100, b"TRAD");
    
    if size > max_size {
        let error = context.with_params(size, max_size);
        api::return_value(ReturnFlags::REVERT, &error.to_bytes());
    }
}
```

## Debugging Techniques

### 1. Event-Based Debugging

```rust
// Debug event emitter
struct DebugLogger;

impl DebugLogger {
    const DEBUG_TOPIC: [u8; 32] = [0xDE; 32];
    
    fn log_value(label: &[u8; 4], value: u64) {
        let mut data = [0u8; 12];
        data[0..4].copy_from_slice(label);
        data[4..12].copy_from_slice(&value.to_le_bytes());
        api::deposit_event(&[Self::DEBUG_TOPIC], &data);
    }
    
    fn log_checkpoint(checkpoint_id: u8) {
        api::deposit_event(&[Self::DEBUG_TOPIC], &[checkpoint_id]);
    }
    
    fn log_storage_key(key: &[u8]) {
        let mut data = [0u8; 36];
        data[0..4].copy_from_slice(b"KEY:");
        data[4..36].copy_from_slice(&key[..32.min(key.len())]);
        api::deposit_event(&[Self::DEBUG_TOPIC], &data);
    }
}

// Usage in development
#[cfg(feature = "debug")]
fn complex_calculation(input: u64) -> u64 {
    DebugLogger::log_value(b"INPT", input);
    
    let step1 = input * 2;
    DebugLogger::log_value(b"STP1", step1);
    
    let step2 = step1 + 100;
    DebugLogger::log_value(b"STP2", step2);
    
    step2
}
```

### 2. State Inspection Functions

```rust
// Add debug-only inspection functions
#[cfg(feature = "debug")]
#[no_mangle]
#[polkavm_derive::polkavm_export]
pub extern "C" fn debug_get_state() {
    input!(
        query_type: u8,
        param: u64,
    );
    
    match query_type {
        1 => { // Get market state
            let market = load_market(param);
            let encoded = encode_market(&market);
            api::return_value(ReturnFlags::empty(), &encoded);
        },
        2 => { // Get user balance
            let mut address = [0u8; 20];
            address[..8].copy_from_slice(&param.to_le_bytes());
            let balance = get_balance(&address);
            api::return_value(ReturnFlags::empty(), &balance.to_le_bytes());
        },
        3 => { // Get storage raw
            let mut key = [0u8; 32];
            key[..8].copy_from_slice(&param.to_le_bytes());
            let mut value = [0u8; 256];
            match api::get_storage(StorageFlags::empty(), &key, &mut value) {
                Ok(size) => {
                    api::return_value(ReturnFlags::empty(), &value[..size]);
                },
                Err(_) => {
                    api::return_value(ReturnFlags::empty(), b"NOT_FOUND");
                }
            }
        },
        _ => {
            api::return_value(ReturnFlags::REVERT, b"UNKNOWN_QUERY");
        }
    }
}
```

### 3. Assertion Macros

```rust
// Custom assertion macros for contracts
macro_rules! contract_assert {
    ($condition:expr, $error_code:expr) => {
        if !$condition {
            #[cfg(feature = "debug")]
            {
                let line = line!();
                let mut data = [0u8; 8];
                data[0..4].copy_from_slice(&line.to_le_bytes());
                data[4..8].copy_from_slice(&$error_code.to_le_bytes());
                api::deposit_event(&[[0xAE; 32]], &data);
            }
            api::return_value(ReturnFlags::REVERT, &$error_code.to_le_bytes());
        }
    };
}

macro_rules! debug_assert_eq {
    ($left:expr, $right:expr) => {
        #[cfg(feature = "debug")]
        {
            if $left != $right {
                let mut data = [0u8; 16];
                data[0..8].copy_from_slice(&$left.to_le_bytes());
                data[8..16].copy_from_slice(&$right.to_le_bytes());
                api::deposit_event(&[[0xEE; 32]], &data);
                panic!("Assertion failed");
            }
        }
    };
}
```

## Error Code Systems

### 1. Structured Error Codes

```rust
// Error code structure: [Category:u8][Subcategory:u8][Specific:u16]
mod ErrorCodes {
    // Categories
    const AUTH: u8 = 0x01;
    const MARKET: u8 = 0x02;
    const MATH: u8 = 0x03;
    const STATE: u8 = 0x04;
    const TRANSFER: u8 = 0x05;
    
    // Auth errors
    pub const UNAUTHORIZED: u32 = (AUTH as u32) << 24 | 0x0001;
    pub const INVALID_SIGNATURE: u32 = (AUTH as u32) << 24 | 0x0002;
    
    // Market errors
    pub const MARKET_NOT_FOUND: u32 = (MARKET as u32) << 24 | 0x0001;
    pub const MARKET_CLOSED: u32 = (MARKET as u32) << 24 | 0x0002;
    pub const INVALID_PRICE: u32 = (MARKET as u32) << 24 | 0x0003;
    
    // Math errors
    pub const OVERFLOW: u32 = (MATH as u32) << 24 | 0x0001;
    pub const UNDERFLOW: u32 = (MATH as u32) << 24 | 0x0002;
    pub const DIVISION_BY_ZERO: u32 = (MATH as u32) << 24 | 0x0003;
}

// Helper to decode error category
fn decode_error_category(error: u32) -> &'static str {
    match (error >> 24) as u8 {
        0x01 => "AUTH",
        0x02 => "MARKET",
        0x03 => "MATH",
        0x04 => "STATE",
        0x05 => "TRANSFER",
        _ => "UNKNOWN",
    }
}
```

### 2. Error Registry Pattern

```rust
struct ErrorRegistry;

impl ErrorRegistry {
    // Store error details in events for debugging
    fn register_error(code: u32, details: &[u8]) {
        let mut data = [0u8; 64];
        data[0..4].copy_from_slice(&code.to_le_bytes());
        data[4..8].copy_from_slice(&api::block_number().to_le_bytes());
        data[8..16].copy_from_slice(&api::now().to_le_bytes());
        
        let detail_len = details.len().min(48);
        data[16..16 + detail_len].copy_from_slice(&details[..detail_len]);
        
        // Error event topic
        let topic = [[0xE4; 32]];
        api::deposit_event(&topic, &data);
    }
    
    fn revert_with_error(code: u32, details: &[u8]) {
        Self::register_error(code, details);
        
        let mut revert_data = [0u8; 36];
        revert_data[0..4].copy_from_slice(&code.to_le_bytes());
        revert_data[4..36].copy_from_slice(&details[..32.min(details.len())]);
        
        api::return_value(ReturnFlags::REVERT, &revert_data);
    }
}
```

## Testing and Validation

### 1. Test Helper Functions

```rust
#[cfg(feature = "test")]
mod test_helpers {
    use super::*;
    
    pub fn force_error(error_type: u8) {
        match error_type {
            1 => panic!("Forced panic"),
            2 => {
                // Force out of gas
                loop {}
            },
            3 => {
                // Force storage error
                let huge_key = [0xFF; 1000];
                api::get_storage(StorageFlags::empty(), &huge_key, &mut []);
            },
            _ => {}
        }
    }
    
    pub fn validate_state_consistency() {
        // Check critical invariants
        let total_supply = get_total_supply();
        let sum_balances = sum_all_balances();
        
        if total_supply != sum_balances {
            let mut error_data = [0u8; 16];
            error_data[0..8].copy_from_slice(&total_supply.to_le_bytes());
            error_data[8..16].copy_from_slice(&sum_balances.to_le_bytes());
            ErrorRegistry::revert_with_error(
                ErrorCodes::STATE_INCONSISTENT,
                &error_data
            );
        }
    }
}
```

### 2. Debugging Entry Points

```rust
// Special entry point for testing error conditions
#[cfg(feature = "test")]
#[no_mangle]
#[polkavm_derive::polkavm_export]
pub extern "C" fn test_error_handling() {
    input!(
        test_case: u8,
        param1: u64,
        param2: u64,
    );
    
    match test_case {
        1 => {
            // Test panic behavior
            if param1 > 100 {
                panic!("Test panic");
            }
        },
        2 => {
            // Test revert behavior
            ErrorRegistry::revert_with_error(
                ErrorCodes::TEST_ERROR,
                &param1.to_le_bytes()
            );
        },
        3 => {
            // Test gas exhaustion
            for i in 0..param1 {
                expensive_operation();
            }
        },
        4 => {
            // Test error propagation
            match risky_operation(param1, param2) {
                Ok(result) => {
                    api::return_value(ReturnFlags::empty(), &result.to_le_bytes());
                },
                Err(e) => {
                    ErrorRegistry::revert_with_error(e.code(), e.details());
                }
            }
        },
        _ => {
            api::return_value(ReturnFlags::REVERT, b"UNKNOWN_TEST");
        }
    }
}
```

## Production vs Development Patterns

### 1. Conditional Compilation

```rust
// Cargo.toml
[features]
default = []
debug = []
test = ["debug"]
production = []

// In contract code
#[cfg(not(feature = "production"))]
fn detailed_error(msg: &str) {
    let mut data = [0u8; 128];
    let len = msg.len().min(128);
    data[..len].copy_from_slice(msg.as_bytes());
    api::return_value(ReturnFlags::REVERT, &data[..len]);
}

#[cfg(feature = "production")]
fn detailed_error(_msg: &str) {
    // In production, use short error codes
    api::return_value(ReturnFlags::REVERT, b"E001");
}
```

### 2. Error Message Strategy

```rust
// Development: Descriptive errors
#[cfg(feature = "debug")]
const ERROR_MESSAGES: &[(&str, &str)] = &[
    ("E001", "Insufficient balance for operation"),
    ("E002", "Market not found with given ID"),
    ("E003", "Unauthorized: caller is not owner"),
    ("E004", "Math overflow in calculation"),
];

// Production: Minimal errors
#[cfg(not(feature = "debug"))]
fn handle_error(code: &str) {
    api::return_value(ReturnFlags::REVERT, code.as_bytes());
}

#[cfg(feature = "debug")]
fn handle_error(code: &str) {
    for (err_code, msg) in ERROR_MESSAGES {
        if *err_code == code {
            api::return_value(ReturnFlags::REVERT, msg.as_bytes());
        }
    }
    api::return_value(ReturnFlags::REVERT, code.as_bytes());
}
```

## Common Error Scenarios

### 1. Storage Errors

```rust
fn safe_storage_read(key: &[u8; 32]) -> Result<Vec<u8>, ContractError> {
    let mut buffer = [0u8; 1024]; // Max expected size
    
    match api::get_storage(StorageFlags::empty(), key, &mut buffer) {
        Ok(size) => {
            if size > buffer.len() {
                return Err(ContractError::StorageError);
            }
            Ok(buffer[..size].to_vec())
        },
        Err(_) => Err(ContractError::StorageError),
    }
}

fn safe_storage_write(key: &[u8; 32], value: &[u8]) -> Result<(), ContractError> {
    // Check size limits
    if value.len() > MAX_STORAGE_VALUE_SIZE {
        return Err(ContractError::ExceedsLimit);
    }
    
    // Check gas before write
    if api::ref_time_left() < MIN_GAS_FOR_STORAGE {
        return Err(ContractError::InsufficientGas);
    }
    
    api::set_storage(StorageFlags::empty(), key, value);
    Ok(())
}
```

### 2. External Call Errors

```rust
fn safe_external_call(
    target: &[u8; 20],
    value: &[u8; 32],
    data: &[u8]
) -> Result<Vec<u8>, ContractError> {
    let result = api::call(
        CallFlags::empty(),
        target,
        0, // forward all gas
        0,
        &[u8::MAX; 32],
        value,
        data,
        None,
    );
    
    match result {
        Ok(()) => {
            // Get return data
            let size = api::return_data_size();
            if size > MAX_RETURN_SIZE {
                return Err(ContractError::ReturnDataTooLarge);
            }
            
            let mut return_data = vec![0u8; size];
            api::return_data_copy(&mut return_data, 0);
            Ok(return_data)
        },
        Err(code) => {
            // Emit debug event with error details
            #[cfg(feature = "debug")]
            {
                let mut data = [0u8; 24];
                data[0..20].copy_from_slice(target);
                data[20..24].copy_from_slice(&(code as u32).to_le_bytes());
                api::deposit_event(&[[0xCA; 32]], &data);
            }
            
            match code {
                ReturnErrorCode::CalleeReverted => Err(ContractError::CallReverted),
                ReturnErrorCode::CalleeTrapped => Err(ContractError::CallTrapped),
                ReturnErrorCode::TransferFailed => Err(ContractError::TransferFailed),
                _ => Err(ContractError::UnknownCallError),
            }
        }
    }
}
```

### 3. Math Operation Errors

```rust
// Safe math operations
trait SafeMath {
    fn safe_add(&self, other: Self) -> Result<Self, ContractError> 
    where Self: Sized;
    
    fn safe_sub(&self, other: Self) -> Result<Self, ContractError>
    where Self: Sized;
    
    fn safe_mul(&self, other: Self) -> Result<Self, ContractError>
    where Self: Sized;
    
    fn safe_div(&self, other: Self) -> Result<Self, ContractError>
    where Self: Sized;
}

impl SafeMath for u64 {
    fn safe_add(&self, other: Self) -> Result<Self, ContractError> {
        self.checked_add(other).ok_or(ContractError::MathOverflow)
    }
    
    fn safe_sub(&self, other: Self) -> Result<Self, ContractError> {
        self.checked_sub(other).ok_or(ContractError::MathUnderflow)
    }
    
    fn safe_mul(&self, other: Self) -> Result<Self, ContractError> {
        self.checked_mul(other).ok_or(ContractError::MathOverflow)
    }
    
    fn safe_div(&self, other: Self) -> Result<Self, ContractError> {
        if other == 0 {
            return Err(ContractError::DivisionByZero);
        }
        Ok(self / other)
    }
}

// Usage
fn calculate_price(amount: u64, rate: u64) -> Result<u64, ContractError> {
    amount.safe_mul(rate)?.safe_div(PRICE_SCALE)
}
```

## Best Practices Summary

1. **Use panic sparingly** - Only for bugs and invariant violations
2. **Return meaningful errors** - Help users and developers understand issues
3. **Implement error codes** - Structured, gas-efficient error reporting
4. **Add debug features** - Conditional compilation for development aids
5. **Test error paths** - Ensure all error conditions are handled
6. **Document errors** - Maintain error code documentation
7. **Monitor gas** - Ensure error handling doesn't cause gas issues
8. **Log strategically** - Use events for debugging in development
9. **Validate inputs early** - Fail fast to save gas
10. **Handle all Results** - Never use `unwrap()` in production

Remember: Good error handling makes contracts more reliable, debuggable, and user-friendly. The investment in proper error handling pays off in reduced debugging time and better user experience.