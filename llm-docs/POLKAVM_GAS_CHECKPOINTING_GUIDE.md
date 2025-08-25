# Gas Checkpointing in PolkaVM Contracts

Gas checkpointing is a technique for monitoring gas consumption at critical points in your contract to prevent unexpected out-of-gas failures. This guide provides comprehensive patterns and best practices for implementing gas management in PolkaVM smart contracts.

## Basic Gas Functions

```rust
// Get remaining computational time (gas)
let gas_left: u64 = api::ref_time_left();

// Get current gas price
let mut price = [0u8; 32];
api::gas_price(&mut price);

// Get block gas limit
let block_limit: u64 = api::gas_limit();
```

## Simple Checkpointing Pattern

```rust
fn complex_operation() {
    // Checkpoint 1: Record initial gas
    let initial_gas = api::ref_time_left();
    
    // Do some work...
    expensive_calculation();
    
    // Checkpoint 2: Check if we've used too much
    let gas_used = initial_gas - api::ref_time_left();
    if gas_used > MAX_ALLOWED_FOR_OPERATION {
        api::return_value(ReturnFlags::REVERT, b"GAS_LIMIT");
    }
    
    // Continue with more work...
}
```

## Advanced Checkpointing Strategies

### 1. Multi-Stage Operations

```rust
fn multi_stage_function() {
    let checkpoint_1 = api::ref_time_left();
    
    // Stage 1: Input validation (cheap)
    validate_inputs();
    
    // Check we have enough gas for expensive operations
    let stage1_cost = checkpoint_1 - api::ref_time_left();
    let estimated_remaining_cost = stage1_cost * 10; // Estimate 10x more
    
    if api::ref_time_left() < estimated_remaining_cost {
        api::return_value(ReturnFlags::REVERT, b"INSUFFICIENT_GAS");
    }
    
    // Stage 2: Storage operations (expensive)
    let checkpoint_2 = api::ref_time_left();
    perform_storage_operations();
    
    // Stage 3: External calls (very expensive)
    let checkpoint_3 = api::ref_time_left();
    if checkpoint_3 < MIN_GAS_FOR_EXTERNAL_CALL {
        api::return_value(ReturnFlags::REVERT, b"INSUFFICIENT_GAS_FOR_CALL");
    }
    make_external_calls();
}
```

### 2. Loop Gas Management

```rust
fn process_array(items: Vec<Item>) {
    const GAS_PER_ITEM: u64 = 50_000;
    let initial_gas = api::ref_time_left();
    
    for (index, item) in items.iter().enumerate() {
        // Check gas every N iterations
        if index % 10 == 0 {
            let gas_used = initial_gas - api::ref_time_left();
            let gas_per_item_actual = gas_used / (index as u64 + 1);
            let remaining_items = items.len() - index;
            
            // Predict if we have enough gas to finish
            if api::ref_time_left() < gas_per_item_actual * remaining_items as u64 {
                // Save progress and exit gracefully
                save_checkpoint(index);
                api::return_value(ReturnFlags::empty(), &index.to_le_bytes());
            }
        }
        
        process_item(item);
    }
}
```

### 3. Gas Reservation Pattern

```rust
fn operation_with_cleanup() {
    // Reserve gas for cleanup
    const CLEANUP_GAS_RESERVE: u64 = 100_000;
    
    let available_gas = api::ref_time_left();
    if available_gas < CLEANUP_GAS_RESERVE * 2 {
        panic!("Insufficient gas");
    }
    
    let usable_gas = available_gas - CLEANUP_GAS_RESERVE;
    
    // Do main operation
    let result = perform_operation();
    
    // Check we still have reserve
    if api::ref_time_left() < CLEANUP_GAS_RESERVE {
        // Emergency: skip optional cleanup
        api::return_value(ReturnFlags::empty(), &result);
    }
    
    // Perform cleanup
    cleanup();
    api::return_value(ReturnFlags::empty(), &result);
}
```

## Real-World Example: AMM Trade with Gas Management

```rust
fn handle_trade_with_gas_management(market_id: u64, size: u64, max_cost: u64) {
    // Define gas requirements for each phase
    const GAS_VALIDATION: u64 = 50_000;
    const GAS_LOAD_MARKET: u64 = 100_000;
    const GAS_CALCULATIONS: u64 = 200_000;
    const GAS_STORAGE_UPDATE: u64 = 150_000;
    const GAS_TRANSFER: u64 = 100_000;
    const GAS_BUFFER: u64 = 50_000;
    
    const TOTAL_GAS_NEEDED: u64 = GAS_VALIDATION + GAS_LOAD_MARKET + 
                                   GAS_CALCULATIONS + GAS_STORAGE_UPDATE + 
                                   GAS_TRANSFER + GAS_BUFFER;
    
    // Check total gas upfront
    let initial_gas = api::ref_time_left();
    if initial_gas < TOTAL_GAS_NEEDED {
        api::return_value(ReturnFlags::REVERT, b"INSUFFICIENT_GAS");
    }
    
    // Phase 1: Validation
    let checkpoint_1 = api::ref_time_left();
    validate_trade_params(market_id, size, max_cost);
    
    // Phase 2: Load market data
    if api::ref_time_left() < (TOTAL_GAS_NEEDED - GAS_VALIDATION) {
        api::return_value(ReturnFlags::REVERT, b"GAS_PHASE2");
    }
    let mut market = load_market(market_id).expect("Market not found");
    
    // Phase 3: Complex calculations
    let checkpoint_2 = api::ref_time_left();
    let (new_lambda, new_f_max, cost) = calculate_trade(
        &market,
        size,
        Direction::Long
    );
    
    // Verify calculations didn't use too much gas
    let calc_gas_used = checkpoint_2 - api::ref_time_left();
    if calc_gas_used > GAS_CALCULATIONS * 2 {
        // Calculations took too long, might indicate attack
        api::return_value(ReturnFlags::REVERT, b"GAS_ATTACK");
    }
    
    // Phase 4: State updates (critical section)
    if api::ref_time_left() < GAS_STORAGE_UPDATE + GAS_TRANSFER + GAS_BUFFER {
        api::return_value(ReturnFlags::REVERT, b"GAS_CRITICAL");
    }
    
    // Update market state
    market.current_lambda = new_lambda;
    market.current_f_max = new_f_max;
    save_market(market_id, &market);
    
    // Phase 5: Transfer funds
    transfer_funds(&caller, cost);
    
    // Return success
    api::return_value(ReturnFlags::empty(), &cost.to_le_bytes());
}
```

## Gas Checkpointing Best Practices

### 1. Fail Fast

```rust
fn expensive_function(data: &[u8]) {
    // Check gas BEFORE doing expensive decoding
    if api::ref_time_left() < MIN_GAS_REQUIRED {
        api::return_value(ReturnFlags::REVERT, b"GAS");
    }
    
    let decoded = expensive_decode(data); // Now safe to proceed
}
```

### 2. Progressive Degradation

```rust
fn get_market_data(market_id: u64, include_history: bool) {
    let mut result = load_basic_market_data(market_id);
    
    // Include optional data only if enough gas
    if include_history && api::ref_time_left() > HISTORY_GAS_COST {
        result.history = load_market_history(market_id);
    }
    
    return_encoded(result);
}
```

### 3. Gas Metering for External Calls

```rust
fn safe_external_call(target: &[u8; 20], value: &[u8; 32]) {
    let available_gas = api::ref_time_left();
    
    // Keep 10% for post-call operations
    let gas_to_forward = (available_gas * 90) / 100;
    
    let result = api::call(
        CallFlags::empty(),
        target,
        gas_to_forward,  // Explicit gas limit
        u64::MAX,        // Proof size
        &[u8::MAX; 32],
        value,
        &[],
        None,
    );
    
    // We still have 10% gas to handle the result
    handle_call_result(result);
}
```

## Common Gas Pitfalls to Avoid

### 1. Don't Check Too Often

```rust
// BAD: Checking gas in tight loop
for i in 0..1000 {
    if api::ref_time_left() < 1000 { // This itself costs gas!
        break;
    }
    simple_operation();
}

// GOOD: Check periodically
for i in 0..1000 {
    if i % 100 == 0 && api::ref_time_left() < 10000 {
        break;
    }
    simple_operation();
}
```

### 2. Don't Trust Gas Estimates

```rust
// BAD: Assuming operation costs
let estimated_cost = 100_000;
if api::ref_time_left() > estimated_cost {
    complex_operation(); // Might cost more!
}

// GOOD: Measure actual costs
let gas_before = api::ref_time_left();
complex_operation();
let actual_cost = gas_before - api::ref_time_left();
// Use actual_cost for future estimates
```

### 3. Handle Gas Exhaustion Gracefully

```rust
// BAD: Just panic
if api::ref_time_left() < needed {
    panic!("Out of gas");
}

// GOOD: Return meaningful error
if api::ref_time_left() < needed {
    let mut error_data = [0u8; 8];
    error_data[0] = 0x01; // Error code for gas
    error_data[1..5].copy_from_slice(&needed.to_le_bytes()[..4]);
    api::return_value(ReturnFlags::REVERT, &error_data);
}
```

## Gas Monitoring Utilities

```rust
// Utility struct for gas profiling
struct GasProfiler {
    checkpoints: Vec<(String, u64)>,
}

impl GasProfiler {
    fn new() -> Self {
        Self {
            checkpoints: Vec::new(),
        }
    }
    
    fn checkpoint(&mut self, label: &str) {
        self.checkpoints.push((
            label.to_string(),
            api::ref_time_left()
        ));
    }
    
    fn report(&self) -> Vec<(String, u64)> {
        let mut costs = Vec::new();
        for i in 1..self.checkpoints.len() {
            let gas_used = self.checkpoints[i-1].1 - self.checkpoints[i].1;
            costs.push((
                self.checkpoints[i].0.clone(),
                gas_used
            ));
        }
        costs
    }
}

// Usage
fn profile_operation() {
    let mut profiler = GasProfiler::new();
    profiler.checkpoint("start");
    
    load_data();
    profiler.checkpoint("after_load");
    
    process_data();
    profiler.checkpoint("after_process");
    
    save_results();
    profiler.checkpoint("after_save");
    
    // Emit gas usage report
    let report = profiler.report();
    // Log or return report
}
```

## Gas Requirements by Operation Type

Here's a rough guide for gas requirements in PolkaVM:

| Operation | Approximate Gas Cost |
|-----------|---------------------|
| Storage Read (32 bytes) | 50,000 - 100,000 |
| Storage Write (32 bytes) | 100,000 - 200,000 |
| Storage Clear | 50,000 - 100,000 |
| External Call (base) | 100,000 + forwarded |
| Hash Calculation | 10,000 - 50,000 |
| Event Emission | 20,000 - 100,000 |
| Basic Arithmetic | 100 - 1,000 |
| Memory Copy (per KB) | 5,000 - 10,000 |

Note: These are estimates and actual costs may vary based on the specific chain configuration.

## Summary

Gas checkpointing in PolkaVM should be:

- **Strategic**: Place checkpoints before expensive operations
- **Predictive**: Estimate remaining costs and fail early
- **Graceful**: Provide meaningful errors when gas runs low
- **Efficient**: Don't check too frequently as checks cost gas too
- **Defensive**: Always reserve gas for critical cleanup operations

Remember: Gas management is especially important in PolkaVM because the two-tier system (Weight + Engine fuel) makes gas consumption less predictable than in simpler VMs. Always test your contracts with various gas limits to ensure they handle edge cases gracefully.