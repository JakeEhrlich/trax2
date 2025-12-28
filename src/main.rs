mod types;
mod interp;
mod parser;

use types::*;
use interp::*;
use parser::*;
use std::collections::HashMap;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 {
        // Run specified .wast file
        let path = Path::new(&args[1]);
        run_wast_file(path);
    } else {
        // Run built-in tests
        run_builtin_tests();

        // Try to run i32.wast if it exists
        let testsuite_path = Path::new("testsuite/i32.wast");
        if testsuite_path.exists() {
            println!("\n--- Running i32.wast ---");
            run_wast_file(testsuite_path);
        }
    }
}

fn run_wast_file(path: &Path) {
    println!("Parsing {}...", path.display());

    let wast_test = match parse_wast_file(path) {
        Ok(t) => t,
        Err(e) => {
            println!("Parse error: {:?}", e);
            return;
        }
    };

    println!("Parsed {} module(s), {} assertion(s)", wast_test.modules.len(), wast_test.assertions.len());

    if wast_test.modules.is_empty() {
        println!("No modules found");
        return;
    }

    let mut passed = 0;
    let mut failed = 0;
    let mut skipped = 0;

    for assertion in &wast_test.assertions {
        match assertion {
            TestAssertion::AssertReturn { module_idx, func_name, args, expected } => {
                if *module_idx >= wast_test.modules.len() {
                    skipped += 1;
                    continue;
                }

                let (module, exports) = &wast_test.modules[*module_idx];
                match run_assert_return(module, exports, func_name, args, expected) {
                    Ok(true) => passed += 1,
                    Ok(false) => {
                        failed += 1;
                        if failed <= 10 {
                            println!("FAIL: {}({:?}) expected {:?}", func_name, args, expected);
                        }
                    }
                    Err(e) => {
                        failed += 1;
                        if failed <= 10 {
                            println!("ERROR: {}({:?}) - {:?}", func_name, args, e);
                        }
                    }
                }
            }
            TestAssertion::AssertTrap { module_idx, func_name, args, message } => {
                if *module_idx >= wast_test.modules.len() {
                    skipped += 1;
                    continue;
                }

                let (module, exports) = &wast_test.modules[*module_idx];
                match run_assert_trap(module, exports, func_name, args) {
                    Ok(true) => passed += 1,
                    Ok(false) => {
                        failed += 1;
                        if failed <= 10 {
                            println!("FAIL (expected trap): {}({:?}) - {}", func_name, args, message);
                        }
                    }
                    Err(e) => {
                        failed += 1;
                        if failed <= 10 {
                            println!("ERROR: {}({:?}) - {:?}", func_name, args, e);
                        }
                    }
                }
            }
            TestAssertion::AssertInvalid { .. } | TestAssertion::AssertMalformed { .. } => {
                skipped += 1;
            }
        }
    }

    println!("\nResults: {} passed, {} failed, {} skipped", passed, failed, skipped);
}

fn run_assert_return(
    module: &Module,
    exports: &HashMap<String, ExportItem>,
    func_name: &str,
    args: &[Value],
    expected: &[Value],
) -> Result<bool, String> {
    let func_idx = match exports.get(func_name) {
        Some(ExportItem::Func(idx)) => *idx,
        _ => return Err(format!("Function '{}' not found", func_name)),
    };

    let interp = ModuleInterpreter::new(module.clone());
    let mut state = InterpState::new();

    let cont = create_call_continuation(&interp, &mut state, func_idx, args.to_vec())
        .ok_or_else(|| "Failed to create continuation".to_string())?;

    match interp.interp(&mut state, cont) {
        InterpResult::Result(values) => {
            let matches = values_match(&values, expected);
            if !matches {
                println!("  Got: {:?}, Expected: {:?}", values, expected);
            }
            Ok(matches)
        }
        InterpResult::Trap => {
            println!("  Unexpected trap");
            Ok(false)
        }
    }
}

fn run_assert_trap(
    module: &Module,
    exports: &HashMap<String, ExportItem>,
    func_name: &str,
    args: &[Value],
) -> Result<bool, String> {
    let func_idx = match exports.get(func_name) {
        Some(ExportItem::Func(idx)) => *idx,
        _ => return Err(format!("Function '{}' not found", func_name)),
    };

    let interp = ModuleInterpreter::new(module.clone());
    let mut state = InterpState::new();

    let cont = create_call_continuation(&interp, &mut state, func_idx, args.to_vec())
        .ok_or_else(|| "Failed to create continuation".to_string())?;

    match interp.interp(&mut state, cont) {
        InterpResult::Trap => Ok(true),
        InterpResult::Result(values) => {
            println!("  Expected trap but got: {:?}", values);
            Ok(false)
        }
    }
}

fn values_match(actual: &[Value], expected: &[Value]) -> bool {
    if actual.len() != expected.len() {
        return false;
    }
    for (a, e) in actual.iter().zip(expected.iter()) {
        if !value_matches(a, e) {
            return false;
        }
    }
    true
}

fn value_matches(actual: &Value, expected: &Value) -> bool {
    match (actual, expected) {
        (Value::I32(a), Value::I32(e)) => a == e,
        (Value::I64(a), Value::I64(e)) => a == e,
        (Value::F32(a), Value::F32(e)) => {
            // Handle NaN specially
            if a.is_nan() && e.is_nan() {
                true
            } else {
                a.to_bits() == e.to_bits()
            }
        }
        (Value::F64(a), Value::F64(e)) => {
            if a.is_nan() && e.is_nan() {
                true
            } else {
                a.to_bits() == e.to_bits()
            }
        }
        _ => false,
    }
}

/// Helper to create an initial continuation for calling a function
fn create_call_continuation(
    interp: &ModuleInterpreter,
    state: &mut InterpState,
    func_id: u32,
    args: Vec<Value>,
) -> Option<Continuation> {
    let func = interp.module.funcs.get(func_id as usize)?;

    // Get function type to know param types
    let rec_type = interp.module.types.get(func.type_idx as usize)?;
    let sub_type = rec_type.subtypes.first()?;
    let (param_types, _) = match &sub_type.composite_type {
        CompositeType::Func(params, results) => (params, results),
    };

    // Build locals: params + declared locals
    let mut local_types: Vec<ValueType> = param_types.types.clone();
    local_types.extend(func.locals.iter().cloned());

    // Calculate space needed
    let locals_size: usize = local_types.iter().map(|t| t.size()).sum();
    let max_stack_size: usize = 4096;
    let frame_size = locals_size + max_stack_size;

    // Allocate from call stack
    let frame_start = state.call_stack_offset;
    let frame_end = frame_start + frame_size;

    if frame_end > state.call_stack.capacity() {
        return None;
    }

    let locals_view = state.call_stack.view(frame_start, frame_start + locals_size)?;
    let locals = TypedLocals::new(locals_view, &local_types)?;

    // Set arguments
    for (i, arg) in args.into_iter().enumerate() {
        if !locals.set(i as u32, arg) {
            return None;
        }
    }

    let stack_view = state.call_stack.view(frame_start + locals_size, frame_end)?;
    let stack = TypedStack::new(stack_view);

    state.call_stack_offset = frame_end;

    let loc = CodeLocation {
        func_id,
        loc: FuncLoc { block_id: NodeIdx(0), instr_idx: 0 },
    };

    Some(Continuation::new(loc, locals, stack))
}

// ============================================================================
// Built-in tests (for quick sanity checking)
// ============================================================================

fn run_builtin_tests() {
    println!("--- Built-in Tests ---");

    // Test 1: Simple add
    let add_module = create_add_module();
    let interp = ModuleInterpreter::new(add_module);
    let mut state = InterpState::new();

    let cont = create_call_continuation(&interp, &mut state, 0, vec![Value::I32(3), Value::I32(5)]);
    match cont {
        Some(c) => {
            match interp.interp(&mut state, c) {
                InterpResult::Result(values) => {
                    if let Some(Value::I32(v)) = values.first() {
                        if *v == 8 {
                            println!("Test 1 (add): PASS - 3 + 5 = {}", v);
                        } else {
                            println!("Test 1 (add): FAIL - expected 8, got {}", v);
                        }
                    }
                }
                InterpResult::Trap => println!("Test 1 (add): FAIL - trapped"),
            }
        }
        None => println!("Test 1 (add): FAIL - couldn't create continuation"),
    }

    // Test 2: Factorial
    let fact_module = create_factorial_module();
    let fact_interp = ModuleInterpreter::new(fact_module);
    let mut fact_state = InterpState::new();

    let fact_cont = create_call_continuation(&fact_interp, &mut fact_state, 0, vec![Value::I32(5)]);
    match fact_cont {
        Some(c) => {
            match fact_interp.interp(&mut fact_state, c) {
                InterpResult::Result(values) => {
                    if let Some(Value::I32(v)) = values.first() {
                        if *v == 120 {
                            println!("Test 2 (factorial): PASS - factorial(5) = {}", v);
                        } else {
                            println!("Test 2 (factorial): FAIL - expected 120, got {}", v);
                        }
                    }
                }
                InterpResult::Trap => println!("Test 2 (factorial): FAIL - trapped"),
            }
        }
        None => println!("Test 2 (factorial): FAIL - couldn't create continuation"),
    }
}

fn create_add_module() -> Module {
    let func_type = CompositeType::Func(
        ResultType { types: vec![ValueType::Number(NumberType::I32), ValueType::Number(NumberType::I32)] },
        ResultType { types: vec![ValueType::Number(NumberType::I32)] },
    );

    let rec_type = RecType {
        subtypes: vec![SubType { composite_type: func_type }],
    };

    let body = vec![
        Instruction::Block(Box::new(BlockInst {
            block_type: BlockType::None,
            instrs: vec![NodeIdx(1), NodeIdx(2), NodeIdx(3), NodeIdx(4)],
            is_loop: false,
            next: None,  // Entry block - implicit return when done
        })),
        Instruction::LocalGet(0),
        Instruction::LocalGet(1),
        Instruction::Binary { op: BinaryOp::Add, typ: NumberType::I32 },
        Instruction::Return,
    ];

    let func = Function {
        type_idx: 0,
        locals: vec![],
        body,
    };

    Module {
        types: vec![rec_type],
        funcs: vec![func],
        mems: vec![],
        start: None,
    }
}

fn create_factorial_module() -> Module {
    let func_type = CompositeType::Func(
        ResultType { types: vec![ValueType::Number(NumberType::I32)] },
        ResultType { types: vec![ValueType::Number(NumberType::I32)] },
    );

    let rec_type = RecType {
        subtypes: vec![SubType { composite_type: func_type }],
    };

    let body = vec![
        Instruction::Block(Box::new(BlockInst {
            block_type: BlockType::None,
            instrs: vec![NodeIdx(1), NodeIdx(2), NodeIdx(3), NodeIdx(4)],
            is_loop: false,
            next: None,  // Entry block - implicit return when done
        })),
        Instruction::LocalGet(0),
        Instruction::Const(Value::I32(1)),
        Instruction::Compare { op: CompareOp::LeS, typ: NumberType::I32 },
        Instruction::If(Box::new(IfInst {
            block_type: BlockType::None,
            then_block: NodeIdx(5),
            else_block: NodeIdx(8),
            next: FuncLoc { block_id: NodeIdx(0), instr_idx: 4 },
        })),
        Instruction::Block(Box::new(BlockInst {
            block_type: BlockType::None,
            instrs: vec![NodeIdx(6), NodeIdx(7)],
            is_loop: false,
            next: Some(FuncLoc { block_id: NodeIdx(0), instr_idx: 4 }),  // After if in entry block
        })),
        Instruction::Const(Value::I32(1)),
        Instruction::Return,
        Instruction::Block(Box::new(BlockInst {
            block_type: BlockType::None,
            instrs: vec![NodeIdx(9), NodeIdx(10), NodeIdx(11), NodeIdx(12), NodeIdx(13), NodeIdx(14), NodeIdx(15)],
            is_loop: false,
            next: Some(FuncLoc { block_id: NodeIdx(0), instr_idx: 4 }),  // After if in entry block
        })),
        Instruction::LocalGet(0),
        Instruction::LocalGet(0),
        Instruction::Const(Value::I32(1)),
        Instruction::Binary { op: BinaryOp::Sub, typ: NumberType::I32 },
        Instruction::Call(0),
        Instruction::Binary { op: BinaryOp::Mul, typ: NumberType::I32 },
        Instruction::Return,
    ];

    let func = Function {
        type_idx: 0,
        locals: vec![],
        body,
    };

    Module {
        types: vec![rec_type],
        funcs: vec![func],
        mems: vec![],
        start: None,
    }
}
