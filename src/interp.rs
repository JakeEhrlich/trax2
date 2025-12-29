use crate::types::*;

// ============================================================================
// Interpreter State and Results
// ============================================================================

#[derive(Debug)]
pub struct InterpState {
    pub mems: Vec<Memory>,
    /// Pre-allocated stack memory for call frames (locals + operand stacks)
    pub call_stack: Memory,
    /// Current offset into call_stack where next frame should be allocated
    pub call_stack_offset: usize,
    // TODO: globals: TypedLocals or similar
}

impl InterpState {
    pub fn new() -> Self {
        // Allocate a generous call stack (e.g., 16 pages = 1MB)
        let call_stack = Memory::new(16);
        InterpState {
            mems: Vec::new(),
            call_stack,
            call_stack_offset: 0,
        }
    }

    pub fn with_memories(mems: Vec<Memory>) -> Self {
        let call_stack = Memory::new(16);
        InterpState {
            mems,
            call_stack,
            call_stack_offset: 0,
        }
    }
}

/// Result of a single interpreter step.
///
/// Note: We use `Option<StepResult>` as the return type from step(),
/// where `None` represents a trap. This allows using `?` for cleaner code.
#[derive(Debug)]
pub enum StepResult {
    /// More progress can be made by this interpreter
    Cont(Continuation),
    /// This interpreter can make no more progress and a more general interpreter
    /// should take over. ModuleInterpreter never returns this - it's for specialized
    /// interpreters (e.g., trace interpreters) that hit side exits.
    Final(Continuation),
    /// Execution completed with return values
    Result(Vec<Value>),
}

#[derive(Debug)]
pub enum InterpResult {
    /// Execution completed with return values
    Result(Vec<Value>),
    /// A trap occurred
    Trap,
}

/// Result of advancing to the next instruction location.
#[derive(Debug)]
enum AdvanceResult {
    /// Continue execution at this location
    Continue(CodeLocation),
    /// Implicit return - we've fallen through the end of the function
    ImplicitReturn,
}

// ============================================================================
// Interpreter Trait
// ============================================================================

pub trait Interpreter {
    /// Execute a single step. Returns None on trap.
    fn step(&self, state: &mut InterpState, cont: Continuation) -> Option<StepResult>;

    fn interp(&self, state: &mut InterpState, init_cont: Continuation) -> InterpResult {
        let mut cont = init_cont;
        loop {
            match self.step(state, cont) {
                Some(StepResult::Cont(new_cont)) => cont = new_cont,
                Some(StepResult::Final(_)) => {
                    // ModuleInterpreter shouldn't return Final, but if it does
                    // we treat it as a trap since there's no more general interpreter
                    return InterpResult::Trap;
                }
                Some(StepResult::Result(values)) => return InterpResult::Result(values),
                None => return InterpResult::Trap,
            }
        }
    }
}

// ============================================================================
// Module Interpreter
// ============================================================================

pub struct ModuleInterpreter {
    pub module: Module,
}

impl ModuleInterpreter {
    pub fn new(module: Module) -> Self {
        ModuleInterpreter { module }
    }

    fn get_func(&self, func_id: u32) -> Option<&Function> {
        self.module.funcs.get(func_id as usize)
    }

    // TODO: move this to be a method on Function
    fn get_instruction<'a>(&self, func: &'a Function, node_idx: NodeIdx) -> Option<&'a Instruction> {
        func.body.get(node_idx.0 as usize)
    }

    /// Get the function type (params, results) for a function
    fn get_func_type(&self, func: &Function) -> Option<(&ResultType, &ResultType)> {
        let rec_type = self.module.types.get(func.type_idx as usize)?;
        let sub_type = rec_type.subtypes.first()?;
        match &sub_type.composite_type {
            CompositeType::Func(params, results) => Some((params, results)),
        }
    }

    /// Helper to create a CodeLocation within the current function
    fn make_loc(&self, func_id: u32, block_id: NodeIdx, instr_idx: u32) -> CodeLocation {
        CodeLocation {
            func_id,
            loc: FuncLoc { block_id, instr_idx },
        }
    }

    /// Helper to advance to the next instruction in a block, or to block.next if done.
    /// Returns None on error (trap), or AdvanceResult indicating where to go next.
    fn advance_loc(&self, func: &Function, loc: CodeLocation) -> Option<AdvanceResult> {
        let block = match self.get_instruction(func, loc.loc.block_id)? {
            Instruction::Block(b) => b,
            _ => return None,
        };
        match block.next_loc(loc.loc) {
            Some(new_loc) => Some(AdvanceResult::Continue(CodeLocation {
                func_id: loc.func_id,
                loc: new_loc,
            })),
            None => Some(AdvanceResult::ImplicitReturn),
        }
    }

    /// Helper to branch to a block (handles loop vs regular block)
    fn branch_to_block(&self, func: &Function, func_id: u32, target: NodeIdx) -> Option<AdvanceResult> {
        let target_block = match self.get_instruction(func, target)? {
            Instruction::Block(b) => b,
            _ => return None,
        };

        if target_block.is_loop {
            // Loop: branch to start of block
            Some(AdvanceResult::Continue(CodeLocation {
                func_id,
                loc: FuncLoc { block_id: target, instr_idx: 0 },
            }))
        } else {
            // Block: branch to after block
            match target_block.next {
                Some(next_loc) => Some(AdvanceResult::Continue(CodeLocation { func_id, loc: next_loc })),
                None => Some(AdvanceResult::ImplicitReturn),
            }
        }
    }

    /// Helper to perform a return (explicit or implicit).
    /// Pops return values from the stack and either returns to parent or completes execution.
    fn do_return(
        &self,
        state: &mut InterpState,
        mut cont: Continuation,
        func: &Function,
    ) -> Option<StepResult> {
        let (_, result_types) = self.get_func_type(func)?;

        // Pop return values
        let mut return_values: Vec<Value> = Vec::with_capacity(result_types.types.len());
        for _ in 0..result_types.types.len() {
            return_values.push(cont.frame.stack.pop()?);
        }
        return_values.reverse();

        // Calculate this frame's size to restore call_stack_offset
        let locals_byte_size = cont.frame.locals.byte_size();
        let stack_capacity = cont.frame.stack.view.capacity - cont.frame.stack.view.start;
        let frame_size = locals_byte_size + stack_capacity;

        // Restore call stack offset
        state.call_stack_offset -= frame_size;

        // If there's a parent continuation, push results there and continue
        if let Some(parent) = cont.parent {
            let mut parent_cont = *parent;
            for val in return_values {
                if !parent_cont.frame.stack.push(val) {
                    return None;
                }
            }
            Some(StepResult::Cont(parent_cont))
        } else {
            // No parent - we're done
            Some(StepResult::Result(return_values))
        }
    }
}

impl Interpreter for ModuleInterpreter {
    fn step(&self, state: &mut InterpState, mut cont: Continuation) -> Option<StepResult> {
        let loc = cont.frame.loc;
        let func = self.get_func(loc.func_id)?;

        // Get the current block
        let block = match self.get_instruction(func, loc.loc.block_id)? {
            Instruction::Block(b) => b,
            _ => return None,
        };

        // Check if we're past the end of this block
        if loc.loc.instr_idx as usize >= block.instrs.len() {
            // We've finished this block, go to next location (or implicit return)
            match block.next {
                Some(next_loc) => {
                    cont.frame.loc = CodeLocation {
                        func_id: loc.func_id,
                        loc: next_loc,
                    };
                    return Some(StepResult::Cont(cont));
                }
                None => {
                    // Implicit return
                    return self.do_return(state, cont, func);
                }
            }
        }

        // Get the instruction index from the block
        let inst_node_idx = block.instrs[loc.loc.instr_idx as usize];
        let inst = self.get_instruction(func, inst_node_idx)?;

        match inst {
            Instruction::Nop => {
                // Do nothing, advance to next instruction
            }

            Instruction::Unreachable => {
                return None;
            }

            Instruction::Drop => {
                cont.frame.stack.pop()?;
            }

            Instruction::Select(_type_hint) => {
                // Pop condition, then two values
                let cond = cont.frame.stack.pop()?;
                let val2 = cont.frame.stack.pop()?;
                let val1 = cont.frame.stack.pop()?;

                let is_truthy = cond.is_truthy()?;
                let result = if is_truthy { val1 } else { val2 };

                if !cont.frame.stack.push(result) {
                    return None;
                }
            }

            Instruction::Block(_inner_block) => {
                // Enter the block - jump to first instruction of this block
                cont.frame.loc = self.make_loc(loc.func_id, inst_node_idx, 0);
                return Some(StepResult::Cont(cont));
            }

            Instruction::If(if_inst) => {
                let cond = cont.frame.stack.pop()?;
                let is_truthy = cond.is_truthy()?;

                let target_block = if is_truthy {
                    if_inst.then_block
                } else {
                    if_inst.else_block
                };

                cont.frame.loc = self.make_loc(loc.func_id, target_block, 0);
                return Some(StepResult::Cont(cont));
            }

            Instruction::Br(target_block_idx) => {
                match self.branch_to_block(func, loc.func_id, *target_block_idx)? {
                    AdvanceResult::Continue(new_loc) => {
                        cont.frame.loc = new_loc;
                        return Some(StepResult::Cont(cont));
                    }
                    AdvanceResult::ImplicitReturn => {
                        return self.do_return(state, cont, func);
                    }
                }
            }

            Instruction::BrIf(target_block_idx) => {
                let cond = cont.frame.stack.pop()?;

                if cond.is_truthy()? {
                    match self.branch_to_block(func, loc.func_id, *target_block_idx)? {
                        AdvanceResult::Continue(new_loc) => {
                            cont.frame.loc = new_loc;
                            return Some(StepResult::Cont(cont));
                        }
                        AdvanceResult::ImplicitReturn => {
                            return self.do_return(state, cont, func);
                        }
                    }
                }
                // Branch not taken, fall through to next instruction
            }

            Instruction::BrTable { labels, default } => {
                let idx = cont.frame.stack.pop()?;
                let idx_val = match idx {
                    Value::I32(v) => v as usize,
                    _ => return None,
                };

                let target = if idx_val < labels.len() {
                    labels[idx_val]
                } else {
                    *default
                };

                match self.branch_to_block(func, loc.func_id, target)? {
                    AdvanceResult::Continue(new_loc) => {
                        cont.frame.loc = new_loc;
                        return Some(StepResult::Cont(cont));
                    }
                    AdvanceResult::ImplicitReturn => {
                        return self.do_return(state, cont, func);
                    }
                }
            }

            Instruction::Call(callee_func_id) => {
                let callee_func = self.get_func(*callee_func_id)?;
                let (param_types, _result_types) = self.get_func_type(callee_func)?;

                // TODO: later optimize - the reverse and Value conversions are not needed,
                // these are just memcpy operations if done right. When we lay out memory
                // correctly on return, the stack expansion is almost free.

                // Pop arguments from stack (in reverse order)
                let mut args: Vec<Value> = Vec::with_capacity(param_types.types.len());
                for _ in 0..param_types.types.len() {
                    args.push(cont.frame.stack.pop()?);
                }
                args.reverse();

                // Build locals: params + declared locals
                let mut local_types: Vec<ValueType> = param_types.types.clone();
                local_types.extend(callee_func.locals.iter().cloned());

                // Calculate space needed for locals
                let locals_size: usize = local_types.iter().map(|t| t.size()).sum();

                // TODO: pre-calculate max stack size for each function to pre-allocate
                // For now, use a reasonable default
                let max_stack_size: usize = 4096; // Conservative estimate

                let frame_size = locals_size + max_stack_size;
                let frame_start = state.call_stack_offset;
                let frame_end = frame_start + frame_size;

                // Check we have space
                if frame_end > state.call_stack.capacity() {
                    return None; // Stack overflow
                }

                // Allocate views from the pre-allocated call stack
                let locals_view = state.call_stack.view(frame_start, frame_start + locals_size)?;
                let new_locals = TypedLocals::new(locals_view, &local_types)?;

                // Set parameter values
                for (i, arg) in args.into_iter().enumerate() {
                    if !new_locals.set(i as u32, arg) {
                        return None;
                    }
                }

                // Stack goes after locals
                let stack_view = state.call_stack.view(frame_start + locals_size, frame_end)?;
                let new_stack = TypedStack::new(stack_view);

                // Update call stack offset
                state.call_stack_offset = frame_end;

                // Find the entry block (should be NodeIdx(0) typically)
                let new_code_loc = self.make_loc(*callee_func_id, NodeIdx(0), 0);

                // Update caller's location to after the call
                // If the caller would implicitly return after this call, we point to
                // the end of the entry block so the implicit return happens when callee returns
                match self.advance_loc(func, loc)? {
                    AdvanceResult::Continue(new_loc) => {
                        cont.frame.loc = new_loc;
                    }
                    AdvanceResult::ImplicitReturn => {
                        // Point to past the end of entry block to trigger implicit return
                        // when we return to this frame
                        let entry_block = match self.get_instruction(func, loc.loc.block_id)? {
                            Instruction::Block(b) => b,
                            _ => return None,
                        };
                        cont.frame.loc = CodeLocation {
                            func_id: loc.func_id,
                            loc: FuncLoc {
                                block_id: loc.loc.block_id,
                                instr_idx: entry_block.instrs.len() as u32,
                            },
                        };
                    }
                }

                // Create new continuation with parent
                let new_cont = Continuation::with_parent(new_code_loc, new_locals, new_stack, cont);

                return Some(StepResult::Cont(new_cont));
            }

            Instruction::Return => {
                return self.do_return(state, cont, func);
            }

            Instruction::LocalGet(local_idx) => {
                let val = cont.frame.locals.get(*local_idx)?;
                if !cont.frame.stack.push(val) {
                    return None;
                }
            }

            Instruction::LocalSet(local_idx) => {
                let val = cont.frame.stack.pop()?;
                if !cont.frame.locals.set(*local_idx, val) {
                    return None;
                }
            }

            Instruction::LocalTee(local_idx) => {
                let val = cont.frame.stack.peek()?;
                if !cont.frame.locals.set(*local_idx, val) {
                    return None;
                }
            }

            Instruction::TableGet(_table_idx) => {
                // TODO: implement tables
                return None;
            }

            Instruction::TableSize(_table_idx) => {
                // TODO: implement tables
                return None;
            }

            Instruction::Load { mem_arg, val_type } => {
                let addr_val = cont.frame.stack.pop()?;

                let addr = match addr_val {
                    Value::I32(a) => a as u64,
                    Value::I64(a) => a as u64,
                    _ => return None,
                };

                let effective_addr = addr + mem_arg.offset;

                let mem = state.mems.get(mem_arg.mem as usize)?;
                let view = mem.full_view();
                let vt = ValueType::Number(*val_type);
                let val = view.get_value(effective_addr as usize, &vt)?;

                if !cont.frame.stack.push(val) {
                    return None;
                }
            }

            Instruction::Store { mem_arg, val_type: _ } => {
                let val = cont.frame.stack.pop()?;
                let addr_val = cont.frame.stack.pop()?;

                let addr = match addr_val {
                    Value::I32(a) => a as u64,
                    Value::I64(a) => a as u64,
                    _ => return None,
                };

                let effective_addr = addr + mem_arg.offset;

                let mem = state.mems.get(mem_arg.mem as usize)?;
                let view = mem.full_view();
                if !view.set_value(effective_addr as usize, &val) {
                    return None;
                }
            }

            Instruction::MemorySize(mem_idx) => {
                let mem = state.mems.get(*mem_idx as usize)?;
                let pages = mem.pages();
                if !cont.frame.stack.push(Value::I32(pages as i32)) {
                    return None;
                }
            }

            Instruction::Const(val) => {
                if !cont.frame.stack.push(*val) {
                    return None;
                }
            }

            Instruction::Unary { op, typ } => {
                let val = cont.frame.stack.pop()?;
                let result = execute_unary_op(*op, *typ, val)?;
                if !cont.frame.stack.push(result) {
                    return None;
                }
            }

            Instruction::Binary { op, typ } => {
                let val2 = cont.frame.stack.pop()?;
                let val1 = cont.frame.stack.pop()?;
                let result = execute_binary_op(*op, *typ, val1, val2)?;
                if !cont.frame.stack.push(result) {
                    return None;
                }
            }

            Instruction::Compare { op, typ } => {
                let val2 = cont.frame.stack.pop()?;
                let val1 = cont.frame.stack.pop()?;
                let result = execute_compare_op(*op, *typ, val1, val2)?;
                if !cont.frame.stack.push(result) {
                    return None;
                }
            }

            Instruction::Convert { op, from, to } => {
                let val = cont.frame.stack.pop()?;
                let result = execute_convert_op(*op, *from, *to, val)?;
                if !cont.frame.stack.push(result) {
                    return None;
                }
            }
        }

        // Advance to next instruction
        match self.advance_loc(func, loc)? {
            AdvanceResult::Continue(new_loc) => {
                cont.frame.loc = new_loc;
                Some(StepResult::Cont(cont))
            }
            AdvanceResult::ImplicitReturn => {
                self.do_return(state, cont, func)
            }
        }
    }
}

// ============================================================================
// Helper functions for WASM float semantics
// ============================================================================

/// WASM nearest for f32 - rounds to nearest, ties to even
fn wasm_nearest_f32(x: f32) -> f32 {
    if x.is_nan() || x.is_infinite() || x == 0.0 {
        return x;
    }
    let rounded = x.round();
    // Check if we're exactly at a .5 boundary
    let diff = (x - rounded).abs();
    if diff == 0.0 {
        rounded
    } else if (x - x.floor()).abs() == 0.5 {
        // Ties to even
        let floor = x.floor();
        let ceil = x.ceil();
        if (floor as i32) % 2 == 0 {
            floor
        } else {
            ceil
        }
    } else {
        rounded
    }
}

/// WASM nearest for f64 - rounds to nearest, ties to even
fn wasm_nearest_f64(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() || x == 0.0 {
        return x;
    }
    let rounded = x.round();
    // Check if we're exactly at a .5 boundary
    let diff = (x - rounded).abs();
    if diff == 0.0 {
        rounded
    } else if (x - x.floor()).abs() == 0.5 {
        // Ties to even
        let floor = x.floor();
        let ceil = x.ceil();
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            ceil
        }
    } else {
        rounded
    }
}

/// WASM min for f32 - propagates NaN, returns -0 if either is -0 and other is +0
fn wasm_min_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == 0.0 && b == 0.0 {
        // Handle signed zero
        if a.to_bits() == 0x8000_0000 || b.to_bits() == 0x8000_0000 {
            -0.0_f32
        } else {
            0.0_f32
        }
    } else {
        a.min(b)
    }
}

/// WASM min for f64 - propagates NaN, returns -0 if either is -0 and other is +0
fn wasm_min_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == 0.0 && b == 0.0 {
        // Handle signed zero
        if a.to_bits() == 0x8000_0000_0000_0000 || b.to_bits() == 0x8000_0000_0000_0000 {
            -0.0_f64
        } else {
            0.0_f64
        }
    } else {
        a.min(b)
    }
}

/// WASM max for f32 - propagates NaN, returns +0 if one is +0 and other is -0
fn wasm_max_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == 0.0 && b == 0.0 {
        // Handle signed zero
        if a.to_bits() == 0 || b.to_bits() == 0 {
            0.0_f32
        } else {
            -0.0_f32
        }
    } else {
        a.max(b)
    }
}

/// WASM max for f64 - propagates NaN, returns +0 if one is +0 and other is -0
fn wasm_max_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == 0.0 && b == 0.0 {
        // Handle signed zero
        if a.to_bits() == 0 || b.to_bits() == 0 {
            0.0_f64
        } else {
            -0.0_f64
        }
    } else {
        a.max(b)
    }
}

// ============================================================================
// Unary Operation Implementation
// ============================================================================

fn execute_unary_op(op: UnaryOp, typ: NumberType, val: Value) -> Option<Value> {
    Some(match (op, typ, val) {
        // Eqz
        (UnaryOp::Eqz, NumberType::I32, Value::I32(x)) => Value::I32(if x == 0 { 1 } else { 0 }),
        (UnaryOp::Eqz, NumberType::I64, Value::I64(x)) => Value::I32(if x == 0 { 1 } else { 0 }),

        // Float ops
        (UnaryOp::Neg, NumberType::F32, Value::F32(x)) => Value::F32(-x),
        (UnaryOp::Neg, NumberType::F64, Value::F64(x)) => Value::F64(-x),
        (UnaryOp::Abs, NumberType::F32, Value::F32(x)) => Value::F32(x.abs()),
        (UnaryOp::Abs, NumberType::F64, Value::F64(x)) => Value::F64(x.abs()),

        // Float math ops
        (UnaryOp::Sqrt, NumberType::F32, Value::F32(x)) => Value::F32(x.sqrt()),
        (UnaryOp::Sqrt, NumberType::F64, Value::F64(x)) => Value::F64(x.sqrt()),
        (UnaryOp::Ceil, NumberType::F32, Value::F32(x)) => Value::F32(x.ceil()),
        (UnaryOp::Ceil, NumberType::F64, Value::F64(x)) => Value::F64(x.ceil()),
        (UnaryOp::Floor, NumberType::F32, Value::F32(x)) => Value::F32(x.floor()),
        (UnaryOp::Floor, NumberType::F64, Value::F64(x)) => Value::F64(x.floor()),
        (UnaryOp::Trunc, NumberType::F32, Value::F32(x)) => Value::F32(x.trunc()),
        (UnaryOp::Trunc, NumberType::F64, Value::F64(x)) => Value::F64(x.trunc()),
        (UnaryOp::Nearest, NumberType::F32, Value::F32(x)) => Value::F32(wasm_nearest_f32(x)),
        (UnaryOp::Nearest, NumberType::F64, Value::F64(x)) => Value::F64(wasm_nearest_f64(x)),

        // Clz (count leading zeros)
        (UnaryOp::Clz, NumberType::I32, Value::I32(x)) => Value::I32(x.leading_zeros() as i32),
        (UnaryOp::Clz, NumberType::I64, Value::I64(x)) => Value::I64(x.leading_zeros() as i64),

        // Ctz (count trailing zeros)
        (UnaryOp::Ctz, NumberType::I32, Value::I32(x)) => Value::I32(x.trailing_zeros() as i32),
        (UnaryOp::Ctz, NumberType::I64, Value::I64(x)) => Value::I64(x.trailing_zeros() as i64),

        // Popcnt (population count / count ones)
        (UnaryOp::Popcnt, NumberType::I32, Value::I32(x)) => Value::I32(x.count_ones() as i32),
        (UnaryOp::Popcnt, NumberType::I64, Value::I64(x)) => Value::I64(x.count_ones() as i64),

        // Extend8S (sign extend from 8 bits)
        (UnaryOp::Extend8S, NumberType::I32, Value::I32(x)) => Value::I32((x as i8) as i32),
        (UnaryOp::Extend8S, NumberType::I64, Value::I64(x)) => Value::I64((x as i8) as i64),

        // Extend16S (sign extend from 16 bits)
        (UnaryOp::Extend16S, NumberType::I32, Value::I32(x)) => Value::I32((x as i16) as i32),
        (UnaryOp::Extend16S, NumberType::I64, Value::I64(x)) => Value::I64((x as i16) as i64),

        // Extend32S (sign extend from 32 bits, i64 only)
        (UnaryOp::Extend32S, NumberType::I64, Value::I64(x)) => Value::I64((x as i32) as i64),

        _ => return None,
    })
}

// ============================================================================
// Binary Operation Implementation
// ============================================================================

fn execute_binary_op(op: BinaryOp, typ: NumberType, v1: Value, v2: Value) -> Option<Value> {
    Some(match (op, typ, v1, v2) {
        // I32 operations
        (BinaryOp::Add, NumberType::I32, Value::I32(a), Value::I32(b)) => {
            Value::I32(a.wrapping_add(b))
        }
        (BinaryOp::Sub, NumberType::I32, Value::I32(a), Value::I32(b)) => {
            Value::I32(a.wrapping_sub(b))
        }
        (BinaryOp::Mul, NumberType::I32, Value::I32(a), Value::I32(b)) => {
            Value::I32(a.wrapping_mul(b))
        }
        (BinaryOp::DivS, NumberType::I32, Value::I32(a), Value::I32(b)) => {
            if b == 0 { return None; }
            // Check for overflow: MIN_INT / -1 would overflow
            if a == i32::MIN && b == -1 { return None; }
            Value::I32(a.wrapping_div(b))
        }
        (BinaryOp::DivU, NumberType::I32, Value::I32(a), Value::I32(b)) => {
            if b == 0 { return None; }
            Value::I32((a as u32).wrapping_div(b as u32) as i32)
        }
        (BinaryOp::RemS, NumberType::I32, Value::I32(a), Value::I32(b)) => {
            if b == 0 { return None; }
            Value::I32(a.wrapping_rem(b))
        }
        (BinaryOp::RemU, NumberType::I32, Value::I32(a), Value::I32(b)) => {
            if b == 0 { return None; }
            Value::I32((a as u32).wrapping_rem(b as u32) as i32)
        }
        (BinaryOp::And, NumberType::I32, Value::I32(a), Value::I32(b)) => Value::I32(a & b),
        (BinaryOp::Or, NumberType::I32, Value::I32(a), Value::I32(b)) => Value::I32(a | b),
        (BinaryOp::Xor, NumberType::I32, Value::I32(a), Value::I32(b)) => Value::I32(a ^ b),
        (BinaryOp::Shl, NumberType::I32, Value::I32(a), Value::I32(b)) => {
            Value::I32(a.wrapping_shl(b as u32))
        }
        (BinaryOp::ShrS, NumberType::I32, Value::I32(a), Value::I32(b)) => {
            Value::I32(a.wrapping_shr(b as u32))
        }
        (BinaryOp::ShrU, NumberType::I32, Value::I32(a), Value::I32(b)) => {
            Value::I32((a as u32).wrapping_shr(b as u32) as i32)
        }
        (BinaryOp::Rotl, NumberType::I32, Value::I32(a), Value::I32(b)) => {
            Value::I32((a as u32).rotate_left(b as u32) as i32)
        }
        (BinaryOp::Rotr, NumberType::I32, Value::I32(a), Value::I32(b)) => {
            Value::I32((a as u32).rotate_right(b as u32) as i32)
        }

        // I64 operations
        (BinaryOp::Add, NumberType::I64, Value::I64(a), Value::I64(b)) => {
            Value::I64(a.wrapping_add(b))
        }
        (BinaryOp::Sub, NumberType::I64, Value::I64(a), Value::I64(b)) => {
            Value::I64(a.wrapping_sub(b))
        }
        (BinaryOp::Mul, NumberType::I64, Value::I64(a), Value::I64(b)) => {
            Value::I64(a.wrapping_mul(b))
        }
        (BinaryOp::DivS, NumberType::I64, Value::I64(a), Value::I64(b)) => {
            if b == 0 { return None; }
            // Check for overflow: MIN_INT / -1 would overflow
            if a == i64::MIN && b == -1 { return None; }
            Value::I64(a.wrapping_div(b))
        }
        (BinaryOp::DivU, NumberType::I64, Value::I64(a), Value::I64(b)) => {
            if b == 0 { return None; }
            Value::I64((a as u64).wrapping_div(b as u64) as i64)
        }
        (BinaryOp::RemS, NumberType::I64, Value::I64(a), Value::I64(b)) => {
            if b == 0 { return None; }
            Value::I64(a.wrapping_rem(b))
        }
        (BinaryOp::RemU, NumberType::I64, Value::I64(a), Value::I64(b)) => {
            if b == 0 { return None; }
            Value::I64((a as u64).wrapping_rem(b as u64) as i64)
        }
        (BinaryOp::And, NumberType::I64, Value::I64(a), Value::I64(b)) => Value::I64(a & b),
        (BinaryOp::Or, NumberType::I64, Value::I64(a), Value::I64(b)) => Value::I64(a | b),
        (BinaryOp::Xor, NumberType::I64, Value::I64(a), Value::I64(b)) => Value::I64(a ^ b),
        (BinaryOp::Shl, NumberType::I64, Value::I64(a), Value::I64(b)) => {
            Value::I64(a.wrapping_shl(b as u32))
        }
        (BinaryOp::ShrS, NumberType::I64, Value::I64(a), Value::I64(b)) => {
            Value::I64(a.wrapping_shr(b as u32))
        }
        (BinaryOp::ShrU, NumberType::I64, Value::I64(a), Value::I64(b)) => {
            Value::I64((a as u64).wrapping_shr(b as u32) as i64)
        }
        (BinaryOp::Rotl, NumberType::I64, Value::I64(a), Value::I64(b)) => {
            Value::I64((a as u64).rotate_left(b as u32) as i64)
        }
        (BinaryOp::Rotr, NumberType::I64, Value::I64(a), Value::I64(b)) => {
            Value::I64((a as u64).rotate_right(b as u32) as i64)
        }

        // F32 operations
        (BinaryOp::Add, NumberType::F32, Value::F32(a), Value::F32(b)) => Value::F32(a + b),
        (BinaryOp::Sub, NumberType::F32, Value::F32(a), Value::F32(b)) => Value::F32(a - b),
        (BinaryOp::Mul, NumberType::F32, Value::F32(a), Value::F32(b)) => Value::F32(a * b),
        (BinaryOp::Div, NumberType::F32, Value::F32(a), Value::F32(b)) => Value::F32(a / b),
        (BinaryOp::Min, NumberType::F32, Value::F32(a), Value::F32(b)) => Value::F32(wasm_min_f32(a, b)),
        (BinaryOp::Max, NumberType::F32, Value::F32(a), Value::F32(b)) => Value::F32(wasm_max_f32(a, b)),
        (BinaryOp::Copysign, NumberType::F32, Value::F32(a), Value::F32(b)) => Value::F32(a.copysign(b)),

        // F64 operations
        (BinaryOp::Add, NumberType::F64, Value::F64(a), Value::F64(b)) => Value::F64(a + b),
        (BinaryOp::Sub, NumberType::F64, Value::F64(a), Value::F64(b)) => Value::F64(a - b),
        (BinaryOp::Mul, NumberType::F64, Value::F64(a), Value::F64(b)) => Value::F64(a * b),
        (BinaryOp::Div, NumberType::F64, Value::F64(a), Value::F64(b)) => Value::F64(a / b),
        (BinaryOp::Min, NumberType::F64, Value::F64(a), Value::F64(b)) => Value::F64(wasm_min_f64(a, b)),
        (BinaryOp::Max, NumberType::F64, Value::F64(a), Value::F64(b)) => Value::F64(wasm_max_f64(a, b)),
        (BinaryOp::Copysign, NumberType::F64, Value::F64(a), Value::F64(b)) => Value::F64(a.copysign(b)),

        _ => return None,
    })
}

// ============================================================================
// Compare Operation Implementation
// ============================================================================

fn execute_compare_op(op: CompareOp, typ: NumberType, v1: Value, v2: Value) -> Option<Value> {
    let result = match (op, typ, v1, v2) {
        // I32 comparisons
        (CompareOp::Eq, NumberType::I32, Value::I32(a), Value::I32(b)) => a == b,
        (CompareOp::Ne, NumberType::I32, Value::I32(a), Value::I32(b)) => a != b,
        (CompareOp::LtS, NumberType::I32, Value::I32(a), Value::I32(b)) => a < b,
        (CompareOp::LtU, NumberType::I32, Value::I32(a), Value::I32(b)) => (a as u32) < (b as u32),
        (CompareOp::GtS, NumberType::I32, Value::I32(a), Value::I32(b)) => a > b,
        (CompareOp::GtU, NumberType::I32, Value::I32(a), Value::I32(b)) => (a as u32) > (b as u32),
        (CompareOp::LeS, NumberType::I32, Value::I32(a), Value::I32(b)) => a <= b,
        (CompareOp::LeU, NumberType::I32, Value::I32(a), Value::I32(b)) => (a as u32) <= (b as u32),
        (CompareOp::GeS, NumberType::I32, Value::I32(a), Value::I32(b)) => a >= b,
        (CompareOp::GeU, NumberType::I32, Value::I32(a), Value::I32(b)) => (a as u32) >= (b as u32),

        // I64 comparisons
        (CompareOp::Eq, NumberType::I64, Value::I64(a), Value::I64(b)) => a == b,
        (CompareOp::Ne, NumberType::I64, Value::I64(a), Value::I64(b)) => a != b,
        (CompareOp::LtS, NumberType::I64, Value::I64(a), Value::I64(b)) => a < b,
        (CompareOp::LtU, NumberType::I64, Value::I64(a), Value::I64(b)) => (a as u64) < (b as u64),
        (CompareOp::GtS, NumberType::I64, Value::I64(a), Value::I64(b)) => a > b,
        (CompareOp::GtU, NumberType::I64, Value::I64(a), Value::I64(b)) => (a as u64) > (b as u64),
        (CompareOp::LeS, NumberType::I64, Value::I64(a), Value::I64(b)) => a <= b,
        (CompareOp::LeU, NumberType::I64, Value::I64(a), Value::I64(b)) => (a as u64) <= (b as u64),
        (CompareOp::GeS, NumberType::I64, Value::I64(a), Value::I64(b)) => a >= b,
        (CompareOp::GeU, NumberType::I64, Value::I64(a), Value::I64(b)) => (a as u64) >= (b as u64),

        // F32 comparisons
        (CompareOp::Eq, NumberType::F32, Value::F32(a), Value::F32(b)) => a == b,
        (CompareOp::Ne, NumberType::F32, Value::F32(a), Value::F32(b)) => a != b,
        (CompareOp::Lt, NumberType::F32, Value::F32(a), Value::F32(b)) => a < b,
        (CompareOp::Gt, NumberType::F32, Value::F32(a), Value::F32(b)) => a > b,
        (CompareOp::Le, NumberType::F32, Value::F32(a), Value::F32(b)) => a <= b,
        (CompareOp::Ge, NumberType::F32, Value::F32(a), Value::F32(b)) => a >= b,

        // F64 comparisons
        (CompareOp::Eq, NumberType::F64, Value::F64(a), Value::F64(b)) => a == b,
        (CompareOp::Ne, NumberType::F64, Value::F64(a), Value::F64(b)) => a != b,
        (CompareOp::Lt, NumberType::F64, Value::F64(a), Value::F64(b)) => a < b,
        (CompareOp::Gt, NumberType::F64, Value::F64(a), Value::F64(b)) => a > b,
        (CompareOp::Le, NumberType::F64, Value::F64(a), Value::F64(b)) => a <= b,
        (CompareOp::Ge, NumberType::F64, Value::F64(a), Value::F64(b)) => a >= b,

        _ => return None,
    };
    Some(Value::I32(if result { 1 } else { 0 }))
}

// ============================================================================
// Convert Operation Implementation
// ============================================================================

fn execute_convert_op(op: ConvertOp, from: NumberType, to: NumberType, val: Value) -> Option<Value> {
    Some(match (op, from, to, val) {
        // i64 -> i32 wrap
        (ConvertOp::Wrap, NumberType::I64, NumberType::I32, Value::I64(x)) => {
            Value::I32(x as i32)
        }

        // i32 -> i64 extend
        (ConvertOp::ExtendS, NumberType::I32, NumberType::I64, Value::I32(x)) => {
            Value::I64(x as i64)
        }
        (ConvertOp::ExtendU, NumberType::I32, NumberType::I64, Value::I32(x)) => {
            Value::I64((x as u32) as i64)
        }

        // f32 -> i32 truncate
        (ConvertOp::TruncS, NumberType::F32, NumberType::I32, Value::F32(x)) => {
            Value::I32(x.trunc() as i32)
        }
        (ConvertOp::TruncU, NumberType::F32, NumberType::I32, Value::F32(x)) => {
            Value::I32(x.trunc() as u32 as i32)
        }

        // f64 -> i32 truncate
        (ConvertOp::TruncS, NumberType::F64, NumberType::I32, Value::F64(x)) => {
            Value::I32(x.trunc() as i32)
        }
        (ConvertOp::TruncU, NumberType::F64, NumberType::I32, Value::F64(x)) => {
            Value::I32(x.trunc() as u32 as i32)
        }

        // f32 -> i64 truncate
        (ConvertOp::TruncS, NumberType::F32, NumberType::I64, Value::F32(x)) => {
            Value::I64(x.trunc() as i64)
        }
        (ConvertOp::TruncU, NumberType::F32, NumberType::I64, Value::F32(x)) => {
            Value::I64(x.trunc() as u64 as i64)
        }

        // f64 -> i64 truncate
        (ConvertOp::TruncS, NumberType::F64, NumberType::I64, Value::F64(x)) => {
            Value::I64(x.trunc() as i64)
        }
        (ConvertOp::TruncU, NumberType::F64, NumberType::I64, Value::F64(x)) => {
            Value::I64(x.trunc() as u64 as i64)
        }

        // i32 -> f32/f64 convert (signed)
        (ConvertOp::ConvertS, NumberType::I32, NumberType::F32, Value::I32(x)) => {
            Value::F32(x as f32)
        }
        (ConvertOp::ConvertS, NumberType::I32, NumberType::F64, Value::I32(x)) => {
            Value::F64(x as f64)
        }
        // i32 -> f32/f64 convert (unsigned)
        (ConvertOp::ConvertU, NumberType::I32, NumberType::F32, Value::I32(x)) => {
            Value::F32((x as u32) as f32)
        }
        (ConvertOp::ConvertU, NumberType::I32, NumberType::F64, Value::I32(x)) => {
            Value::F64((x as u32) as f64)
        }

        // i64 -> f32/f64 convert (signed)
        (ConvertOp::ConvertS, NumberType::I64, NumberType::F32, Value::I64(x)) => {
            Value::F32(x as f32)
        }
        (ConvertOp::ConvertS, NumberType::I64, NumberType::F64, Value::I64(x)) => {
            Value::F64(x as f64)
        }
        // i64 -> f32/f64 convert (unsigned)
        (ConvertOp::ConvertU, NumberType::I64, NumberType::F32, Value::I64(x)) => {
            Value::F32((x as u64) as f32)
        }
        (ConvertOp::ConvertU, NumberType::I64, NumberType::F64, Value::I64(x)) => {
            Value::F64((x as u64) as f64)
        }

        // f64 -> f32 demote
        (ConvertOp::Demote, NumberType::F64, NumberType::F32, Value::F64(x)) => {
            Value::F32(x as f32)
        }

        // f32 -> f64 promote
        (ConvertOp::Promote, NumberType::F32, NumberType::F64, Value::F32(x)) => {
            Value::F64(x as f64)
        }

        // Reinterpret
        (ConvertOp::Reinterpret, NumberType::I32, NumberType::F32, Value::I32(x)) => {
            Value::F32(f32::from_bits(x as u32))
        }
        (ConvertOp::Reinterpret, NumberType::F32, NumberType::I32, Value::F32(x)) => {
            Value::I32(x.to_bits() as i32)
        }
        (ConvertOp::Reinterpret, NumberType::I64, NumberType::F64, Value::I64(x)) => {
            Value::F64(f64::from_bits(x as u64))
        }
        (ConvertOp::Reinterpret, NumberType::F64, NumberType::I64, Value::F64(x)) => {
            Value::I64(x.to_bits() as i64)
        }

        _ => return None,
    })
}
